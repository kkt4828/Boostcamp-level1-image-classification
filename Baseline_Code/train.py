import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MaskBaseDataset
from dataset import CustomMaskDataset
from loss import create_criterion
from dataset import Sampler
import wandb
from sklearn.metrics import f1_score

from sklearn.metrics import ConfusionMatrixDisplay


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)  # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = int(np.ceil(n ** 0.5))
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def train(model_dir, args):
    seed_everything(args.seed)

    # save_dir = increment_path(os.path.join(model_dir, args.name))
    save_dir = os.path.join(model_dir, args.name)
    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: MaskBaseDataset
    dataset = dataset_module(val_ratio=args.test_size)
    num_classes = dataset.num_classes  # 18

    # -- augmentation
    transform_module1 = getattr(import_module("dataset"), args.augmentation1)  # default: MyAugmenation1
    transform_module2 = getattr(import_module("dataset"), args.augmentation2)  # default: MyAugmenation2
    transform1 = transform_module1(
        prob=0.2,
        resize=args.resize
    )
    transform2 = transform_module2(
        prob=0.2,
        resize=args.resize
    )
    dataset.set_transform(transform1, transform2)

    # -- data_loader

    train_set_list, val_set_list = dataset.split_dataset(args.kfolds, args.seed)
    for i in range(1, args.kfolds):
        train_set = train_set_list[i]
        val_set = val_set_list[i]
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            num_workers=multiprocessing.cpu_count() // 2,
            shuffle=False,
            sampler=Sampler(train_set),
            pin_memory=use_cuda,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_set,
            batch_size=args.valid_batch_size,
            num_workers=multiprocessing.cpu_count() // 2,
            shuffle=False,
            pin_memory=use_cuda,
            drop_last=True,
        )

        # -- model
        if args.model == 'BaseModel':
            model_module = getattr(import_module("model"), args.model)  # default: BaseModel
            model = model_module(
                num_classes=num_classes
            ).to(device)
            model = torch.nn.DataParallel(model)
        elif args.model == 'MyModel':
            model_module = getattr(import_module('model'), args.model)
            model = model_module(num_classes=num_classes).to(device)
            model = torch.nn.DataParallel(model)
        elif args.model == 'Resnet152' or args.model == 'Densenet121':
            model_module = getattr(import_module('model'), args.model)
            model = model_module(num_classes=num_classes).to(device)
            model = torch.nn.DataParallel(model)

        # -- loss & metric
        if args.criterion == 'cross_entropy':
            if args.weight_loss == True:
                weight_list = [0.007555536607057687,
                               0.02658967690560686,
                               0.023173126241757932,
                               0.005666652455293265,
                               0.01178406135589395,
                               0.016140037343481208,
                               0.03777768303528843,
                               0.13294838452803429,
                               0.11586563120878968,
                               0.028333262276466323,
                               0.05892030677946975,
                               0.08070018671740603,
                               0.03777768303528843,
                               0.13294838452803429,
                               0.11586563120878968,
                               0.028333262276466323,
                               0.05892030677946975,
                               0.08070018671740603]
                weight_list = torch.tensor(weight_list).to(device)
                criterion = create_criterion(args.criterion,
                                             label_smoothing=0.1,
                                             weight = weight_list)  # default: cross_entropy
            else:
                criterion = create_criterion(args.criterion,
                                             label_smoothing=0.1
                                             )
        else:
            criterion = create_criterion(args.criterion)
        if args.optimizer == 'SGD':
            opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
            optimizer = opt_module(
                filter(lambda p: p.requires_grad, model.parameters()),
                # model.parameters(),
                lr=args.lr,
                weight_decay=1e-4,
                momentum=0.9
            )
        else:
            opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
            optimizer = opt_module(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr,
                weight_decay=5e-4,
            )
        scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

        # -- logging
        if args.logging == 'tensorboard':
            logger = SummaryWriter(log_dir=save_dir)
            with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
                json.dump(vars(args), f, ensure_ascii=False, indent=4)

            best_val_acc = 0
            best_val_loss = np.inf
            for epoch in range(args.epochs):
                # train loop
                model.train()
                loss_value = 0
                matches = 0
                for idx, train_batch in enumerate(train_loader):
                    inputs, labels = train_batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    outs = model(inputs)
                    preds = torch.argmax(outs, dim=-1)
                    loss = criterion(outs, labels)

                    loss.backward()
                    optimizer.step()

                    loss_value += loss.item()
                    matches += (preds == labels).sum().item()
                    if (idx + 1) % args.log_interval == 0:
                        train_loss = loss_value / args.log_interval
                        train_acc = matches / args.batch_size / args.log_interval
                        current_lr = get_lr(optimizer)
                        print(
                            f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                            f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                        )
                        logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                        logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)

                        loss_value = 0
                        matches = 0

                scheduler.step()

                # val loop
                with torch.no_grad():
                    print("Calculating validation results...")
                    model.eval()
                    val_loss_items = []
                    val_acc_items = []
                    figure = None
                    for val_batch in val_loader:
                        inputs, labels = val_batch
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        outs = model(inputs)
                        preds = torch.argmax(outs, dim=-1)

                        loss_item = criterion(outs, labels).item()
                        acc_item = (labels == preds).sum().item()
                        val_loss_items.append(loss_item)
                        val_acc_items.append(acc_item)

                        if figure is None:
                            inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                            inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                            figure = grid_image(
                                inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                            )

                    val_loss = np.sum(val_loss_items) / len(val_loader)
                    val_acc = np.sum(val_acc_items) / len(val_set)
                    best_val_loss = min(best_val_loss, val_loss)
                    if val_acc > best_val_acc:
                        print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                        torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                        best_val_acc = val_acc
                    torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
                    print(
                        f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                        f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
                    )
                    logger.add_scalar("Val/loss", val_loss, epoch)
                    logger.add_scalar("Val/accuracy", val_acc, epoch)
                    logger.add_figure("results", figure, epoch)
                    print()
        else:
            config = {'Num_Epochs': args.epochs, 'lr_rate': args.lr, 'optimizer': args.optimizer, 'seed': args.seed,
                      'Lossfn': args.criterion}
            wandb.init(project='test_kkt', entity='hbage', group=f'{args.name}_kkt_{args.ver}', config=config)
            wandb.run.name = f'{args.name}_kkt_{args.ver}_{i}'
            # with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
            #     json.dump(vars(args), f, ensure_ascii=False, indent=4)

            best_val_acc = 0
            best_val_loss = np.inf
            best_val_f1 = 0
            for epoch in range(args.epochs):
                # train loop
                model.train()
                loss_value = 0
                matches = 0

                for idx, train_batch in enumerate(train_loader):
                    inputs, labels = train_batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    outs = model(inputs)
                    preds = torch.argmax(outs, dim=-1)
                    loss = criterion(outs, labels)

                    loss.backward()
                    optimizer.step()

                    loss_value += loss.item()
                    matches += (preds == labels).sum().item()
                    if (idx + 1) % args.log_interval == 0:
                        train_loss = loss_value / args.log_interval
                        train_acc = matches / args.batch_size / args.log_interval
                        train_f1 = f1_score(labels.to('cpu').numpy(),
                                            preds.to('cpu').numpy(),
                                            average='macro', zero_division=0)
                        current_lr = get_lr(optimizer)
                        print(
                            f"Fold{i} Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                            f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || training f1 {train_f1:4.4} || lr {current_lr}"
                        )

                        wandb.log({'train-loss': train_loss, 'train-acc': train_acc, 'current_lr': current_lr,
                                   'train-f1': train_f1})

                        loss_value = 0
                        matches = 0

                scheduler.step()

                # val loop
                with torch.no_grad():
                    print("Calculating validation results...")
                    model.eval()
                    val_loss_items = []
                    val_acc_items = []
                    val_pred_items = []
                    val_label_items = []
                    figure = None
                    for val_batch in val_loader:
                        inputs, labels = val_batch
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        outs = model(inputs)
                        preds = torch.argmax(outs, dim=-1)

                        loss_item = criterion(outs, labels).item()
                        acc_item = (labels == preds).sum().item()
                        val_loss_items.append(loss_item)
                        val_acc_items.append(acc_item)
                        for pred, label in zip(torch.clone(preds).detach().to('cpu').numpy(),
                                               torch.clone(labels).detach().to('cpu').numpy()):
                            val_pred_items.append(pred)
                            val_label_items.append(label)

                        if figure is None:
                            inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                            inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                            figure = grid_image(
                                inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                            )

                    val_loss = np.sum(val_loss_items) / len(val_loader)
                    val_acc = np.sum(val_acc_items) / len(val_set)
                    val_f1 = f1_score(val_pred_items, val_label_items, average='macro', zero_division=0)
                    # best_val_loss = min(best_val_loss, val_loss)
                    if val_f1 > best_val_f1:
                        print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                        print(f"New best model's val f1 score : {val_f1:4.4}! saving the best model..")
                        print(f"New best model's val loss : {val_loss:4.4}! saving the best model..")
                        torch.save(model.module.state_dict(), f"{save_dir}/best{args.ver}_{i}.pth")
                        best_val_f1 = val_f1
                        best_val_acc = val_acc
                        best_val_loss = val_loss
                    torch.save(model.module.state_dict(), f"{save_dir}/last{args.ver}_{i}.pth")
                    print(
                        f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.4}, f1 : {val_f1:4.4} || "
                        f"best model acc : {best_val_acc:4.2%}, best model loss: {best_val_loss:4.2}, best model f1 : {best_val_f1:4.4}"
                    )
                    wandb.log({'Val-loss': val_loss, 'Val-accuracy': val_acc, 'Val-f1': val_f1})

                    # if epoch == args.epochs - 1:
                    #     # cm = sns.heatmap(confusion_matrix(val_label_items, val_pred_items) / sum(confusion_matrix(val_label_items, val_pred_items)), vmin=0,
                    #     #                  vmax=1, cbar=False, cmap='Reds', annot=True)
                    #     cm = ConfusionMatrixDisplay.from_predictions(y_pred=val_pred_items, y_true=val_label_items, normalize='all', colorbar=False, cmap = 'Blues', include_values=True)
                    #     wandb.log({f'Val-confusion-matrix-{i}' : cm})

                    print()
            wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 1)')
    parser.add_argument('--kfolds', type=int, default=1, help='number of kfolds to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='CustomMaskDataset',
                        help='dataset augmentation type (default: CustomMaskDataset)')
    parser.add_argument('--augmentation1', type=str, default='MyAugmenation1',
                        help='data augmentation type (default: MyAugmenation1)')
    parser.add_argument('--augmentation2', type=str, default='MyAugmenation2',
                        help='data augmentation type (default: MyAugmenation2)')
    parser.add_argument("--resize", nargs="+", type=list, default=[224, 224],
                        help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=1000,
                        help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='MyModel', help='model type (default: MyModel)')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='cross_entropy',
                        help='criterion type (default: cross_entropy)')
    parser.add_argument('--weight_loss', type=bool, default=False, help='If True, add weight_list created by image path & label dataframe to cross_entropy parameter')
    parser.add_argument('--lr_decay_step', type=int, default=20,
                        help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--logging', type=str, default='wandb',
                        help='choose logging method wandb or tensorboard(defalut : wandb)')
    parser.add_argument('--log_interval', type=int, default=20,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--test_size', default=0.2, help='test size for train,test split (default : 0.2')
    # Container environment
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--ver', type=str, default='ver0.0')

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(
        # data_dir,
        model_dir,
        args)
