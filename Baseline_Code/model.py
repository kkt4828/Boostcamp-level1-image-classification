import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models


class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.models = torchvision.models.efficientnet_b0(pretrained=True)
        self.models.classifier[1] = nn.Linear(1280, num_classes)

    def forward(self, x):

        return self.models(x)

class Resnet152(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.models = torchvision.models.resnet152(pretrained=True)
        self.models.fc = nn.Sequential(nn.Linear(in_features=2048, out_features=1024),
                                       nn.LeakyReLU(inplace=True),
                                       nn.Linear(in_features=1024, out_features=num_classes))

    def forward(self, x):
        return self.models(x)

class Densenet121(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.models = torchvision.models.densenet121(pretrained=True)
        self.models.classifier = nn.Sequential(nn.Linear(in_features=1024, out_features=256),
                                       nn.LeakyReLU(inplace=True),
                                       nn.Linear(in_features=256, out_features=num_classes))

    def forward(self, x):
        return self.models(x)




# class MyModel(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()
#         self.models = torchvision.models.efficientnet_b0(pretrained=True)
#         self.models.classifier[1] = nn.Linear(1280, num_classes)
#
#         """
#         1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
#         2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
#         3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
#         """
#
#
#     def forward(self, x):
#         """
#         1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
#         2. 결과로 나온 output 을 return 해주세요
#         """
#         return self.models(x)