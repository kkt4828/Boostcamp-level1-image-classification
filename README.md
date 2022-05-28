## Project Overview
+ 코로나 확산에 따른 일상생활에서 마스크 착용이 의무화가 되면서 감염을 막기위해서는 코, 입을 올바르게 착용하는 것이 중요한 상황
+ 공공장소에서 이를 검사하려면 추가적인 인적자원이 필요
+ 카메라에 비춰진 얼굴 이미지로 마스크 착용여부와 올바르게 쓴게 맞는지 가려내는 시스템이 필요  

## 최종 결과
>Public f1 Score : 0.7568

>Private f1 Score : 0.7514

>최종 순위 : 11/48

## Objective
성별, 연령, 마스크 착용 여부에 따라 사진을 총 18개의 클래스로 분류

## How to Solve

EDA를 통해 데이터를 분석하고 분석한 결과를 바탕으로 데이터 전처리를 진행했다. 그 후 적절한 모델을 선정하여 학습시키고 다양한 기법들을 활용하여 성능을 증가시켜나갔다.  
자세한 내용은 Wrap up Report에 기술해 두었다.

## Data Overview
|class|mask|gender|age|
|---|---|---|---|
|0|Wear|Male|<30|
|1|Wear|Male|>=30 and <60|
|2|Wear|Male|>=60|
|3|Wear|Female|<30|
|4|Wear|Female|>=30 and <60|
|5|Wear|Female|>=60|
|6|Incorrect|Male|<30|
|7|Incorrect|Male|>=30 and <60|
|8|Incorrect|Male|>=60|
|9|Incorrect|Female|<30|
|10|Incorrect|Female|>=30 and <60|
|11|Incorrect|Female|>=60|
|12|Not Wear|Male|<30|
|13|Not Wear|Male|>=30 and <60|
|14|Not Wear|Male|>=60|
|15|Not Wear|Female|<30|
|16|Not Wear|Female|>=30 and <60|
|17|Not Wear|Female|>=60|  


## Explanation
+ **Best directory**  
학습된 모델을 모아놓은 directory

>모델 설명  

| 모델 이름 | 설명 |
| :--------: | :-------- |
| CoAtNet_81.pth | **CoAtNet-b0** transformation4. imbalancedsampler. label에 따라 다른 augmentation. label smoothing. |
| mask_model.pt | **EfficientNet-b0** transformation4. imbalancedsampler. label smoothing|
| Resnet_80.pth | **ResNet18** transformation4. labelsmoothing. imbalancedsampler. label에 따른 augmentation. <58 label change |
| Resnet_mixup.pth | **ResNet18** transformation4. imbalancedsampling. Mixup. |


## 활용 기법

### EDA

- pandas,matplotlib,seaborn등의 라이브러리를 활용했다.
- 나이,성별 : Seaborn의 CountPlot 활용했다.
- 마스크 착용 여부 : Seaborn의 BarPlot 활용했다.

+ **EDA, Augmentation, Sampler.ipynb**  
EDA, Augmentation 기법 적용, Sampler 분석 노트북


### Augmentation
- ToGray: 흑백 효과를 추가하여 머리색이 나이를 판단하는 기준에 머리색뿐이 아닌 주름살같은 피부요인도 고려해보게끔 유도하였다.
- CourseDropOut: 사진에 임의의 hole을 발생시켜서 학습효과 향상을 기대하였다.
- ChannelDropOut: 임의의 색 변화를 통해 다양한 요인에서 학습을 시켰다.
- MultiplicativeNoise: Noise 효과를 줘서 화질이 조금 안좋은 사진에서도 학습시켰다.
- HorizontalFlip: 거꾸로 된 사람은 없기 때문에 좌우 반전만 시도했다.
- CenterCrop: 사람의 안면 부분만 따로 빼내기 위해 사용했다.
- Resize: 모델의 크기에 맞게 조정하기 위해 사용했다.
- Normalize: 사진의 색을 최대한 원본색에 가깝게 하기위해 사용했다.

### Class Imbalance & Data Labeling
- ImbalancedSampler 기법을 사용하여 under sampling과 over sampling을 적절히 섞었다.
- Smooth labeling 기법을 사용하여 one-hot encoding을 방지했다.
- Cross Entropy대신 Focal Loss를 활용했다.
- 60세 이상의 자료가 부족했기 때문에 57세 이상부터 60대로 분류했다.

+ **Preprocessing.ipynb**  
Data Pipeline 용 코드


+ **Ensemble.ipynb**  
Best directory에 있는 학습된 모델들을 soft-voting을 통해 valid 및 submission file을 만들 수 있는  코드
