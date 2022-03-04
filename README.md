# [P stage1] 우린아직젊조 
## Project Overview
+ 코로나 확산에 따른 일상생활에서 마스크 착용이 의무화가 되면서 감염을 막기위해서는 코, 입을 올바르게 착용하는 것이 중요한 상황
+ 공공장소에서 이를 검사하려면 추가적인 인적자원이 필요
+ 카메라에 비춰진 얼굴 이미지로 마스크 착용여부와 올바르게 쓴게 맞는지 가려내는 시스템이 필요  


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


## Result
+ F1-score: 0.7514
+ accuracy: 79.9524


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


+ **Ensemble.ipynb**  
Best directory에 있는 학습된 모델들을 soft-voting을 통해 valid 및 submission file을 만들 수 있는  코드
