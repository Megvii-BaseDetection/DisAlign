# Classification Experiments

## Dataset Preparation

<details>
<summary>ImageNet-LT</summary>
- Download ImageNet Full dataset 
- ImageNet-LT list and frequency file:
```
# train/val/test split
https://drive.google.com/drive/folders/1j7Nkfe6ZhzKFXePHdsseeeGI877Xu1yf

# Frequency File
wget https://github.com/Megvii-BaseDetection/DisAlign/releases/download/LVIS/imagenet_lt_category_frequency.json
```

```bash
cvpods/datasets/imagenetlt
├── imagenet_lt_category_frequency.json
├── ImageNet_LT_test.txt
├── ImageNet_LT_train.txt
├── ImageNet_LT_val.txt
├── train/
└── val/
```

</details>

<details>
<summary>iNaturalist2018</summary>

</details>

<details>
<summary>Places-LT</summary>

</details>


## Model Zoo


### ImageNet-LT

| Name                                                          | Cls Norm | Input Size | Epoch |   Top-1 Acc(Test) | Top-5 Acc(Test)  | Trained Model                             |
| ------------------------------------------------------------ | ---------- | ---------- | --------   | ------ | ------- | ----------------------------------------- |
| [ResNet-50](imagenetlt/resnet50/res50.scratch.imagenet_lt.224size.90e)        |       | 224  | 90  |   |    | [LINK(TODO)]() |
| [ResNet-50](imagenetlt/resnet50/res50.scratch.imagenet_lt.224size.cosine.90e) | Cosine| 224  | 90  |   |    | [LINK(TODO)](model_final.pth) |
| [ResNeXt-50](imagenetlt/resnext50/resx50.scratch.imagenet_lt.224size.90e)        |       | 224  | 90  | 46.918  | 73.088  | [LINK](https://github.com/Megvii-BaseDetection/DisAlign/releases/tag/LVIS#:~:text=resx50.scratch.imagenet_lt.224size.90e.model_final_plain.pth) |
| [ResNeXt-50](imagenetlt/resnext50/resx50.scratch.imagenet_lt.224size.cosine.90e) | Cosine| 224  | 90  |   |   | [LINK(TODO)](model_final.pth) |
| [ResNet-50+DisAlign](imagenetlt/resnet50/res50.scratch.imagenet_lt.224size.90e.disalign.10e)        |       | 224  | 90  |   |    | [LINK(TODO)](model_final.pth) |
| [ResNet-50+DisAlign](imagenetlt/resnet50/res50.scratch.imagenet_lt.224size.cosine.90e.disalign.10e) | Cosine| 224  | 90  |   |    | [LINK(TODO)](model_final.pth) |
| [ResNeXt-50+DisAlign](imagenetlt/resnext50/resx50.scratch.imagenet_lt.224size.90e.disalign.10e)        |       | 224  | 90  | 52.438  | 77.068  | [LINK](https://github.com/Megvii-BaseDetection/DisAlign/releases/download/LVIS/resx50.scratch.imagenet_lt.224size.90e.disalign.10e.model_final_plain.pth) |
| [ResNeXt-50+DisAlign](imagenetlt/resnext50/resx50.scratch.imagenet_lt.224size.cosine.90e.disalign.10e) | Cosine| 224  | 90  |   |    | [LINK(TODO)](model_final.pth) |




