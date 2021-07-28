# Classification Experiments

## Dataset Preparation

<details>
<summary>ImageNet-LT</summary>

```bash
cvpods/datasets/imagenetlt
├── category_frequency.json
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
| [ResNet-50](imagenetlt/resnet50/res50.scratch.imagenet_lt.224size.90e)        |       | 224  | 90  |   |    | [LINK](model_final.pth) |
| [ResNet-50](imagenetlt/resnet50/res50.scratch.imagenet_lt.224size.cosine.90e) | Cosine| 224  | 90  |   |    | [LINK](model_final.pth) |
| [ResNeXt-50](imagenetlt/resnext50/resx50.scratch.imagenet_lt.224size.90e)        |       | 224  | 90  | 46.918  | 73.088  | [LINK](model_final.pth) |
| [ResNeXt-50](imagenetlt/resnext50/resx50.scratch.imagenet_lt.224size.cosine.90e) | Cosine| 224  | 90  |   |   | [LINK](model_final.pth) |
| [ResNet-50+DisAlign](imagenetlt/resnet50/res50.scratch.imagenet_lt.224size.90e.disalign.10e)        |       | 224  | 90  |   |    | [LINK](model_final.pth) |
| [ResNet-50+DisAlign](imagenetlt/resnet50/res50.scratch.imagenet_lt.224size.cosine.90e.disalign.10e) | Cosine| 224  | 90  |   |    | [LINK](model_final.pth) |
| [ResNeXt-50+DisAlign](imagenetlt/resnext50/resx50.scratch.imagenet_lt.224size.90e.disalign.10e)        |       | 224  | 90  | 52.438  | 77.068  | [LINK](model_final.pth) |
| [ResNeXt-50+DisAlign](imagenetlt/resnext50/resx50.scratch.imagenet_lt.224size.cosine.90e.disalign.10e) | Cosine| 224  | 90  |   |    | [LINK](model_final.pth) |




