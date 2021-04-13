# Instructions for LVIS Dataset

## Prepare Dataset

Please put the dataset and annotation into the `cvpods` project as following:
```
.
└── datasets
    ├── coco 
    │     ├── annotations
    │     ├── train2017
    │     └── val2017
    ├── lvis
    │     ├── lvis_v0.5_train.json
    │     ├── lvis_v0.5_val.json
    │     ├── lvis_v1_train.json
    │     └── lvis_v1_val.json
    └── .....
```
## Training and Inference
- Enter one project folder 
- Traning with:
```
pods_train --num-gpus 8
```
- Inference with:
```
pods_test --num-gpus 8
```


## Model Zoo

(* denotes the cosine classifier)

### LVIS V0.5(Mask R-CNN)

> We refactor the code of the internal version and re-train all experiments, the performance results have a little difference(higher) with the reported in the original paper.

#### ResNet-50

| Name                                                          | Cls Norm | input size | lr sched | train time (s/iter) | train mem (GB) | box AP | mask AP | Trained Model                             |
| ------------------------------------------------------------ | ---------- | ---------- | -------- | ------------------- | -------------- | ------ | ------- | ----------------------------------------- |
| [MaskRCNN-R50-FPN](lvis0.5/mask_rcnn.res50.fpn.lvis.multiscale.1x) | |640-800        | 90k      | 0.486               | 5.26           | 20.4   | 20.7    | [LINK](model_final.pth) |
| [MaskRCNN-R50-FPN](lvis0.5/mask_rcnn.res50.fpn.lvis.multiscale.cos_norm.1x) | Cosine|640-800        | 90k      | 0.500               | 5.26           | 23.0   | 23.8    | [LINK](model_final.pth) |
| [MaskRCNN-R50-FPN-RFS](lvis0.5/mask_rcnn.res50.fpn.lvis.multiscale.rfs.1x) | |640-800    | 90k      | 0.485               | 5.25           | 23.5   | 24.2    | [LINK](model_final.pth) |
| [MaskRCNN-R50-FPN-RFS](lvis0.5/mask_rcnn.res50.fpn.lvis.multiscale.rfs.1x) | Cosine| 640-800    | 90k      | 0.485               | 5.25           |  24.5   | 24.9   | [LINK](model_final.pth) |
| [MaskRCNN-R50-FPN-DisAlign](lvis0.5/mask_rcnn.res50.fpn.lvis.multiscale.disalign.1x) | | 640-800        | 90k      | 0.486               | 5.26           |  23.7  | 24.3    | [LINK](model_final.pth) |
| [MaskRCNN-R50-FPN-DisAlign](lvis0.5/mask_rcnn.res50.fpn.lvis.multiscale.cos_norm.disalign.1x) | Cosine|  640-800        | 90k      | 0.500               | 5.26    |  26.3 | 27.1  | [LINK](model_final.pth) |
| [MaskRCNN-R50-FPN-RFS-DisAlign](lvis0.5/mask_rcnn.res50.fpn.lvis.multiscale.cos_norm.disalign.1x) | Cosine| 640-800        | 90k      | 0.500               | 5.26    |  27.1 | 27.5  | [LINK](model_final.pth) |

#### ResNet-101

| Name                                                           | Cls Norm | input size | lr sched | train time (s/iter) | train mem (GB) | box AP | mask AP | Trained Model                             |
| ------------------------------------------------------------ | ---------- | ---------- | -------- | ------------------- | -------------- | ------ | ------- | ----------------------------------------- |
| [MaskRCNN-R101-FPN](lvis0.5/mask_rcnn.res50.fpn.lvis.multiscale.1x) |  |640-800        | 90k      |           |            | 22.6   |  22.8   | [LINK](model_final.pth) |
| [MaskRCNN-R101-FPN](lvis0.5/mask_rcnn.res50.fpn.lvis.multiscale.1x) | Cosine |640-800        | 90k      |           |            | 24.8   |  25.3   | [LINK](model_final.pth) |
| [MaskRCNN-R101-FPN-RFS](lvis0.5/mask_rcnn.res50.fpn.lvis.multiscale.1x) | Cosine |640-800        | 90k      |           |            | 26.6   |  26.8   | [LINK](model_final.pth) |
| [MaskRCNN-R101-FPN-DisAlign](lvis0.5/mask_rcnn.res50.fpn.lvis.multiscale.1x) | | 640-800        | 90k      |           |            | 25.9    | 26.2   | [LINK](model_final.pth) |
| [MaskRCNN-R101-FPN-DisAlign](lvis0.5/mask_rcnn.res50.fpn.lvis.multiscale.1x) | Cosine|640-800        | 90k      |           |            | 27.6    | 28.1    | [LINK](model_final.pth) |
| [MaskRCNN-R101-FPN-RFS-DisAlign](lvis0.5/mask_rcnn.res50.fpn.lvis.multiscale.1x) | Cosine |640-800        | 90k      |           |            |  28.7  | 28.9    | [LINK](model_final.pth) |

#### ResNeXt-101

| Name                                                         | Cls Norm | input size | lr sched | train time (s/iter) | train mem (GB) | box AP | mask AP | Trained Model                             |
| ------------------------------------------------------------ | ---------- | ---------- | -------- | ------------------- | -------------- | ------ | ------- | ----------------------------------------- |
| [MaskRCNN-X101-FPN](lvis0.5/mask_rcnn.res50.fpn.lvis.multiscale.1x) | |640-800        | 90k      |           |            |  24.8  | 25.2   | [LINK](model_final.pth) |
| [MaskRCNN-X101-FPN](lvis0.5/mask_rcnn.res50.fpn.lvis.multiscale.1x) | Cosine| 640-800        | 90k      |           |            | 27.4   | 28.4   | [LINK](model_final.pth) |
| [MaskRCNN-X101-FPN-DisAlign](lvis0.5/mask_rcnn.res50.fpn.lvis.multiscale.1x) | | 640-800        | 90k      |           |            |   26.9 | 27.3   | [LINK](model_final.pth) |
| [MaskRCNN-X101-FPN-DisAlign](lvis0.5/mask_rcnn.res50.fpn.lvis.multiscale.1x) | Cosine | 640-800        | 90k      |           |          | 29.6   | 30.2   | [LINK](model_final.pth) |

#### Cascade R-CNN



### LVIS 1.0(Mask R-CNN)

#### ResNet-50
| Name                                                         | Cls Norm| input size | lr sched | train time (s/iter) | train mem (GB) | box AP | mask AP | Trained Model                             |
| ------------------------------------------------------------ | ---------- | ---------- | -------- | ------------------- | -------------- | ------ | ------- | ----------------------------------------- |
| [MaskRCNN-R50-FPN](lvis1.0/mask_rcnn.res50.fpn.lvis.multiscale.1x) | |640-800        | 180k      | 0.486               | 5.26           | 18.8   | 18.3    | [LINK](model_final.pth) |
| [MaskRCNN-R50-FPN](lvis1.0/mask_rcnn.res50.fpn.lvisv1.multiscale.cos_norm.1x) | Cosine | 640-800        | 180k      | 0.500               | 5.26           | 21.3   | 21.1    | [LINK](model_final.pth) |
| [MaskRCNN-R50-FPN-RFS](lvis1.0/mask_rcnn.res50.fpn.lvisv1.multiscale.rfs.1x) | |640-800    | 180k      | 0.485               | 5.25           | 22.9   | 22.5    | [LINK](model_final.pth) |
| MaskRCNN-R50-FPN-RFS(A1)  || 640-800    | 180k      |             |        |     | 22.3    | |
| [MaskRCNN-R50-FPN-DisAlign](lvis1.0/mask_rcnn.res50.fpn.lvis.multiscale.1x) | |640-800        | 180k      | 0.486               | 5.26           | 21.9  | 21.3   | [LINK](model_final.pth) |
| [MaskRCNN-R50-FPN-DisAlign](lvis1.0/mask_rcnn.res50.fpn.lvisv1.multiscale.cos_norm.1x)| Cosine | 640-800        | 180k      | 0.500               | 5.26           |  24.8  | 24.2  | [LINK](model_final.pth) |


#### ResNet-101

#### ResNeXt-101

- A1: Evaluating Large-Vocabulary Object Detectors: The Devil is in the Details