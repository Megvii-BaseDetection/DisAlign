# Semantic Segmentation with Distribution Alignment

## Setup
1. Install mmcv-full
2. Clone mmsegmentation via git and install it
2. Download the checkpoints of pre-trained model from the mmsegmentation
3. Prepare the dataset of ADE20K

## Training
Please edit the `exps/setup_env.sh` to use your own path

1. FCN model
```
source exps/setup_env.sh

bash exps/ade20k_fcn_disalign/disalign_fcn_r50-d8_512x512_160k_ade20k.sh
```

## Model Zoo

- Baseline Results

```bash
# ResNet-50 Backbone
bash exps/fcn_r50-d8_512x512_160k_ade20k.sh
# ResNet-101 Backbone
bash exps/fcn_r101-d8_512x512_160k_ade20k.sh
# ResNeSt-101 Backbone
bash exps/fcn_s101-d8_512x512_160k_ade20k.sh
```

| Method        |AugTest|  mIoU |  mAcc | mHeadIoU | mBodyIoU | mTailIoU | mHeadAcc | mBodyAcc | mTailAcc |Log|
|---------------|-------|------|-------|----------|----------|----------|----------|----------|----------|---|
|FCN-R50-D8-160K| False | 36.1 | 45.41 |  62.53   |  38.12   |  27.58   |  76.88   |  48.82   |  34.51   ||
|FCN-R50-D8-160K| True   |38.08 | 46.27 |  64.64   |  39.95   |  29.62   |  78.64   |   49.3   |  35.41   ||
|FCN-R101-D8-160K| False | 39.91 | 49.62 |  65.28   |  41.96   |  31.65   |  79.14   |  52.58   |  39.58   ||
|FCN-R101-D8-160K| True  | 41.4 | 50.21 |  66.97   |  43.32   |  33.17   |  80.61   |  52.88   |  40.15   ||
|FCN-S101-D8-160K| False | 45.62 | 57.76 |   66.6   |  47.54   |  38.63   |  78.77   |  62.14   |  48.94   ||
|FCN-S101-D8-160K| True | 46.16 | 57.34 |  67.56   |  47.99   |  39.12   |  79.37   |  61.73   |  48.24   ||

- DisAlign Results
```bash
# TODO: Updated
```