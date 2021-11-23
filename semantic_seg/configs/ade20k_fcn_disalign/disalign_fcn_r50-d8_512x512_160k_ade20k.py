import os
from mmcv import Config

mmseg_home = os.environ["MMSEG_HOME"]

cfg = Config.fromfile(os.path.join(
    mmseg_home,
    'configs/fcn/fcn_r50-d8_512x512_160k_ade20k.py'
    )
)
# runtime settings
cfg.runner = dict(type='IterBasedRunner', max_iters=4000)
cfg.checkpoint_config = dict(by_epoch=False, interval=800)
cfg.evaluation = dict(interval=800, metric='mIoU', pre_eval=True)

# model
cfg.model.decode_head.type="DisAlignFCNHead"
cfg.model.decode_head.loss_decode=dict(
    type="GRWCrossEntropyLoss",
    use_sigmoid=False,
    loss_weight=1.0,
    class_weight="./data/ade/ADEChallengeData2016/objectInfo150.txt",
    exp_scale=0.2
)

cfg.model.auxiliary_head.loss_decode.loss_weight=0.0

# dataset 
cfg.data.val.type="ADE20KLTDataset"
cfg.data.test.type="ADE20KLTDataset"
cfg.data.train.type="ADE20KLTDataset"

