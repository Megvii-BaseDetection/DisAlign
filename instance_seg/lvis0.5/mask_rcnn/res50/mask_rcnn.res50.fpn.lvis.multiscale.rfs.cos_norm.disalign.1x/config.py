import os.path as osp

from cvpods.configs.rcnn_fpn_config import RCNNFPNConfig

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="../mask_rcnn.res50.fpn.lvis.multiscale.rfs.cos_norm.1x/log/plain_model_final.pth",
        MASK_ON=True,
        BACKBONE=dict(FREEZE=True),
        PROPOSAL_GENERATOR=dict(FREEZE=True),
        RESNETS=dict(DEPTH=50),
        ROI_HEADS=dict(
            NUM_CLASSES=1230,
            SCORE_THRESH_TEST=0.0001,
            FREEZE_FEAT=True
        ),
        ROI_BOX_HEAD=dict(
            GRW_SCALE=0.8,
            COSINE_SCALE_MODE='learn',
            COSINE_SCALE_INIT=20.0,
        )
    ),
    DATASETS=dict(
        TRAIN=("lvis_v0.5_train",),
        TEST=("lvis_v0.5_val",),
    ),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            # STEPS=(60000, 80000),
            # MAX_ITER=90000,
            STEPS=(4000, ),
            MAX_ITER=4500,
        ),
        OPTIMIZER=dict(
            BASE_LR=0.02,
        ),
        IMS_PER_BATCH=16,
    ),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                ("ResizeShortestEdge",
                 dict(short_edge_length=(640, 672, 704, 736, 768, 800),
                      max_size=1333, sample_style="choice")),
                ("RandomFlip", dict()),
            ],
            TEST_PIPELINES=[
                ("ResizeShortestEdge",
                 dict(short_edge_length=800, max_size=1333, sample_style="choice")),
            ],
        ),
    ),
    TEST=dict(
        DETECTIONS_PER_IMAGE=300,
        EVAL_PERIOD=1500,
    ),
    OUTPUT_DIR=osp.join(
        '/data/Outputs/model_logs/cvpods_playground',
        osp.split(osp.realpath(__file__))[0].split("playground_disalign/")[-1]),
)


class MaskRCNNConfig(RCNNFPNConfig):
    def __init__(self):
        super(MaskRCNNConfig, self).__init__()
        self._register_configuration(_config_dict)


def check_checkpoint(tgt_ckpt):
    import torch
    src_name = "model_final.pth"    
    tgt_name = "plain_model_final.pth"

    if osp.exists(tgt_ckpt):
        return None
    else:
        src_ckpt = tgt_ckpt.replace(tgt_name, src_name)
        model = torch.load(src_ckpt, map_location='cpu')

        plain_ckpt = {"model":model['model']}
        torch.save(plain_ckpt, tgt_ckpt)
        print("Save the plain checkpoint of the model")

config = MaskRCNNConfig()

check_checkpoint(tgt_ckpt=config.MODEL.WEIGHTS)
