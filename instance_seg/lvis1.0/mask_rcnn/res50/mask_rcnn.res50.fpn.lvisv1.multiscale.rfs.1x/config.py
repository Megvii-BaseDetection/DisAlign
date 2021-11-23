import os.path as osp

from cvpods.configs.rcnn_fpn_config import RCNNFPNConfig

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="detectron2://ImageNetPretrained/MSRA/R-50.pkl",
        MASK_ON=True,
        RESNETS=dict(DEPTH=50),
        ROI_HEADS=dict(
            NUM_CLASSES=1203,
            # SCORE_THRESH_TEST=0,
            SCORE_THRESH_TEST=0.0001,
        )
    ),
    DATASETS=dict(
        TRAIN=("lvis_v1_train",),
        TEST=("lvis_v1_val",),
    ),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            STEPS=(120000, 160000),
            MAX_ITER=180000, # 180000 * 16 / 100000 ~ 28.8 epochs
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
    DATALOADER=dict(
        SAMPLER_TRAIN="RepeatFactorTrainingSampler",
        REPEAT_THRESHOLD=0.001,
    ),
    TEST=dict(
        DETECTIONS_PER_IMAGE=300,
    ),
    OUTPUT_DIR=osp.join(
        '/data/Outputs/model_logs/cvpods_playground',
        osp.split(osp.realpath(__file__))[0].split("playground_disalign/")[-1]),
)


class MaskRCNNConfig(RCNNFPNConfig):
    def __init__(self):
        super(MaskRCNNConfig, self).__init__()
        self._register_configuration(_config_dict)


config = MaskRCNNConfig()
