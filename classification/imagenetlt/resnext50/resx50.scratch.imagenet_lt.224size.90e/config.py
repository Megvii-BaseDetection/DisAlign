import os.path as osp
import torchvision.transforms as transforms

from cvpods.configs.base_classification_config import BaseClassificationConfig

_config_dict = dict(
    GLOBAL=dict(
        DUMP_TEST=True,
    ),
    MODEL=dict(
        # PIXEL_MEAN=[0.406, 0.456, 0.485],  # BGR
        # PIXEL_STD=[0.225, 0.224, 0.229],
        WEIGHTS="",
        AS_PRETRAIN=True,  # Automatically convert ckpt to pretrain pkl
        RESNETS=dict(
            DEPTH=50,
            NUM_CLASSES=1000,
            STRIDE_IN_1X1=False,  # default true for msra models
            NUM_GROUPS=32,
            WIDTH_PER_GROUP=4,
            NORM="BN",
            ZERO_INIT_RESIDUAL=True,  # default false, use true for all subsequent models
            BOTTLENECK_WIDTH=2, # for resnext model
            OUT_FEATURES=["linear"],
        ),
    ),
    DATASETS=dict(
        TRAIN=("imagenetlt_train",),
        # TEST=("imagenetlt_val",), # for valiation
        TEST=("imagenetlt_test",),
        JSON_PATH="./imagenetlt/category_frequency.json",
    ),
    DATALOADER=dict(
        NUM_WORKERS=12,
    ),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            NAME="WarmupCosineLR",
            MAX_EPOCH=90,
            WARMUP_ITERS=0,
            # use epoch wise lr scheduler
            EPOCH_WISE=True,
        ),
        OPTIMIZER=dict(
            BASE_LR=0.1,
            WEIGHT_DECAY=0.0005,
            WEIGHT_DECAY_NORM=0.0005,
        ),
        CHECKPOINT_PERIOD=10,
        IMS_PER_BATCH=256,
    ),
    INPUT=dict(
        # FORMAT="BGR",
        AUG=dict(
            TRAIN_PIPELINES=[
                ("Torch_RRC", transforms.RandomResizedCrop(224)),
                ("Torch_RHF", transforms.RandomHorizontalFlip()),
                ("Torch_CJ", transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)),
            ],
            TEST_PIPELINES=[
                ("Torch_R", transforms.Resize(256)),
                ("Torch_CC", transforms.CenterCrop(224)),
            ]
        )
    ),
    TEST=dict(
        EVAL_PERIOD=10,
    ),
    OUTPUT_DIR=osp.join(
        '/data/Outputs/model_logs/cvpods_playground',
        osp.split(osp.realpath(__file__))[0].split("playground/")[-1]),
)


class ClassificationConfig(BaseClassificationConfig):
    def __init__(self):
        super(ClassificationConfig, self).__init__()
        self._register_configuration(_config_dict)


config = ClassificationConfig()
