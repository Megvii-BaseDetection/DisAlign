import logging

from cvpods.layers import ShapeSpec
from cvpods.modeling.backbone import Backbone
from cvpods.modeling.backbone import build_resnet_backbone
from cvpods.modeling.meta_arch.imagenet import Classification
from cvpods.layers import DisAlignLinear

def build_backbone(cfg, input_shape=None):
    """
    Build a backbone from `cfg.MODEL.BACKBONE.NAME`.

    Returns:
        an instance of :class:`Backbone`
    """
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

    backbone = build_resnet_backbone(cfg, input_shape)
    # replace the nn.Linear  with DisAlignLinear
    in_feature = backbone.linear.in_features
    out_feature = backbone.linear.out_features
    backbone._modules.pop("linear")
    backbone.linear = DisAlignLinear(
        in_features=in_feature,
        out_features=out_feature,
    )

    assert isinstance(backbone, Backbone)
    return backbone


def build_model(cfg):

    cfg.build_backbone = build_backbone

    model = Classification(cfg)
    import json
    import cvpods
    import os.path as osp
    from cvpods.modeling.losses import GRWCrossEntropyLoss
    from cvpods.utils import PathManager
    
    data_root = osp.join(
        osp.split(osp.split(cvpods.__file__)[0])[0], "datasets")
    longtail_json = osp.join(data_root, cfg.DATASETS.JSON_PATH)
    with PathManager.open(longtail_json, 'r') as f:
        category_frequency = json.load(f)
    
    model.loss_evaluator = GRWCrossEntropyLoss(
        num_samples_list=category_frequency['train_shots'],
        num_classes=cfg.MODEL.RESNETS.NUM_CLASSES,
        exp_scale=cfg.MODEL.GRW_SCALE
    )
    # set grad flag for stage-2 traning.
    for p in model.parameters():
        p.requires_grad = False
    
    model.network.linear.logit_scale.requires_grad = True
    model.network.linear.logit_bias.requires_grad = True
    for p in model.network.linear.confidence_layer.parameters():
        p.requires_grad = True
    
    model.loss_evaluator.to(model.device)

    logger = logging.getLogger("cvpods")
    logger.info("Model:\n{}".format(model))
    for name, p in model.named_parameters():
        if p.requires_grad:
            logger.info("{} is set requires_gread=True\n".format(name))
    return model
