import logging

from cvpods.layers import ShapeSpec
from cvpods.modeling.backbone import Backbone
from cvpods.modeling.backbone.fpn import build_resnet_fpn_backbone
from cvpods.modeling.meta_arch.rcnn import GeneralizedRCNN
from cvpods.modeling.proposal_generator import RPN
from cvpods.modeling.roi_heads import StandardROIHeads
from cvpods.modeling.roi_heads.box_head import FastRCNNConvFCHead
from cvpods.modeling.roi_heads.mask_head import MaskRCNNConvUpsampleHead

logger = logging.getLogger('cvpods')


def build_backbone(cfg, input_shape=None):
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))
    backbone = build_resnet_fpn_backbone(cfg, input_shape)
    assert isinstance(backbone, Backbone)
    return backbone


def build_proposal_generator(cfg, input_shape):
    return RPN(cfg, input_shape)


def build_roi_heads(cfg, input_shape):
    from roi_heads.roi_heads import DisAlignStandardROIHeads
    return DisAlignStandardROIHeads(cfg, input_shape)


def build_box_head(cfg, input_shape):
    return FastRCNNConvFCHead(cfg, input_shape)


def build_mask_head(cfg, input_shape):
    return MaskRCNNConvUpsampleHead(cfg, input_shape)


def build_model(cfg):
    cfg.build_backbone = build_backbone
    cfg.build_proposal_generator = build_proposal_generator
    cfg.build_roi_heads = build_roi_heads
    cfg.build_box_head = build_box_head
    cfg.build_mask_head = build_mask_head

    model = GeneralizedRCNN(cfg)
    if cfg.MODEL.BACKBONE.FREEZE:
        for p in model.backbone.parameters():
            p.requires_grad = False
        # print("froze backbone parameters")

    if cfg.MODEL.PROPOSAL_GENERATOR.FREEZE:
        for p in model.proposal_generator.parameters():
            p.requires_grad = False
        # print("froze proposal generator parameters")

    if cfg.MODEL.ROI_HEADS.FREEZE_FEAT:
        for p in model.roi_heads.parameters():
            p.requires_grad = False
        # print("froze roi_box_head parameters")

    model.roi_heads.box_predictor.logit_scale.requires_grad = True
    model.roi_heads.box_predictor.logit_bias.requires_grad = True
    for p in model.roi_heads.box_predictor.confidence_layer.parameters():
        p.requires_grad = True

    for k, v in model.named_parameters():
        if v.requires_grad:
            logger.warning(
                "'{}' has been set with requires_grad=True ".format(
                    k
                )
            )
    return model
