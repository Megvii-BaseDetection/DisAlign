from cvpods.layers import ShapeSpec
from cvpods.modeling.backbone import Backbone
from cvpods.modeling.backbone.fpn import build_resnet_fpn_backbone
from cvpods.modeling.meta_arch.rcnn import GeneralizedRCNN
from cvpods.modeling.proposal_generator import RPN
from cvpods.modeling.roi_heads import StandardROIHeads
from cvpods.modeling.roi_heads.box_head import FastRCNNConvFCHead
from cvpods.modeling.roi_heads.mask_head import MaskRCNNConvUpsampleHead


def build_backbone(cfg, input_shape=None):
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))
    backbone = build_resnet_fpn_backbone(cfg, input_shape)
    assert isinstance(backbone, Backbone)
    return backbone


def build_proposal_generator(cfg, input_shape):
    return RPN(cfg, input_shape)


def build_roi_heads(cfg, input_shape):
    from cvpods.modeling.roi_heads.fast_rcnn import CosineFastRCNNOutputLayers
    roi_head = StandardROIHeads(cfg, input_shape)
    
    roi_head.box_predictor = CosineFastRCNNOutputLayers(
        roi_head.box_head.output_size,
        roi_head.num_classes,
        roi_head.cls_agnostic_bbox_reg,
        scale_mode=cfg.MODEL.ROI_BOX_HEAD.COSINE_SCALE_MODE,
        scale_init=cfg.MODEL.ROI_BOX_HEAD.COSINE_SCALE_INIT
    )
    return roi_head


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
    return model
