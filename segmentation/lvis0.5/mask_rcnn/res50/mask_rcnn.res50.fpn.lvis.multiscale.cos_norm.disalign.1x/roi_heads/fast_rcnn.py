import torch
from torch.nn import functional as F

import json
from cvpods.modeling.roi_heads.fast_rcnn import FastRCNNOutputs


class GRWFastRCNNOutputs(FastRCNNOutputs):
    def __init__(
        self, box2box_transform, pred_class_logits, pred_proposal_deltas, proposals, smooth_l1_beta,
        grw_scale
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform/Box2BoxTransformRotated):
                box2box transform instance for proposal-to-detection transformations.
            pred_class_logits (Tensor): A tensor of shape (R, K + 1) storing the predicted class
                logits for all R predicted object instances.
                Each row corresponds to a predicted object instance.
            pred_proposal_deltas (Tensor): A tensor of shape (R, K * B) or (R, B) for
                class-specific or class-agnostic regression. It stores the predicted deltas that
                transform proposals into final box detections.
                B is the box dimension (4 or 5).
                When B is 4, each row is [dx, dy, dw, dh (, ....)].
                When B is 5, each row is [dx, dy, dw, dh, da (, ....)].
            proposals (list[Instances]): A list of N Instances, where Instances i stores the
                proposals for image i, in the field "proposal_boxes".
                When training, each Instances must have ground-truth labels
                stored in the field "gt_classes" and "gt_boxes".
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
            grw_scale (int): Scale of the Generalized Re-weight Loss
        """
        super().__init__(
            box2box_transform, pred_class_logits, pred_proposal_deltas, proposals, smooth_l1_beta
        )
        self.ce_loss_exp_reweight = self._init_weights(exp_scale=grw_scale)

    def _init_weights(self, exp_scale=0.7):
        import os
        import cvpods
        import numpy as np
        folder_name = "/".join(cvpods.__path__[0].split('/')[:-1])
        num_samples_list = np.load(os.path.join(folder_name, "datasets/lvis/num_shots_v0.5.npy"))

        num_foreground = len(num_samples_list)
        assert num_foreground > 0, "num_samples_list is empty"
        # assert exp_scale <= 1.0 and exp_scale >= 0, "exp_scale must less than 1.0 and "

        num_shots = num_samples_list
        ratio_list = num_shots / np.sum(num_shots)
        exp_reweight =  1 / (ratio_list**exp_scale)

        exp_reweight = exp_reweight / np.sum(exp_reweight) * num_foreground
        exp_reweight = torch.tensor(exp_reweight).float()
        final_reweight = torch.ones(num_foreground+1)
        final_reweight[:-1] = exp_reweight

        return final_reweight

    def softmax_cross_entropy_loss(self):
        """
        Compute the softmax cross entropy loss for box classification.

        Returns:
            scalar Tensor
        """
        if self._no_instances:
            return 0.0 * self.pred_class_logits.sum()
        else:
            self._log_accuracy()
            return F.cross_entropy(self.pred_class_logits, 
                                self.gt_classes, 
                                weight=self.ce_loss_exp_reweight.to(self.pred_class_logits.device),
                                reduction="mean")
