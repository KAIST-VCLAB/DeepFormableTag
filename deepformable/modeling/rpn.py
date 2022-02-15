"""
This code is modified from detectron2 implementation, 
to add adaptive loss to region proposal network.
Copyright (c) Mustafa B. Yaldiz (VCLAB, KAIST) All Rights Reserved.
"""
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn

from fvcore.nn import giou_loss
from detectron2.layers import cat, ShapeSpec
from detectron2.modeling import PROPOSAL_GENERATOR_REGISTRY
from detectron2.structures import Boxes
from detectron2.utils.events import get_event_storage
from detectron2.modeling.proposal_generator import RPN

from deepformable.layers import AdaptiveLoss


@PROPOSAL_GENERATOR_REGISTRY.register()
class RPN_AdaptiveLoss(RPN):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__(cfg, input_shape)
        adaptive_loss = cfg.MODEL.PROPOSAL_GENERATOR.ADAPTIVE_LOSS
        self.bbox_loss_function = AdaptiveLoss(loss_type='l1') if adaptive_loss else nn.L1Loss(reduction='sum')
        self.class_loss_function = AdaptiveLoss(loss_type='bce') if adaptive_loss else nn.BCELoss(reduction='sum')
    
    @torch.jit.unused
    def losses(
        self,
        anchors: List[Boxes],
        pred_objectness_logits: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        pred_anchor_deltas: List[torch.Tensor],
        gt_boxes: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Return the losses from a set of RPN predictions and their associated ground-truth.

        Args:
            anchors (list[Boxes or RotatedBoxes]): anchors for each feature map, each
                has shape (Hi*Wi*A, B), where B is box dimension (4 or 5).
            pred_objectness_logits (list[Tensor]): A list of L elements.
                Element i is a tensor of shape (N, Hi*Wi*A) representing
                the predicted objectness logits for all anchors.
            gt_labels (list[Tensor]): Output of :meth:`label_and_sample_anchors`.
            pred_anchor_deltas (list[Tensor]): A list of L elements. Element i is a tensor of shape
                (N, Hi*Wi*A, 4 or 5) representing the predicted "deltas" used to transform anchors
                to proposals.
            gt_boxes (list[Tensor]): Output of :meth:`label_and_sample_anchors`.

        Returns:
            dict[loss name -> loss value]: A dict mapping from loss name to loss value.
                Loss names are: `loss_rpn_cls` for objectness classification and
                `loss_rpn_loc` for proposal localization.
        """
        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)  # (N, sum(Hi*Wi*Ai))

        # Log the number of positive/negative anchors per-image that's used in training
        pos_mask = gt_labels == 1
        num_pos_anchors = pos_mask.sum().item()
        num_neg_anchors = (gt_labels == 0).sum().item()
        storage = get_event_storage()
        storage.put_scalar("rpn/num_pos_anchors", num_pos_anchors / num_images)
        storage.put_scalar("rpn/num_neg_anchors", num_neg_anchors / num_images)

        if self.box_reg_loss_type == "smooth_l1":
            anchors = type(anchors[0]).cat(anchors).tensor  # Ax(4 or 5)
            gt_anchor_deltas = [self.box2box_transform.get_deltas(anchors, k) for k in gt_boxes]
            gt_anchor_deltas = torch.stack(gt_anchor_deltas)  # (N, sum(Hi*Wi*Ai), 4 or 5)
            # ====CHANGE_ON_LOSS====
            localization_loss = self.bbox_loss_function(
                cat(pred_anchor_deltas, dim=1)[pos_mask], gt_anchor_deltas[pos_mask])
            # localization_loss = smooth_l1_loss(
            #     cat(pred_anchor_deltas, dim=1)[pos_mask],
            #     gt_anchor_deltas[pos_mask],
            #     self.smooth_l1_beta,
            #     reduction="sum",
            # )
            # ======================
        elif self.box_reg_loss_type == "giou":
            pred_proposals = self._decode_proposals(anchors, pred_anchor_deltas)
            pred_proposals = cat(pred_proposals, dim=1)
            pred_proposals = pred_proposals.view(-1, pred_proposals.shape[-1])
            pos_mask = pos_mask.view(-1)
            localization_loss = giou_loss(
                pred_proposals[pos_mask], cat(gt_boxes)[pos_mask], reduction="sum"
            )
        else:
            raise ValueError(f"Invalid rpn box reg loss type '{self.box_reg_loss_type}'")

        valid_mask = gt_labels >= 0
        # ====CHANGE_ON_LOSS====
        objectness_loss = self.class_loss_function(
            cat(pred_objectness_logits, dim=1)[valid_mask],
            gt_labels[valid_mask].to(torch.float32)
        )
        # ======================
        normalizer = self.batch_size_per_image * num_images
        return {
            "loss_rpn_cls": objectness_loss / normalizer,
            "loss_rpn_loc": localization_loss / normalizer,
        }