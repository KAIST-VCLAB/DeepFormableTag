# Copyright (c) Mustafa B. Yaldiz (VCLAB, KAIST) All Rights Reserved.
import numpy as np
from typing import Dict, List, Union
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, Linear, ShapeSpec, cat, nonzero_tuple, get_norm
from detectron2.utils.registry import Registry
from detectron2.structures import Instances

from deepformable.layers import AdaptiveLoss

__all__ = ["CornerHead", "build_corner_head", "ROI_CORNER_HEAD_REGISTRY", "PointsPredictor", "CornerHeadV2"]

ROI_CORNER_HEAD_REGISTRY = Registry("ROI_CORNER_HEAD")
ROI_CORNER_HEAD_REGISTRY.__doc__ = """
Registry for corner heads
"""

def build_corner_head(cfg, input_shape):
    name = cfg.MODEL.ROI_CORNER_HEAD.NAME
    return ROI_CORNER_HEAD_REGISTRY.get(name)(cfg, input_shape)


class PointsPredictor(nn.Module):
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        num_points: int = 4,
        std: float =0.001,
    ):
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        
        self.num_points = num_points
        self.points_predictor = Linear(input_size, num_points*2)  # 4*2 corner predictions require linear layer

        nn.init.normal_(self.points_predictor.weight, std=std)
        nn.init.constant_(self.points_predictor.bias, 0)

    @property
    def device(self):
        return self.points_predictor.weight.device

    def forward(self, x: torch.Tensor):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        return self.points_predictor(x).view(-1,self.num_points,2)


@ROI_CORNER_HEAD_REGISTRY.register()
class CornerHead(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        num_classes: int,
        smooth_l1_beta: float = 0.0,
        loss_weight: Union[float, Dict[str, float]] = 1.0,
    ):
        super().__init__()
        
        # Each point will use regression weight hyper-param
        self.corner_pred = PointsPredictor(input_shape, std=0.0002)
        
        self.num_classes = num_classes
        # self.smooth_l1_beta = smooth_l1_beta
        self.loss_weight = loss_weight
        self.loss_function = AdaptiveLoss(loss_type='l1')

    @property
    def device(self):
        return self.corner_pred.device

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "input_shape": input_shape,
            "num_classes"           : cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "smooth_l1_beta"        : cfg.MODEL.ROI_CORNER_HEAD.SMOOTH_L1_BETA,
            "loss_weight"           : cfg.MODEL.ROI_CORNER_HEAD.LOSS_WEIGHT,
        }

    def forward(self, x: torch.Tensor, proposals_sampled: List[Instances]):
        box_type = type(proposals_sampled[0].proposal_boxes)
        proposal_boxes = box_type.cat([p.proposal_boxes for p in proposals_sampled])
        if self.training:
            # gt_boxes = box_type.cat([p.gt_boxes for p in proposals_sampled])
            gt_classes = cat([p.gt_classes for p in proposals_sampled], dim=0)
            fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < self.num_classes))[0]
            gt_corners = cat([p.gt_corners for p in proposals_sampled], dim=0)[fg_inds]
            proposal_boxes = proposal_boxes[fg_inds]
            x = x[fg_inds]

        corner_deltas = self.corner_pred(x)
        corner_deltas = torch.clamp(corner_deltas, -2.0, 2.0)
        box_sizes = proposal_boxes.tensor[:,2:] - proposal_boxes.tensor[:,:2]
        box_centers = proposal_boxes.get_centers()
        
        if self.training:
            gt_deltas = (gt_corners - box_centers.view(-1,1,2)) / box_sizes.view(-1,1,2)
            loss_corner_reg = self.loss_function(
                corner_deltas, gt_deltas) / (gt_classes.numel() * 8.0)
            # loss_corner_reg = smooth_l1_loss(
            #     corner_deltas,
            #     gt_deltas,
            #     self.smooth_l1_beta,
            #     reduction="sum"
            # ) * self.loss_weight / (gt_classes.numel() * 8.0)
            return {"loss_corner_reg": loss_corner_reg * self.loss_weight}
        
        corners = box_centers.view(-1,1,2) + corner_deltas * box_sizes.view(-1,1,2)

        i, corner_batches = 0, []
        for p in proposals_sampled:
            data_len = len(p.proposal_boxes)
            corner_batches.append(corners[i:i+data_len])
            i += data_len

        return corner_batches


@ROI_CORNER_HEAD_REGISTRY.register()
class CornerHeadV2(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        num_classes: int,
        conv_dims: List[int] = [64],
        fc_dims: List[int] = [128, 64],
        loss_weight: Union[float, Dict[str, float]] = 1.0,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        self.input_shape = (input_shape.channels, (input_shape.height or 1), (input_shape.width or 1))
        output_size = self.input_shape

        self.conv_norm_relus = []
        for k, conv_dim in enumerate(conv_dims):
            conv = Conv2d(
                output_size[0],
                conv_dim,
                kernel_size=3,
                padding=0,
                bias=True,
                norm=get_norm("", conv_dim),
                activation=F.relu,
            )
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            output_size = (conv_dim, output_size[1]-2, output_size[2]-2)
        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)

        self.fcs = []
        for k, fc_dim in enumerate(fc_dims):
            fc = Linear(np.prod(output_size), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            output_size = fc_dim
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)
        
        # Each point will use regression weight hyper-param
        self.corner_pred = PointsPredictor(output_size, num_points=1, std=0.0002)
        self.loss_weight = loss_weight
        self.loss_function = AdaptiveLoss(loss_type='l1')

    @property
    def device(self):
        return self.corner_pred.device

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "input_shape"           : input_shape,
            "num_classes"           : cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "conv_dims"             : cfg.MODEL.ROI_CORNER_HEAD.CONV_DIMS,
            "fc_dims"               : cfg.MODEL.ROI_CORNER_HEAD.FC_DIMS,
            "loss_weight"           : cfg.MODEL.ROI_CORNER_HEAD.LOSS_WEIGHT,
        }

    def forward(self, x: torch.Tensor, proposals_sampled: List[Instances]):
        corner_features, thetas, norm_factor = x
        
        if self.training:
            # gt_boxes = box_type.cat([p.gt_boxes for p in proposals_sampled])
            gt_classes = cat([p.gt_classes for p in proposals_sampled], dim=0)
            fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < self.num_classes))[0]
            gt_corners = cat([p.gt_corners for p in proposals_sampled], dim=0)[fg_inds].view(-1,2)
            corner_features = corner_features[fg_inds].view(-1,*self.input_shape)
            thetas = thetas[fg_inds]
        
        x, thetas = corner_features.view(-1,*self.input_shape), thetas.view(-1,2,3)
        for layer in self.conv_norm_relus:
            x = layer(x)

        if len(self.fcs):
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            for layer in self.fcs:
                x = F.relu(layer(x))

        corner_deltas = self.corner_pred(x)
        corner_deltas = torch.clamp(corner_deltas, -2.0, 2.0).view(-1,2)

        corners = torch.cat([corner_deltas, torch.ones(corner_deltas.size(0), 1, device=self.device)], dim=1)
        corners = (torch.matmul(thetas, corners.unsqueeze(-1)) + 1.0).view(-1,2) * (norm_factor*0.5)
        
        if self.training:
            loss_corner_reg = self.loss_function(corners, gt_corners) / (gt_classes.numel() * 8.0)
            return {"loss_corner_reg": loss_corner_reg * self.loss_weight}
        
        corners = corners.view(-1,4,2)
        
        # if self.training:
        #     last_row = torch.tensor([0,0,1], device=self.device)
        #     thetas_w_last = torch.cat([thetas, last_row.expand(thetas.size(0), 1, 3)], dim=1)
        #     thetas_inv = torch.inverse(thetas_w_last)
        #     gt_corners = gt_corners / (norm_factor*0.5) - 1.0
        #     gt_corners = torch.cat(
        #         [gt_corners, torch.ones(gt_corners.size(0), 1, device=self.device)],
        #          dim=1).unsqueeze(-1)
        #     gt_deltas = torch.matmul(thetas_inv, gt_corners)[:, [0,1], 0]
        #     loss_corner_reg = self.loss_function(corner_deltas, gt_deltas) / (gt_classes.numel() * 8.0)
        #     return {"loss_corner_reg": loss_corner_reg}
        
        # corners = torch.cat([corner_deltas, torch.ones(corner_deltas.size(0), 1, device=self.device)], dim=1)
        # corners = (torch.matmul(thetas, corners.unsqueeze(-1)) + 1.0).view(-1,2) * (norm_factor*0.5)
        # corners = corners.view(-1,4,2)

        i, corner_batches = 0, []
        for p in proposals_sampled:
            data_len = len(p.proposal_boxes)
            corner_batches.append(corners[i:i+data_len])
            i += data_len

        return corner_batches
