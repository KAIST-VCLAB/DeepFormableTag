# Copyright (c) Mustafa B. Yaldiz (VCLAB, KAIST) All Rights Reserved.
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import fvcore.nn.weight_init as weight_init
from fvcore.nn import smooth_l1_loss
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.structures import ImageList, Instances
from detectron2.config import configurable
from detectron2.layers import Conv2d, Linear, ShapeSpec, get_norm, cat, nonzero_tuple

from detectron2.modeling.poolers import ROIPooler

from .naive_transform_head import ROI_TRANSFORM_HEAD_REGISTRY
from .corner_head import PointsPredictor
from deepformable.layers import AdaptiveLoss


__all__ = ["SpatialTransformerHead", "SpatialTransformerHeadV2"]



@ROI_TRANSFORM_HEAD_REGISTRY.register()
class SpatialTransformerHead(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        in_features,
        pooler: ROIPooler,
        conv_dims: List[int],
        fc_dims: List[int], 
        conv_norm="",
        num_classes: int,
        transformer_resolution: int,
        affine_predictor: bool = False,
        smooth_l1_beta: float = 0.0,
        loss_weight: Union[float, Dict[str, float]] = 1.0,
    ):
        super().__init__()
        assert len(conv_dims) + len(fc_dims) > 0

        self.in_features = in_features
        self.pooler = pooler
        self.num_classes = num_classes

        output_size = (input_shape.channels, input_shape.height, input_shape.width)

        self.conv_norm_relus = []
        for k, conv_dim in enumerate(conv_dims):
            conv = Conv2d(
                output_size[0],
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            output_size = (conv_dim, output_size[1], output_size[2])
        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        self._decoding_output_size = (output_size[0], transformer_resolution, transformer_resolution)

        self.fcs = []
        for k, fc_dim in enumerate(fc_dims):
            fc = Linear(np.prod(output_size), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            output_size = fc_dim
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)
        self._corner_output_size = output_size

        self.transformer_resolution = transformer_resolution
        self.affine_predictor = None
        if affine_predictor:
            self.affine_predictor = Linear(output_size, 3 * 2)
            self.affine_predictor.weight.data.zero_()
            self.affine_predictor.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32))
        else:
            self.sample_predictor = PointsPredictor(
                output_size, num_points=transformer_resolution*transformer_resolution,  std=0.001)
        # self.smooth_l1_beta = smooth_l1_beta
        self.sampling_loss_function = AdaptiveLoss(loss_type='l1')
        self.loss_weight = loss_weight

    @classmethod
    def from_config(cls, cfg, input_shape):
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_TRANSFORM_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_TRANSFORM_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_TRANSFORM_HEAD.POOLER_TYPE

        num_conv = cfg.MODEL.ROI_TRANSFORM_HEAD.NUM_CONV
        conv_dim = cfg.MODEL.ROI_TRANSFORM_HEAD.CONV_DIM
        num_fc = cfg.MODEL.ROI_TRANSFORM_HEAD.NUM_FC
        fc_dim = cfg.MODEL.ROI_TRANSFORM_HEAD.FC_DIM

        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        return {
            "input_shape": ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution),
            "in_features": in_features,
            "pooler": pooler,
            "conv_dims": [conv_dim] * num_conv,
            "fc_dims": [fc_dim] * num_fc,
            "conv_norm": cfg.MODEL.ROI_TRANSFORM_HEAD.NORM,
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "affine_predictor": cfg.MODEL.ROI_TRANSFORM_HEAD.AFFINE_PREDICTOR_ON,
            "transformer_resolution": cfg.MODEL.ROI_TRANSFORM_HEAD.TRANSFORMER_RESOLUTION,
            "smooth_l1_beta"        : cfg.MODEL.ROI_CORNER_HEAD.SMOOTH_L1_BETA,
            "loss_weight"           : cfg.MODEL.ROI_TRANSFORM_HEAD.LOSS_WEIGHT,
        }

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals_sampled: List[Instances],
        targets: Optional[List[Instances]] = None,
    ):
        features = [features[f] for f in self.in_features]
        proposal_boxes = [p.proposal_boxes for p in proposals_sampled]
        x = self.pooler(features, proposal_boxes)

        for layer in self.conv_norm_relus:
            x = layer(x)
        conv_features = x

        if len(self.fcs):
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            for layer in self.fcs:
                x = F.relu(layer(x))
        corner_features = x
        
        if self.affine_predictor:
            theta = self.affine_predictor(x)
            theta = theta.view(-1, 2, 3)
            sample_point_deltas = F.affine_grid(
                theta, 
                (x.shape[0], 1, self.transformer_resolution, self.transformer_resolution),
                align_corners=False)
            sample_point_deltas = sample_point_deltas.view(-1, self.transformer_resolution*self.transformer_resolution, 2)
        else:
            sample_point_deltas = self.sample_predictor(x)

        sample_point_deltas = torch.clamp(sample_point_deltas, min=-2.0, max=2.0)       # Make sure things dont become NAN

        box_type = type(proposals_sampled[0].proposal_boxes)
        proposal_boxes_cat = box_type.cat(proposal_boxes)
        box_sizes = proposal_boxes_cat.tensor[:,2:] - proposal_boxes_cat.tensor[:,:2]
        box_centers = proposal_boxes_cat.get_centers()
        
        loss_sample_reg = {}
        if self.training:
            gt_classes = cat([p.gt_classes for p in proposals_sampled], dim=0)
            fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < self.num_classes))[0]
            gt_fg_sample_locs = cat([p.gt_sample_locs for p in proposals_sampled], dim=0)[fg_inds]
            
            predicted_fg_sample_deltas = sample_point_deltas[fg_inds]
            fg_box_centers, fg_box_sizes = box_centers[fg_inds], box_sizes[fg_inds]

            gt_deltas = (gt_fg_sample_locs - fg_box_centers.view(-1,1,2)) / (fg_box_sizes.view(-1,1,2) * 0.5)
            loss_sample_reg = self.sampling_loss_function(predicted_fg_sample_deltas, gt_deltas
                ) * self.loss_weight / (gt_classes.numel() * 2.0 * self.transformer_resolution*self.transformer_resolution)
            # loss_sample_reg = {"loss_sample_reg": smooth_l1_loss(
            #     predicted_fg_sample_deltas,
            #     gt_deltas,
            #     self.smooth_l1_beta,
            #     reduction="sum"
            # ) * self.loss_weight / (gt_classes.numel() * 2.0 * self.transformer_resolution*self.transformer_resolution)}
            loss_sample_reg = {"loss_sample_reg": loss_sample_reg}
        
        sample_point_deltas = sample_point_deltas.view(
            -1, self.transformer_resolution, self.transformer_resolution, 2)

        transformed_features = F.grid_sample(
            conv_features, sample_point_deltas,
            mode='bilinear', padding_mode='border',
            align_corners=False)

        return corner_features, transformed_features, None, loss_sample_reg

    @property
    def output_shape(self):
        """
        Returns:
            ShapeSpec: the output feature shape
        """
        o1 = self._corner_output_size
        o2 = self._decoding_output_size
        out = ShapeSpec(channels=o1), ShapeSpec(channels=o2[0], height=o2[1], width=o2[2])
        return out



@ROI_TRANSFORM_HEAD_REGISTRY.register()
class SpatialTransformerHeadV2(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        in_features,
        pooler: ROIPooler,
        num_classes: int,
        stem_channels: int,
        transformer_resolution: int,
        corner_sample_resolution: int,
        conv_dims: List[int] = [256],
        fc_common_dims: List[int] = [256],
        fc_corner_dims: List[int] = [256],
        fc_resample_dims: List[int] = [256],
        loss_weight: Union[float, Dict[str, float]] = 1.0,
        include_sample_predictions=False,
    ):
        super().__init__()
        assert len(conv_dims) + len(fc_common_dims) > 0

        self.in_features = in_features
        self.pooler = pooler
        self.num_classes = num_classes
        self.include_sample_predictions = include_sample_predictions

        output_size = (input_shape.channels, input_shape.height, input_shape.width)

        self.conv_norm_relus = []
        for k, conv_dim in enumerate(conv_dims):
            conv = Conv2d(
                output_size[0],
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=True,
                norm=get_norm("", conv_dim),
                activation=F.relu,
            )
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            output_size = (conv_dim, output_size[1], output_size[2])
        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        self._decoding_output_size = (output_size[0], transformer_resolution, transformer_resolution)

        self.fc_common = []
        for k, fc_dim in enumerate(fc_common_dims):
            fc = Linear(np.prod(output_size), fc_dim)
            self.add_module("fc_common{}".format(k + 1), fc)
            self.fc_common.append(fc)
            output_size = fc_dim
        for layer in self.fc_common:
            weight_init.c2_xavier_fill(layer)
        fc_common_output_size = output_size
        self._corner_output_size = (stem_channels, corner_sample_resolution, corner_sample_resolution)

        # Corner prediction head
        self.fc_corner = []
        for k, fc_dim in enumerate(fc_corner_dims):
            fc = Linear(np.prod(output_size), fc_dim)
            self.add_module("fc_corner{}".format(k + 1), fc)
            self.fc_corner.append(fc)
            output_size = fc_dim
        for layer in self.fc_corner:
            weight_init.c2_xavier_fill(layer)
        fc_corner_output_size = output_size

        self.affine_predictor = Linear(fc_corner_output_size, 4 * 2 * 3)
        self.affine_predictor.weight.data.zero_()
        self.affine_predictor.bias.data.copy_(
            torch.tensor(
            [
                 1, 0, 0, 0,  1, 0,
                -1, 0, 0, 0,  1, 0,
                -1, 0, 0, 0, -1, 0,
                 1, 0, 0, 0, -1, 0
            ], dtype=torch.float32))

        output_size = fc_common_output_size
        self.fc_resample = []
        for k, fc_dim in enumerate(fc_resample_dims):
            fc = Linear(np.prod(output_size), fc_dim)
            self.add_module("fc_resample{}".format(k + 1), fc)
            self.fc_resample.append(fc)
            output_size = fc_dim
        for layer in self.fc_resample:
            weight_init.c2_xavier_fill(layer)

        self.transformer_resolution = transformer_resolution

        self.sample_predictor = PointsPredictor(
            output_size, num_points=transformer_resolution*transformer_resolution,  std=0.001)
        
        self.sampling_loss_function = AdaptiveLoss(loss_type='l1')
        self.loss_weight = loss_weight

    @classmethod
    def from_config(cls, cfg, input_shape):
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_TRANSFORM_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_TRANSFORM_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_TRANSFORM_HEAD.POOLER_TYPE

        num_conv = cfg.MODEL.ROI_TRANSFORM_HEAD.NUM_CONV
        conv_dim = cfg.MODEL.ROI_TRANSFORM_HEAD.CONV_DIM

        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        return {
            "input_shape": ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution),
            "in_features": in_features,
            "pooler": pooler,
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "stem_channels": cfg.MODEL.RESNETS.STEM_OUT_CHANNELS,
            "transformer_resolution": cfg.MODEL.ROI_TRANSFORM_HEAD.TRANSFORMER_RESOLUTION,
            "corner_sample_resolution": cfg.MODEL.ROI_CORNER_HEAD.SAMPLE_RESOLUTION,
            "conv_dims": [conv_dim] * num_conv,
            "fc_common_dims": cfg.MODEL.ROI_TRANSFORM_HEAD.FC_COMMON_DIMS,
            "fc_corner_dims": cfg.MODEL.ROI_TRANSFORM_HEAD.FC_CORNER_DIMS,
            "fc_resample_dims": cfg.MODEL.ROI_TRANSFORM_HEAD.FC_RESAMPLE_DIMS,
            "loss_weight": cfg.MODEL.ROI_TRANSFORM_HEAD.LOSS_WEIGHT,
            "include_sample_predictions": False,
        }

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals_sampled: List[Instances],
        targets: Optional[List[Instances]] = None,
    ):
        # Make sure low-level features are passed
        # stem_features = features['stem'] if 'stem' in features else images.tensor
        stem_features = features['stem']

        features = [features[f] for f in self.in_features]
        proposal_boxes = [p.proposal_boxes for p in proposals_sampled]

        box_type = type(proposals_sampled[0].proposal_boxes)
        proposal_boxes_cat = box_type.cat(proposal_boxes)
        box_sizes = proposal_boxes_cat.tensor[:,2:] - proposal_boxes_cat.tensor[:,:2]
        box_centers = proposal_boxes_cat.get_centers()

        x = self.pooler(features, proposal_boxes)
        
        for layer in self.conv_norm_relus:
            x = layer(x)
        conv_features = x

        if len(self.fc_common):
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            for layer in self.fc_common:
                x = F.relu(layer(x))
        fc_common_features = x

        if len(self.fc_corner):
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            for layer in self.fc_corner:
                x = F.relu(layer(x))
        fc_corner_features = x

        # Calculate corner samples
        theta = self.affine_predictor(x)
        theta = theta.view(-1, 4, 2, 3)
        corner_sample_resolution = self._corner_output_size[1]
        index, device = 0, box_sizes.device
        last_row = torch.tensor([0,0,1], device=device)
        norm_factor = torch.tensor(images.tensor.shape[3:1:-1], device=device)
        thetas, corner_deltas = [], []
        for p in proposals_sampled:
            box_sizes_i = box_sizes[index:index+len(p)] / norm_factor
            box_centers_i = box_centers[index:index+len(p)] / (norm_factor*0.5) - 1.0
            tx = torch.stack([box_sizes_i[:,0], torch.zeros_like(box_sizes_i[:,0]), box_centers_i[:,0]], dim=-1)
            ty = torch.stack([torch.zeros_like(box_sizes_i[:,1]), box_sizes_i[:,1], box_centers_i[:,1]], dim=-1)
            t = torch.stack([tx, ty], dim=1).unsqueeze(1)
            theta_i = theta[index:index+len(p)]
            theta_i = torch.cat([theta_i, last_row.expand(theta_i.size(0), 4, 1, 3)], dim=2)
            theta_transformed = torch.matmul(t, theta_i)
            corner_deltas.append(
                F.affine_grid(
                    theta_transformed.view(-1, 2, 3), 
                    (theta_transformed.shape[0]*4, 1, corner_sample_resolution, corner_sample_resolution),
                    align_corners=False))
            thetas.append(theta_transformed)
            index += len(p)
        
        thetas = torch.cat(thetas, dim=0).view(-1,4,2,3)
        batchable = all([i.size(0)==corner_deltas[0].size(0) for i in corner_deltas])
        if batchable:
            corner_deltas = torch.stack(corner_deltas, dim=0).view(
                -1, theta_transformed.shape[0]*4, corner_sample_resolution*corner_sample_resolution, 2)
            corner_features = F.grid_sample(
                stem_features,
                corner_deltas,
                mode='bilinear', padding_mode='border',
                align_corners=False).permute(0,2,1,3).contiguous().view(
                    -1, 4, self._corner_output_size[0], corner_sample_resolution, corner_sample_resolution)
        else:
            corner_features = []
            for stem_features_i, corner_delta in zip(stem_features, corner_deltas):
                corner_features.append(F.grid_sample(
                    stem_features_i.expand(corner_delta.shape[0], *stem_features_i.shape),
                    corner_delta,
                    mode='bilinear', padding_mode='border',
                    align_corners=False).view(-1, 4, self._corner_output_size[0], corner_sample_resolution, corner_sample_resolution))
            corner_features = torch.cat(corner_features, dim=0)
        corner_features = corner_features, thetas, norm_factor

        x = fc_common_features
        if len(self.fc_resample):
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            for layer in self.fc_resample:
                x = F.relu(layer(x))
        fc_resample_features = x

        sample_point_deltas = self.sample_predictor(fc_resample_features)
        sample_point_deltas = torch.clamp(sample_point_deltas, min=-2.0, max=2.0)       # Make sure things dont become NAN
        
        loss_sample_reg = {}
        if self.training:
            gt_classes = cat([p.gt_classes for p in proposals_sampled], dim=0)
            fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < self.num_classes))[0]
            gt_fg_sample_locs = cat([p.gt_sample_locs for p in proposals_sampled], dim=0)[fg_inds]
            
            predicted_fg_sample_deltas = sample_point_deltas[fg_inds]
            fg_box_centers, fg_box_sizes = box_centers[fg_inds], box_sizes[fg_inds]

            gt_deltas = (gt_fg_sample_locs - fg_box_centers.view(-1,1,2)) / (fg_box_sizes.view(-1,1,2) * 0.5)
            loss_sample_reg = self.sampling_loss_function(predicted_fg_sample_deltas, gt_deltas
                ) * self.loss_weight / (gt_classes.numel() * 2.0 * self.transformer_resolution*self.transformer_resolution)
            loss_sample_reg = {"loss_sample_reg": loss_sample_reg}
        
        sample_prediction_batches = None
        if self.include_sample_predictions and not self.training:
            sample_predictions = sample_point_deltas * box_sizes.view(-1,1,2) * 0.5 + box_centers.view(-1,1,2)
            i, sample_prediction_batches = 0, []
            for p in proposals_sampled:
                data_len = len(p.proposal_boxes)
                sample_prediction_batches.append(sample_predictions[i:i+data_len])
                i += data_len

        sample_point_deltas = sample_point_deltas.view(
            -1, self.transformer_resolution, self.transformer_resolution, 2)

        transformed_features = F.grid_sample(
            conv_features, sample_point_deltas,
            mode='bilinear', padding_mode='border',
            align_corners=False)

        return corner_features, transformed_features, sample_prediction_batches, loss_sample_reg

    @property
    def output_shape(self):
        """
        Returns:
            ShapeSpec: the output feature shape
        """
        o1 = self._corner_output_size
        o2 = self._decoding_output_size
        out = ShapeSpec(channels=o1[0], height=o1[1], width=o1[2]), ShapeSpec(channels=o2[0], height=o2[1], width=o2[2])
        return out