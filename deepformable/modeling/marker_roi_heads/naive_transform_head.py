# Copyright (c) Mustafa B. Yaldiz (VCLAB, KAIST) All Rights Reserved.
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.structures import ImageList, Instances
from detectron2.config import configurable
from detectron2.layers import Conv2d, Linear, ShapeSpec, get_norm
from detectron2.utils.registry import Registry
from detectron2.modeling.poolers import ROIPooler

__all__ = ["NaiveTransformHead", "build_transform_head", "ROI_TRANSFORM_HEAD_REGISTRY"]

ROI_TRANSFORM_HEAD_REGISTRY = Registry("ROI_TRANSFORM_HEAD")
ROI_TRANSFORM_HEAD_REGISTRY.__doc__ = """
Registry for transform heads, which transforms features into normalized
space for corner and class prediction.
"""

def build_transform_head(cfg, input_shape):
    """
    Build a box head defined by `cfg.MODEL.ROI_BOX_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_TRANSFORM_HEAD.NAME
    return ROI_TRANSFORM_HEAD_REGISTRY.get(name)(cfg, input_shape)


@ROI_TRANSFORM_HEAD_REGISTRY.register()
class NaiveTransformHead(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        in_features,
        pooler: ROIPooler,
        conv_dims: List[int],
        fc_dims: List[int], 
        conv_norm=""
    ):
        super().__init__()
        assert len(conv_dims) + len(fc_dims) > 0

        self.in_features = in_features
        self.pooler = pooler

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

        self.fcs = []
        for k, fc_dim in enumerate(fc_dims):
            fc = Linear(np.prod(output_size), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            output_size = fc_dim
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)
        self._output_size = output_size

    @classmethod
    def from_config(cls, cfg, input_shape):
        # TODO: Create new parameters for transform head in the config
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
        }

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ):
        features = [features[f] for f in self.in_features]
        x = self.pooler(features, [p.proposal_boxes for p in proposals])
        
        for layer in self.conv_norm_relus:
            x = layer(x)
        if len(self.fcs):
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            for layer in self.fcs:
                x = F.relu(layer(x))
        return x, x, None, {}

    @property
    def output_shape(self):
        """
        Returns:
            ShapeSpec: the output feature shape
        """
        out = ShapeSpec(channels=self._output_size)
        return out, out