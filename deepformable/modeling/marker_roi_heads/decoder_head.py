# Copyright (c) Mustafa B. Yaldiz (VCLAB, KAIST) All Rights Reserved.
import numpy as np
from typing import List
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, Linear, get_norm, ShapeSpec, cat, nonzero_tuple
from detectron2.utils.registry import Registry
from detectron2.structures import Instances

from deepformable.layers import AdaptiveLoss

__all__ = ["DecoderHead", "build_decoder_head", "ROI_DECODER_HEAD_REGISTRY"]

ROI_DECODER_HEAD_REGISTRY = Registry("ROI_DECODER_HEAD")
ROI_DECODER_HEAD_REGISTRY.__doc__ = """
Registry for corner heads
"""

def build_decoder_head(cfg, input_shape):
    name = cfg.MODEL.ROI_DECODER_HEAD.NAME
    return ROI_DECODER_HEAD_REGISTRY.get(name)(cfg, input_shape)



@ROI_DECODER_HEAD_REGISTRY.register()
class DecoderHead(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        num_classes: int,
        num_bits: int,
        conv_dims: List[int],
        conv_norm="",
        fc_dims: List[int], 
        with_decoder: bool = True,
        decoding_loss_type: str = 'mse',
        decoding_loss_weight: float = 1.0,
        class_loss_weight: float = 1.0,
    ):
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        
        output_size = (input_shape.channels, (input_shape.height or 1), (input_shape.width or 1))

        self.conv_norm_relus = []
        for k, conv_dim in enumerate(conv_dims):
            conv = Conv2d(
                output_size[0],
                conv_dim,
                kernel_size=3,
                padding=0,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
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
        
        self.with_decoder = with_decoder
        self.num_classes, self.num_bits = num_classes, num_bits
        if with_decoder:
            output_size = np.prod(output_size)
            self.decoder = Linear(output_size, num_bits)
            nn.init.normal_(self.decoder.weight, std=0.01)
            nn.init.constant_(self.decoder.bias, 0)
            self.cls_score = Linear(output_size, 1)
        else:
            self.cls_score = Linear(output_size, num_classes + 1)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)
        
        self.decoding_loss_func = AdaptiveLoss(loss_type=decoding_loss_type)
        self.decoding_loss_weight = decoding_loss_weight
        self.objectness_loss_func = AdaptiveLoss(loss_type='bce')
        self.class_loss_weight = class_loss_weight

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "input_shape": input_shape,
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "num_bits": cfg.MODEL.MARKER_GENERATOR.NUM_GENERATION_BITS,
            "with_decoder": cfg.MODEL.ROI_DECODER_HEAD.DECODER_ON,
            "decoding_loss_type": cfg.MODEL.ROI_DECODER_HEAD.LOSS_TYPE,
            "decoding_loss_weight": cfg.MODEL.ROI_DECODER_HEAD.DECODING_LOSS_WEIGHT,
            "class_loss_weight": cfg.MODEL.ROI_DECODER_HEAD.CLASS_LOSS_WEIGHT,
            "conv_dims": cfg.MODEL.ROI_DECODER_HEAD.CONV_DIMS,
            "fc_dims": cfg.MODEL.ROI_DECODER_HEAD.FC_DIMS
        }

    def forward(self, x: torch.Tensor, proposals_sampled: List[Instances]):
        if self.training:
            gt_classes = cat([p.gt_classes for p in proposals_sampled], dim=0)
            fg_list = (gt_classes >= 0) & (gt_classes < self.num_classes)
            fg_inds = nonzero_tuple(fg_list)[0]
            gt_objectness = fg_list.to(torch.float32).view(-1,1)

        # Apply conv and relus
        for layer in self.conv_norm_relus:
            x = layer(x)
        
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        
        if len(self.fcs):    
            for layer in self.fcs:
                x = F.relu(layer(x))
        
        obj_scores = self.cls_score(x)
        
        decoded_message = None
        if self.with_decoder:
            if self.training:
                x = x[fg_inds]
            decoded_message = torch.sigmoid(self.decoder(x))

        if self.training:
            if self.with_decoder:
                objectness_loss = self.objectness_loss_func(obj_scores, gt_objectness)
                gt_message = cat([p.gt_message for p in proposals_sampled], dim=0)[fg_inds]
                decoding_loss = self.decoding_loss_func(decoded_message, gt_message)
                div_factor = max((gt_classes.numel() * self.num_bits), 1)
                losses = {
                    'objectness_loss': objectness_loss * self.class_loss_weight / obj_scores.size(0),
                    'decoding_loss': decoding_loss * self.decoding_loss_weight / div_factor
                }
            else:
                # TODO: Modify for adaptive clipping
                loss_cls = F.cross_entropy(
                    obj_scores, gt_classes, reduction="mean") * self.class_loss_weight
                losses = {'loss_cls': loss_cls}
            return losses
        
        i, score_batches, message_batches = 0, [], []
        for p in proposals_sampled:
            data_len = len(p.proposal_boxes)
            score_batches.append(obj_scores[i:i+data_len])
            if self.with_decoder:
                message_batches.append(decoded_message[i:i+data_len])
            else:
                message_batches.append(None)
            i += data_len

        return score_batches, message_batches
