# Copyright (c) Mustafa B. Yaldiz (VCLAB, KAIST) All Rights Reserved.
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.structures import ImageList, Instances

from detectron2.modeling import ROI_HEADS_REGISTRY, ROIHeads

from .naive_transform_head import build_transform_head
from .corner_head import build_corner_head
from .decoder_head import build_decoder_head



@ROI_HEADS_REGISTRY.register()
class MarkerROIHeads(ROIHeads):
    """
    This class implements the corner prediction and decoding tasks.
    It returns a dictionary of outputs that later converted to
    instances after postprocessing.
    """
    @configurable
    def __init__(
        self,
        *,
        transform_head: nn.Module,
        corner_head: nn.Module,
        decoder_head: nn.Module,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.transform_head = transform_head
        self.corner_head = corner_head
        self.decoder_head = decoder_head

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg)
        transform_head = build_transform_head(cfg, input_shape)
        ret["transform_head"] = transform_head
        corner_input_shape, decoder_input_shape = transform_head.output_shape
        ret["corner_head"] = build_corner_head(cfg, corner_input_shape)
        ret["decoder_head"] = build_decoder_head(cfg, decoder_input_shape)
        return ret

    @property
    def device(self):
        return self.corner_head.device
    
    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Dict], Dict]:
        # del images
        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        # del targets
        
        if self.training:
            corner_features, decoding_features, sample_locations_batch, losses = self.transform_head(images, features, proposals, targets)
            losses.update(self.corner_head(corner_features, proposals))
            losses.update(self.decoder_head(decoding_features, proposals))
            del images, targets
            return [], losses
        
        corner_features, decoding_features, sample_locations_batch, _ = self.transform_head(images, features, proposals, targets)
        corners_batch = self.corner_head(corner_features, proposals)
        obj_scores_batch, decoded_messages_batch = self.decoder_head(decoding_features, proposals)

        results = []
        for i in range(len(proposals)):
            output = {
                "corners": corners_batch[i], "obj_scores": obj_scores_batch[i],
                "decoded_messages": decoded_messages_batch[i],
                "image_shape": proposals[i].image_size}
            if sample_locations_batch:
                output["sample_locations"] = sample_locations_batch[i]
            results.append(output)

        return results, {}