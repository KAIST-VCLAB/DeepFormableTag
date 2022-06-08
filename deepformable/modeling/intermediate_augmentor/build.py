"""
Copyright (c) Mustafa B. Yaldiz (VCLAB, KAIST) All Rights Reserved.
TODO: 
- Support batch operations for the images. Currently the input is (C, H, W).
"""

import torch
from torch import nn

from detectron2.utils.registry import Registry
from abc import ABCMeta, abstractmethod

INTERMEDIATE_AUGMENTOR_REGISTRY = Registry("INTERMEDIATE_AUGMENTOR") 
INTERMEDIATE_AUGMENTOR_REGISTRY.__doc__ = """
Registry for the differentiable intermediate augmentations after rendering
"""


def build_intermediate_augmentations(cfg):
    """
    Build the intermediate augmentor, defined by ``cfg.INTERMEDIATE_AUGMENTOR``.
    """
    augmentations = []
    for aug_name in cfg.INTERMEDIATE_AUGMENTOR.AUG_LIST:
        aug = INTERMEDIATE_AUGMENTOR_REGISTRY.get(aug_name)(cfg)
        aug.to(torch.device(cfg.MODEL.DEVICE))
        augmentations.append(aug)
     
    return augmentations


class IntermediateAugmentor(nn.Module, metaclass=ABCMeta):
    """
    Abstract base class for intermediate augmentors.
    apply_image transformations needs to be differentiable!
    """
    def __init__(
        self,
        skip_coords=False,
    ):
        super().__init__()
        self.skip_coords = skip_coords
    
    @abstractmethod
    def apply_image(self, image):
        """
        Apply transformation to the images
        """
        pass
    
    def apply_coords(self, coords):
        """
        Apply transformation to the coordinates of the labels
        """
        return coords
    
    def generate_params(self, image, gt_instances, strength=None):
        """
        Generates random numbers required to apply augmentations
        """
        return

    def apply_instances(self, gt_instances):
        if self.skip_coords or gt_instances is None:
            return gt_instances
        
        with torch.no_grad():
            if gt_instances.has("gt_sample_locs"):
                sample_loc_shape = gt_instances.gt_sample_locs.shape
                gt_instances.gt_sample_locs = self.apply_coords(gt_instances.gt_sample_locs.view(-1,2)).view(sample_loc_shape)
            if gt_instances.has("gt_segm"):
                gt_instances.gt_segm = self.apply_coords(gt_instances.gt_segm.view(-1,2)).view(-1,8,2)
            elif gt_instances.has("gt_masks"):
                device, dtype = gt_instances.gt_boxes.device, gt_instances.gt_boxes.tensor.dtype
                polygons = torch.as_tensor(gt_instances.gt_masks.polygons, dtype=dtype, device=device)
                polygons = self.apply_coords(polygons.view(-1,2)).view(-1,8)
                gt_instances.gt_masks.polygons = [[i.cpu().numpy()] for i in polygons]
            elif gt_instances.has("gt_boxes"):
                gt_instances.gt_boxes.tensor = self.apply_coords(gt_instances.gt_boxes.tensor.view(-1,2)).view(-1,4)
            
        return gt_instances
    
    @classmethod
    def fix_instances(cls, gt_instances):
        if gt_instances.has("gt_segm"):
            min_c, max_c = torch.min(gt_instances.gt_segm, dim=1)[0], torch.max(gt_instances.gt_segm, dim=1)[0]
            gt_instances.gt_boxes.tensor = torch.cat([min_c, max_c], dim=1)
            gt_instances.gt_corners = gt_instances.gt_segm[:,[0,2,4,6]]
        elif gt_instances.has("gt_masks"):
            device, dtype = gt_instances.gt_boxes.device, gt_instances.gt_boxes.tensor.dtype
            polygons = torch.as_tensor(gt_instances.gt_masks.polygons, dtype=dtype, device=device).view(-1,4,2)
            min_c, max_c = torch.min(polygons, dim=1)[0], torch.max(polygons, dim=1)[0]
            gt_instances.gt_boxes.tensor = torch.cat([min_c, max_c], dim=1)
        
        # # Convert segmentation to polygon masks
        # segm = gt_instances.gt_segm.flatten(start_dim=1)
        # polygons_per_instance = torch.chunk(segm, segm.shape[0])
        # polygon_masks = []
        # for instance in polygons_per_instance:
        #     polygon_masks.append([instance.squeeze().cpu()])
        # gt_instances._fields["gt_masks"] = PolygonMasks(polygon_masks)
        # gt_instances.remove("gt_segm")
        return gt_instances

    def forward(self, image, gt_instances):
        # image.shape is (C, H, W)
        self.generate_params(image, gt_instances)
        image = self.apply_image(image)
        if not self.skip_coords:
            gt_instances = self.apply_instances(gt_instances)
        return image, gt_instances
