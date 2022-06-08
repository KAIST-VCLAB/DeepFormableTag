"""
This code implemented by Andreas Meulueman and Mustafa B. Yaldiz
Copyright (c) (VCLAB, KAIST) All Rights Reserved.
"""
import itertools

import torch
from torch import nn

import kornia

from detectron2.config import configurable

from .build import INTERMEDIATE_AUGMENTOR_REGISTRY, IntermediateAugmentor
from deepformable.utils import sample_param


def create_perspective_sampling_grid(image_size, target_corners, device):
    target_corners = torch.tensor([
        [target_corners[0, 0], target_corners[0, 1]],
        [target_corners[1, 0], 1 - target_corners[1, 1]],
        [1 - target_corners[2, 0], target_corners[2, 1]],
        [1 - target_corners[3, 0], 1 - target_corners[3, 1]]], device=device)
    
    target_corners[:, 0] = target_corners[:, 0] * (image_size[1])
    target_corners[:, 1] = target_corners[:, 1] * (image_size[0])

    source_corners = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], device=device).float()
    source_corners[:, 0] = source_corners[:, 0] * (image_size[1])
    source_corners[:, 1] = source_corners[:, 1] * (image_size[0])

    homography = kornia.geometry.find_homography_dlt(source_corners.unsqueeze(0), 
        target_corners.unsqueeze(0), 
        torch.ones(1, 4, device=device))

    x, y = torch.meshgrid([torch.arange(0, image_size[0], device=device), 
        torch.arange(0, image_size[1], device=device)])

    x = x.float() + 0.5
    y = y.float() + 0.5
    coord = torch.cat((y.unsqueeze(-1), x.unsqueeze(-1)), 2).view(-1, 2).float()

    grid = kornia.geometry.linalg.transform_points(
        torch.inverse(homography), coord).view(image_size[0], image_size[1], 2)
    grid = (grid / torch.tensor([image_size[1], image_size[0]], device=device).view(1, 1, 2)) * 2 - 1
    
    return grid, homography

@INTERMEDIATE_AUGMENTOR_REGISTRY.register()
class PerspectiveAugmentor(IntermediateAugmentor):
    @configurable
    def __init__(
        self,
        *,
        corner_shift_range,
    ):
        super().__init__(False)
        self.corner_shift_range=corner_shift_range

    @classmethod
    def from_config(cls, cfg):
        return {
            "corner_shift_range": cfg.INTERMEDIATE_AUGMENTOR.PerspectiveAugmentor.CORNER_SHIFT_RANGE,
        }

    def apply_image(self, image):
        return torch.nn.functional.grid_sample(image.unsqueeze(0), self.grid.unsqueeze(0), align_corners=False)[0]
    
    def apply_coords(self, coords):
        return kornia.geometry.linalg.transform_points(self.homography, coords.unsqueeze(0))[0]

    def generate_params(self, image, gt_instances, strength=None):
        device = image.device
        if self.training:
            target_corners = sample_param(
                self.corner_shift_range, shape=(4,2),
                strength=strength, device=device)
        else:
            target_corners = torch.zeros((4, 2), device=device)
            target_corners[[0,1],:] = sample_param(
                self.corner_shift_range, strength=strength, 
                training=False, device=device)
        self.grid, self.homography = create_perspective_sampling_grid(
            image.shape[-2:], target_corners, device=device)