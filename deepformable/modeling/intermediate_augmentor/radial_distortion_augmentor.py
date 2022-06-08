"""
This code implemented by Andreas Meulueman and Mustafa B. Yaldiz
Copyright (c) (VCLAB, KAIST) All Rights Reserved.
"""
import itertools

import torch
import torch.nn.functional as F

from detectron2.config import configurable
from .build import INTERMEDIATE_AUGMENTOR_REGISTRY, IntermediateAugmentor
from deepformable.utils import sample_param

@INTERMEDIATE_AUGMENTOR_REGISTRY.register()
class RadialDistortionAugmentor(IntermediateAugmentor):
    @configurable
    def __init__(
        self,
        *,
        undistort_iter,
        focal_length_range,
        center_shift_range,
        distortion_range
    ):
        super().__init__(False)
        self.undistort_iter=undistort_iter
        self.focal_length_range = focal_length_range
        self.center_shift_range = center_shift_range
        self.distortion_range = distortion_range

    @classmethod
    def from_config(cls, cfg):
        return {
            "undistort_iter": cfg.INTERMEDIATE_AUGMENTOR.RadialDistortionAugmentor.UNDISTORT_ITER,
            "focal_length_range": cfg.INTERMEDIATE_AUGMENTOR.RadialDistortionAugmentor.FOCAL_LENGTH_RANGE,
            "center_shift_range": cfg.INTERMEDIATE_AUGMENTOR.RadialDistortionAugmentor.CENTER_SHIFT_RANGE,
            "distortion_range": cfg.INTERMEDIATE_AUGMENTOR.RadialDistortionAugmentor.DISTORTION_RANGE,
        }

    def apply_image(self, image):
        return F.grid_sample(image.unsqueeze(0), self.grid.unsqueeze(0), align_corners=False)[0]
    
    def distort(self, coord):
        xy = (coord.float() - self.center.view(1, 2)) / self.focal_length.view(1, 2)

        r2 = xy[:, 0]**2 + xy[:, 1]**2

        r2_distorted = 1
        for i in range(self.k.shape[0]):
            r2_distorted = r2_distorted + self.k[i] * r2**(i+1)

        xy_distorted = xy * r2_distorted.unsqueeze(-1)

        return xy_distorted * self.focal_length.view(1, 2) + self.center.view(1, 2)

    def undistort(self, coord):
        xy = (coord.float() - self.center.view(1, 2)) / self.focal_length.view(1, 2)
        xy0 = xy.clone()

        for iteration in range(self.undistort_iter):
            r2 = (xy[:, 0])**2 + (xy[:, 1])**2

            # This works up to the third order 
            #r2_undistorted = (1+((self.k[5]*r2 + self.k[4])*r2 + self.k[3])*r2)/(1 + ((self.k[2]*r2 + self.k[1])*r2 + self.k[0])*r2)
            r2_undistorted = (1 + ((self.k[2]*r2 + self.k[1])*r2 + self.k[0])*r2)

            xy = xy0 / r2_undistorted.unsqueeze(-1)
            x = xy[:, 0]
            x[r2_undistorted < 0] = -1
            y = xy[:, 1]
            y[r2_undistorted < 0] = -1

        return xy * self.focal_length.view(1, 2) + self.center.view(1, 2)

    def apply_coords(self, coords):
        return self.distort(coords)

    def generate_params(self, image, gt_instances, strength=None):
        image_size, device = image.shape[-2:], image.device
        image_size_xy = torch.tensor([image_size[1], image_size[0]], device=device)

        self.focal_length = sample_param(
            self.focal_length_range, shape=(2,), 
            strength = None if strength == None else 1.0-strength,
            training=self.training, device=device) * torch.max(image_size_xy)
        center_shift = sample_param(
            self.center_shift_range, shape=(2,), strength=strength,
            training=self.training, device=device)
        if self.training:
            center_shift *= 1 if torch.randn(1) > 0.5 else -1
        self.center = (0.5 + center_shift) * image_size_xy
        self.k = -sample_param(
            self.distortion_range, shape=(3,), strength=strength,
            training=self.training, device=device)

        x, y = torch.meshgrid([torch.arange(0, image_size[0], device=device), 
            torch.arange(0, image_size[1], device=device)])
        x = x.float() + 0.5
        y = y.float() + 0.5
        coord = torch.cat((y.unsqueeze(-1), x.unsqueeze(-1)), 2).view(-1, 2)

        grid = self.undistort(coord).view(image_size[0], image_size[1], 2)
        self.grid = (grid / image_size_xy.view(1, 1, 2)) * 2 - 1
