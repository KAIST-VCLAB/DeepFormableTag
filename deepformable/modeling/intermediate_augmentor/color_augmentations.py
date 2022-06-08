"""
This code implemented by Andreas Meulueman and Mustafa B. Yaldiz
Copyright (c) (VCLAB, KAIST) All Rights Reserved.
"""
import torch
from torch import nn
import torch.nn.functional as F
import kornia
import numpy as np

import detectron2
from detectron2.config import configurable

from .build import INTERMEDIATE_AUGMENTOR_REGISTRY, IntermediateAugmentor
from deepformable.utils import (
    get_disk_blur_kernel, sample_param, 
    rgb_to_hls, hls_to_rgb,
)


@INTERMEDIATE_AUGMENTOR_REGISTRY.register()
class GammaAugmentor(IntermediateAugmentor):
    @configurable
    def __init__(
        self,
        *,        
        gamma_range,
    ):
        super().__init__(True)
        self.gamma_range = gamma_range

    @classmethod
    def from_config(cls, cfg):
        return {
            "gamma_range": cfg.INTERMEDIATE_AUGMENTOR.GammaAugmentor.GAMMA_RANGE,
        }

    def apply_image(self, image):
        return (F.relu(image) + 1e-8) ** self.gamma

    def generate_params(self, image, gt_instances, strength=None):
        self.gamma = sample_param(
            self.gamma_range, strength=strength, 
            training=self.training, device=image.device)


@INTERMEDIATE_AUGMENTOR_REGISTRY.register()
class GammaCorrector(IntermediateAugmentor):
    @configurable
    def __init__(
        self,
        gamma,
    ):
        super().__init__(True)
        self.register_buffer("gamma", torch.tensor(1.0/gamma), False)

    @classmethod
    def from_config(cls, cfg):
        return {"gamma": cfg.RENDERER.GAMMA}

    def apply_image(self, image):
        return torch.clamp((F.relu(image) + 1e-8) ** self.gamma.to(image.device), 0, 1)


@INTERMEDIATE_AUGMENTOR_REGISTRY.register()
class DefocusBlurAugmentor(IntermediateAugmentor):
    @configurable
    def __init__(
        self,
        *,
        blur_radius_range,
    ):
        super().__init__(True)
        self.blur_radius_range = blur_radius_range

    @classmethod
    def from_config(cls, cfg):
        return {
            "blur_radius_range": cfg.INTERMEDIATE_AUGMENTOR.DefocusBlurAugmentor.BLUR_RADIUS_RANGE,
        }

    def apply_image(self, image):
        pad = self.kernel.size(-1)//2
        padded_image = F.pad(image.unsqueeze(0),
            pad=(pad, pad, pad, pad),
            mode="reflect")
        return F.conv2d(
            padded_image,
            self.kernel.expand(3,1,self.kernel.shape[-1], self.kernel.shape[-1]),
            groups=3, padding=0)[0]

    def generate_params(self, image, gt_instances, strength=None):
        device = image.device
        blur_radius = sample_param(
            self.blur_radius_range, strength=strength, 
            training=self.training, device=device)
        self.kernel = get_disk_blur_kernel(blur_radius, device=device)


@INTERMEDIATE_AUGMENTOR_REGISTRY.register()
class MotionBlurAugmentor(IntermediateAugmentor):
    @configurable
    def __init__(
        self,
        *,
        blur_radius_range,
    ):
        super().__init__(True)
        self.blur_radius_range = blur_radius_range

    @classmethod
    def from_config(cls, cfg):
        return {
            "blur_radius_range": cfg.INTERMEDIATE_AUGMENTOR.MotionBlurAugmentor.BLUR_RADIUS_RANGE,
        }

    def apply_image(self, image):
        return kornia.filters.motion_blur(
            image.unsqueeze(0), self.blur_radius, self.angle, 
            self.direction, border_type='replicate', mode='bilinear')[0]

    def generate_params(self, image, gt_instances, strength=None):
        device = image.device
        blur_radius = sample_param(
            self.blur_radius_range, shape=(1,), strength=strength, 
            training=self.training, device=device)
        self.blur_radius = (torch.round(blur_radius).int()*2+1).item()
        self.angle = sample_param(
            (0,180,30), shape=(1,),
            training=self.training, device=device) # Blur at 30° for testing
        self.direction = torch.zeros(1, device=image.device)


@INTERMEDIATE_AUGMENTOR_REGISTRY.register()
class HueShiftAugmentor(IntermediateAugmentor):
    @configurable
    def __init__(
        self,
        *,
        hue_shift_range,
    ):
        super().__init__(True)
        self.hue_shift_range = hue_shift_range

    @classmethod
    def from_config(cls, cfg):
        return {
            "hue_shift_range": cfg.INTERMEDIATE_AUGMENTOR.HueShiftAugmentor.HUE_SHIFT_RANGE,
        }

    def apply_image(self, image):
        image = torch.clamp(image, 0, 1 - 1e-6)
        hsv = rgb_to_hls(image)
        hsv[0, :, :] = torch.fmod(hsv[0, :, :] + self.hue_shift * kornia.constants.pi, 2*kornia.constants.pi)
        return hls_to_rgb(hsv)

    def generate_params(self, image, gt_instances, strength=None):
        device = image.device
        self.hue_shift = sample_param(
            self.hue_shift_range, strength=strength, 
            training=self.training, device=device)


@INTERMEDIATE_AUGMENTOR_REGISTRY.register()
class BrightnessAugmentor(IntermediateAugmentor):
    @configurable
    def __init__(
        self,
        *,
        brightness_range,
    ):
        super().__init__(True)
        self.brightness_range = brightness_range

    @classmethod
    def from_config(cls, cfg):
        return {
            "brightness_range": cfg.INTERMEDIATE_AUGMENTOR.BrightnessAugmentor.BRIGHTNESS_RANGE, 
        }
    def apply_image(self, image):
        return image * self.factor
    
    def generate_params(self, image, gt_instances, strength=None):
        device = image.device
        if strength == None:
            self.factor = sample_param(
                self.brightness_range, training=self.training, device=device)
        else:
            brightness_range = (self.brightness_range[0], 1.0, 0.4)
            self.factor = sample_param(
                brightness_range, strength=1.0 - strength,
                training=self.training, device=device)


@INTERMEDIATE_AUGMENTOR_REGISTRY.register()
class NoiseAugmentor(IntermediateAugmentor):
    @configurable
    def __init__(
        self,
        *,       
        noise_range,
    ):
        super().__init__(True)
        self.noise_range=noise_range

    @classmethod
    def from_config(cls, cfg):
        return {
            "noise_range": cfg.INTERMEDIATE_AUGMENTOR.NoiseAugmentor.NOISE_RANGE,
        }

    def apply_image(self, image):
        return image + self.sigma * torch.randn(image.shape, device=image.device)

    def generate_params(self, image, gt_instances, strength=None):
        self.sigma = sample_param(
            self.noise_range, strength=strength, 
            training=self.training, device=image.device)
