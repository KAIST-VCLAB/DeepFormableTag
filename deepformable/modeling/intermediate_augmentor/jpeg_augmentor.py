"""
JPEG compression augmentation, see https://github.com/ando-khachatryan/HiDDeN
Modified by Andreas Meulueman and Mustafa B. Yaldiz.
Copyright (c) 2018 ando-khachatryan
"""
import torch
from torch import nn
import torch.nn.functional as F
import kornia
import numpy as np

import detectron2
from detectron2.config import configurable

from .build import INTERMEDIATE_AUGMENTOR_REGISTRY, IntermediateAugmentor
from deepformable.utils import sample_param


def gen_filters(size_x: int, size_y: int, dct_or_idct_fun: callable) -> np.ndarray:
    tile_size_x = 8
    filters = np.zeros((size_x * size_y, size_x, size_y))
    for k_y in range(size_y):
        for k_x in range(size_x):
            for n_y in range(size_y):
                for n_x in range(size_x):
                    filters[k_y * tile_size_x + k_x, n_y, n_x] = dct_or_idct_fun(n_y, k_y, size_y) * dct_or_idct_fun(n_x,
                                                                                                            k_x,
                                                                                                            size_x)
    return filters

def create_jpeg_masks(min_keep:int=1, max_keep:int=64):
    index_order = np.array(
        sorted(((x, y) for x in range(8) for y in range(8)),
        key=lambda p: (p[0] + p[1], -p[1] if (p[0] + p[1]) % 2 else p[1]))
    )
    masks = []
    for keep_count in range(min_keep, max_keep):
        mask = np.zeros((8, 8))
        mask[index_order[:keep_count,0], index_order[:keep_count,1]] = 1
        masks.append(mask)
    return np.stack(masks,axis=0)

def dct_coeff(n, k, N):
    return np.cos(np.pi / N * (n + 1. / 2.) * k)

def idct_coeff(n, k, N):
    return (int(0 == n) * (- 1 / 2) + np.cos(
        np.pi / N * (k + 1. / 2.) * n)) * np.sqrt(1 / (2. * N))


@INTERMEDIATE_AUGMENTOR_REGISTRY.register()
class JPEGAugmentor(IntermediateAugmentor):
    @configurable
    def __init__(
        self,
        *,
        y_range,
        uv_range,
        max_image_size,
    ):
        super().__init__(True)
        self.y_range = y_range
        self.uv_range = uv_range
        self.register_buffer("dct_conv_weights",
            torch.tensor(gen_filters(8, 8, dct_coeff), dtype=torch.float32).unsqueeze(1), False)
        self.register_buffer("idct_conv_weights",
            torch.tensor(gen_filters(8, 8, idct_coeff), dtype=torch.float32).view(64,64,1,1), False)
        self.register_buffer("jpeg_masks",
            torch.tensor(create_jpeg_masks(), dtype=torch.float32), False)

    @property
    def device(self):
        return self.dct_conv_weights.device

    @classmethod
    def from_config(cls, cfg):
        return {
            "y_range": cfg.INTERMEDIATE_AUGMENTOR.JPEGAugmentor.Y_QUALITY_RANGE,
            "uv_range": cfg.INTERMEDIATE_AUGMENTOR.JPEGAugmentor.UV_QUALITY_RANGE,
            "max_image_size": cfg.INTERMEDIATE_AUGMENTOR.MAX_IMAGE_SIZE,
        }

    def apply_image(self, image):
        image = image.unsqueeze(0)
        N, C, H, W = image.shape

        mask = self.jpeg_masks[None, self.mask_keep_weights].view(1,C,8,8,1,1)

        # Convert to YUV
        if C == 3:
            image = kornia.color.rgb_to_yuv(image)
        # Pad image
        image_padded = F.pad(image, (0, (8 - W) % 8, 0, (8 - H) % 8), 'replicate')
        H_pad, W_pad = image_padded.shape[-2:]
        
        # Apply dct transform
        image_dct = F.conv2d(
            image_padded, self.dct_conv_weights.repeat(C,1,1,1), 
            stride=8, groups=C)
        image_dct = image_dct.view(N,C,8,8,*image_dct.shape[-2:])
        # Mask in dct domain
        image_dct_masked = image_dct * mask
        # Convert back to idct
        image_idct = F.conv2d(
            image_dct_masked.view(N, C*64, *image_dct.shape[-2:]),
            self.idct_conv_weights.repeat(C,1,1,1), groups=C)
        image_idct = image_idct.view(N,3,8,8,*image_dct.shape[-2:])\
                        .permute(0,1,4,2,5,3).contiguous().view(-1,C,H_pad,W_pad)

        # Convert back to RGB
        if C == 3:
            image_idct = kornia.color.yuv_to_rgb(image_idct)
        
        return torch.clamp(image_idct[0,:,:H,:W], 0, 1)
    
    def generate_params(self, image, gt_instances, strength=None):
        if image.device != self.device:
            self.to(image.device)
        y_weight = sample_param(
            self.y_range, strength=strength, 
            training=self.training, device=self.device)
        if image.shape[0] == 3:
            u_weight, v_weight = sample_param(
                self.uv_range, shape=(2,), strength=strength, 
                training=self.training, device=self.device)
            self.mask_keep_weights = (int(y_weight), int(u_weight.item()), int(v_weight.item()))
        else:
            self.mask_keep_weights = (int(y_weight),) * image.shape[0]