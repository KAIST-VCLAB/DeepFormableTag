# Copyright (c) Mustafa B. Yaldiz (VCLAB, KAIST) All Rights Reserved.
import math
from typing import List, Optional
import numpy as np

import kornia
import torch
import  torch.nn.functional as F

def sample_param(
    param_range, shape=1, strength=None,
    training=True, device=torch.device("cpu")
):
    min_v, max_v, test_v = param_range
    if training or strength != None:
        if strength:
            rand_val = torch.ones(shape, device=device) * strength
        else:
            rand_val = torch.rand(shape, device=device)
        rand_val = min_v + (max_v-min_v) * rand_val
    else:
        rand_val = torch.ones(shape, device=device) * test_v
    return rand_val.item() if shape==1 else rand_val

@torch.jit.script
def get_disk_blur_kernel(
    kernel_radius: float, upscale_factor: int = 4, 
    device: torch.device=torch.device("cpu")
):
    # TODO: Approximate implementation, fix for exact one
    r = kernel_radius * upscale_factor
    kernel_scaled_size = (math.ceil(kernel_radius)*2+1)*upscale_factor
    kernel = torch.ones(kernel_scaled_size, kernel_scaled_size, device=device)
    x, y = torch.meshgrid([
        torch.linspace(
            -(kernel_scaled_size-1.0)/2.0, (kernel_scaled_size-1.0)/2.0, kernel_scaled_size, device=device)]*2)
    kernel[x**2 + y**2 > r**2] = 0
    kernel = F.avg_pool2d(kernel.unsqueeze(0), (upscale_factor, upscale_factor))
    return (kernel / torch.sum(kernel, (1, 2)))

def rgb_to_hls(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to HLS
    The image data is assumed to be in the range of (0, 1).

    Args:
        input (torch.Tensor): RGB Image to be converted to HLS.


    Returns:
        torch.Tensor: HLS version of the image.
    """

    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    maxc: torch.Tensor = image.max(-3)[0]
    minc: torch.Tensor = image.min(-3)[0]

    imax: torch.Tensor = image.max(-3)[1]

    l: torch.Tensor = (maxc + minc) / 2  # luminance
    l2 = maxc + minc + 1e-8

    deltac: torch.Tensor = maxc - minc

    s: torch.Tensor = torch.where(l < 0.5, deltac / (l2), deltac /
                                  (torch.tensor(2.) - (l2)))  # saturation

    deltac = deltac + 1e-8

    hi: torch.Tensor = torch.zeros_like(deltac)

    hi[imax == 0] = (((g - b) / deltac) % 6)[imax == 0]
    hi[imax == 1] = (((b - r) / deltac) + 2)[imax == 1]
    hi[imax == 2] = (((r - g) / deltac) + 4)[imax == 2]

    h: torch.Tensor = 2. * kornia.constants.pi.to(image.device) * (60. * hi) / 360.  # hue [0, 2*pi]

    image_hls: torch.Tensor = torch.stack([h, l, s], dim=-3)

    image_hls[torch.isnan(image_hls)] = 0.

    return image_hls

def hls_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an HLS image to RGB
    The image data is assumed to be in the range of (0, 1).

    Args:
        input (torch.Tensor): HLS Image to be converted to RGB.


    Returns:
        torch.Tensor: RGB version of the image.
    """

    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    h: torch.Tensor = image[..., 0, :, :] * 360 / (2 * kornia.constants.pi.to(image.device))
    l: torch.Tensor = image[..., 1, :, :]
    s: torch.Tensor = image[..., 2, :, :]

    kr = (0 + h / 30) % 12
    kg = (8 + h / 30) % 12
    kb = (4 + h / 30) % 12
    a = s * torch.min(l, torch.tensor(1.) - l)

    ones_k = torch.ones_like(kr)

    fr: torch.Tensor = l - a * torch.max(torch.min(torch.min(kr - torch.tensor(3.),
                                                             torch.tensor(9.) - kr), ones_k), -1 * ones_k)
    fg: torch.Tensor = l - a * torch.max(torch.min(torch.min(kg - torch.tensor(3.),
                                                             torch.tensor(9.) - kg), ones_k), -1 * ones_k)
    fb: torch.Tensor = l - a * torch.max(torch.min(torch.min(kb - torch.tensor(3.),
                                                             torch.tensor(9.) - kb), ones_k), -1 * ones_k)

    out: torch.Tensor = torch.stack([fr, fg, fb], dim=-3)

    return out