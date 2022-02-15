"""
This code is modified from detectron2 implementation, 
changes are logged in comments. 
Copyright (c) Mustafa B. Yaldiz (VCLAB, KAIST) All Rights Reserved.
"""
from __future__ import division
from typing import List, Dict, Optional, Tuple
import numpy as np

import torch
from torch.nn import functional as F

from detectron2.layers.wrappers import shapes_to_tensor
from detectron2.structures import ImageList as Detectron2_Imagelist


class ImageList(Detectron2_Imagelist):
    @staticmethod
    def from_tensors(
        tensors: List[torch.Tensor], size_divisibility: int = 0, pad_value: float = 0.0
    ) -> "ImageList":
        """
        Detectron2's ImageList implementation modified 
        to allow proper gradient flow.
        """
        assert len(tensors) > 0
        assert isinstance(tensors, (tuple, list))
        for t in tensors:
            assert isinstance(t, torch.Tensor), type(t)
            assert t.shape[:-2] == tensors[0].shape[:-2], t.shape

        image_sizes = [(im.shape[-2], im.shape[-1]) for im in tensors]
        image_sizes_tensor = [shapes_to_tensor(x) for x in image_sizes]
        max_size = torch.stack(image_sizes_tensor).max(0).values

        if size_divisibility > 1:
            stride = size_divisibility
            # the last two dims are H,W, both subject to divisibility requirement
            max_size = (max_size + (stride - 1)).div(stride, rounding_mode="floor") * stride

        # handle weirdness of scripting and tracing ...
        if torch.jit.is_scripting():
            max_size: List[int] = max_size.to(dtype=torch.long).tolist()
        else:
            if torch.jit.is_tracing():
                image_sizes = image_sizes_tensor

        batched_imgs = []
        for img, image_size in zip(tensors, image_sizes):
            padding_size = [0, max_size[-1] - image_size[1], 0, max_size[-2] - image_size[0]]
            batched_imgs.append(F.pad(img, padding_size, value=pad_value))
        batched_imgs = torch.stack(batched_imgs, dim=0)

        return ImageList(batched_imgs.contiguous(), image_sizes)