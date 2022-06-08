# Copyright (c) Mustafa B. Yaldiz (VCLAB, KAIST) All Rights Reserved.
import logging
import numpy as np
import math
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

from detectron2.config import configurable

from deepformable.utils import sample_param, get_disk_blur_kernel

@torch.jit.script
def tensor_dot(a, b):
    return torch.sum(a * b, dim=1).unsqueeze(1)

@torch.jit.script
def tensor_normalize(m, eps:float=1e-8):
    length = torch.sqrt(torch.sum(m * m, dim=1)).unsqueeze(1)
    length = torch.clamp(length, min=eps)
    return m / length

@torch.jit.script
def render_specular(
    light_dir, view_dir, normals, roughness, 
    fresnel: float=0.1, eps:float=1e-8
):
    halfway_vector = tensor_normalize(
        (light_dir+view_dir) * 0.5, eps=eps)

    NoH = F.relu(tensor_dot(normals, halfway_vector))
    NoV = F.relu(tensor_dot(normals, view_dir))
    NoL = F.relu(tensor_dot(normals, light_dir))
    VoH = F.relu(tensor_dot(view_dir, halfway_vector))
    
    # GGX
    alpha = roughness  * roughness
    tmp = alpha / torch.clamp((NoH * NoH * (alpha * alpha - 1.0) + 1.0), min=eps)
    D = tmp * tmp * (1.0 / np.pi)
    
    # SmithG
    k = alpha * 0.5
    # G = (NoL / torch.clamp(NoL * (1.0 - k) + k, min=eps)) * (NoV / torch.clamp(NoV * (1.0 - k) + k, min=eps))
    G = (NoL * NoV) / torch.clamp((NoL * (1.0 - k) + k) * (NoV * (1.0 - k) + k), min=eps)
    
    # Fresnel
    coeff = VoH * (-5.55473 * VoH - 6.98316)
    F_term = fresnel + (1.0 - fresnel) * torch.pow(2.0, coeff)

    f_s = D * G * F_term / torch.clamp(4.0 * NoL * NoV, min=eps)
    return f_s * NoL * np.pi


@torch.jit.script
def transform_points(h, p, eps: float = 1e-8):
    p_homogenius = F.pad(p, [0,1], 'constant', 1.0)
    points_transformed = torch.bmm(p_homogenius.expand(h.size(0),-1,-1), h.permute(0,2,1))
    z_vec = points_transformed[..., -1:]
    mask = torch.abs(z_vec) > eps
    scale = torch.where(mask, 1.0 / (z_vec + eps), torch.ones_like(z_vec)) # Is this right? kornia implementation
    return scale * points_transformed[..., :-1]


@torch.jit.script
def draw_patches(
    image_size: Tuple[int, int],
    patches,
    homographies,
    blur_kernel_radius: float,
):
    device = patches.device
    image_height, image_width = image_size
    patch_height, patch_width = patches.size(2), patches.size(3)
    corner_pts = torch.tensor([[-1,-1],[1,-1],[1,1],[-1,1]], dtype=torch.float, device=device)
    y, x = torch.meshgrid([torch.arange(0, image_height, device=patches.device), 
            torch.arange(0, image_width, device=device)])
    coord = torch.cat((x.float().unsqueeze(-1) + 0.5, y.float().unsqueeze(-1) + 0.5), 2)

    target_pts = transform_points(homographies, corner_pts)

    blur_kernel = get_disk_blur_kernel(blur_kernel_radius, device=device)

    margin = blur_kernel.size(1) // 2
    target_bbox = torch.stack(
        [torch.min(target_pts, 1)[0]-margin, torch.max(target_pts, 1)[0]+margin], dim=1)

    image_max = torch.tensor([image_width, image_height], device=device, dtype=torch.float32)
    mask = target_bbox < image_max
    target_bbox = torch.where(mask, target_bbox, image_max)
    target_bbox = torch.clamp(target_bbox, min=0.0)

    # We upsample the marker to a size matching the area taken in the target image
    target_bbox_size = torch.ceil(target_bbox[:,1] - target_bbox[:,0])
    scale_batch = target_bbox_size / torch.tensor([patch_width, patch_height], device=device)
    scale_batch_clamped = torch.max(scale_batch, 1)[0]
    scale_batch_clamped = torch.clamp(scale_batch_clamped, min=1.0)
    scale_batch_clamped = torch.ceil(scale_batch_clamped)

    placed_patches, placed_alphas = [], []
    for patch, bbox, scale, h_inv in zip(patches, target_bbox, scale_batch_clamped, torch.inverse(homographies)):
        bbox = bbox.int()
        # We upsample the marker to a size matching the area taken in the target image
        upsampled_patch = F.interpolate(
            patch.unsqueeze(0), 
            scale_factor=scale.item(), mode='nearest')
        alpha = torch.ones(upsampled_patch.shape, device=device)

        # Pad and smoothen the patches and alpha
        padded_patch = F.pad(
            upsampled_patch, [2*margin]*4, mode='replicate')
        filtered_patch = F.conv2d(
            padded_patch, 
            blur_kernel.expand(3,1,blur_kernel.shape[-1], blur_kernel.shape[-1]), 
            groups=3, padding=0)
        padded_alpha = F.pad(alpha, [2*margin]*4)
        filtered_alpha = F.conv2d(
            padded_alpha, 
            blur_kernel.expand(3,1,blur_kernel.shape[-1], blur_kernel.shape[-1]), 
            groups=3, padding=0)

        # Get only the coordinates in the target image of the pixels where the marker will be placed for efficiency
        cropped_coord = coord[bbox[0, 1]:bbox[1, 1], bbox[0, 0]:bbox[1, 0]].contiguous()
        
        # Scale the h matrix
        scale_mtx = torch.diag(
            torch.tensor([
                (filtered_patch.shape[-1] - 2 * margin) / filtered_patch.shape[-1],
                (filtered_patch.shape[-2] - 2 * margin) / filtered_patch.shape[-2],
                1.0], 
                device=device))
        h_inv_scaled = scale_mtx @ h_inv
        # Get the sampling coordinates in the marker corresponding to the pixels where the marker will be placed
        cropped_grid = transform_points(
            h_inv_scaled.unsqueeze(0), cropped_coord.view(1, -1, 2)
        ).view(cropped_coord.shape)

        # Get the sampling coordinates in the marker corresponding to the full target image by filling with -2 elswhere
        grid = -2*torch.ones((1, image_height, image_width, 2), device=device)
        grid[0, bbox[0, 1]:bbox[1, 1], bbox[0, 0]:bbox[1, 0]] = cropped_grid

        # Place the marker and their alpha blending
        placed_patches.append(
            F.grid_sample(filtered_patch, grid, mode='bilinear', padding_mode='zeros', align_corners=False)[0])
        placed_alphas.append(
            F.grid_sample(filtered_alpha, grid, mode='bilinear', padding_mode='zeros', align_corners=False)[0])
    return placed_patches, placed_alphas


class MarkerRenderer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.epsilon = cfg.RENDERER.EPSILON
        self.blur_range = cfg.RENDERER.BLUR_RANGE
        self.roughness_range = cfg.RENDERER.ROUGHNESS_RANGE
        self.board_diffuse_range = cfg.RENDERER.DIFFUSE_RANGE
        self.normal_noise_range = cfg.RENDERER.NORMAL_NOISE_RANGE
        self.specular_range = cfg.RENDERER.SPECULAR_RANGE
        self.render_specular_term = True if cfg.RENDERER.SHADING_METHOD == "cook-torrance" else False
    
    def forward(self, d, markers):
        # Prepare the data
        gt_instances, gt_board_instances = d["instances"], d["board_instances"]
        device = gt_instances.homography.device
        image, view_dirs = d["image"], d["view_dir"]
        image_size = tuple(image.shape[-2:])

        # Prepare the parameters
        blur_kernel_radius = sample_param(
            self.blur_range, training=self.training, device=device)
        gt_board_instances.roughness = sample_param(
            self.roughness_range, training=self.training, 
            shape=(gt_board_instances.homography.size(0),), device=device)
        diffuse = sample_param(
            self.board_diffuse_range, training=self.training, 
            shape=(gt_board_instances.homography.size(0), 3), device=device)
        
        if self.render_specular_term:
            # Calculate specular power that avoids washed out specular regions
            specular_term_max = render_specular(
                gt_board_instances.refl_dir, gt_board_instances.view_dir, 
                gt_board_instances.normal, gt_board_instances.roughness.view(-1,1).repeat(1,3))
            avg_color = gt_board_instances.avg_color
            brightness_max = gt_board_instances.brightness_max
            specular_power = (avg_color / (torch.max(specular_term_max, dim=1)[0].view(-1,1) + 1e-8))\
                * (1.1-brightness_max)
            gt_board_instances.specular_power = specular_power * sample_param(
                self.specular_range, shape=(len(brightness_max),), 
                training=self.training, device=device).unsqueeze(-1)

        # Render markers
        placed_markers, placed_marker_alphas = draw_patches(
            image_size, markers, gt_instances.homography, 
            blur_kernel_radius)
        
        # Render boards
        boards_diffuse = diffuse.view(-1,3,1,1).repeat(1, 1, 4, 4)
        placed_boards, placed_board_alphas = draw_patches(
            image_size, boards_diffuse, gt_board_instances.homography, 
            blur_kernel_radius)

        if self.render_specular_term:
            # Calculate per board params
            specular_dirs, surface_normals, roughness, specular_power = [], [], [], []
            for i, board_alpha in enumerate(placed_board_alphas):
                board_hard_mask = (board_alpha != 0).float()
                b_i = gt_board_instances[i]
                specular_dirs.append(board_hard_mask*b_i.refl_dir.view(-1,1,1))
                surface_normals.append(board_hard_mask*b_i.normal.view(-1,1,1))
                roughness.append(board_hard_mask*b_i.roughness.view(-1,1,1))
                specular_power.append(board_alpha*b_i.specular_power.view(-1,1,1))
            specular_dirs, surface_normals = sum(specular_dirs), sum(surface_normals)
            roughness, specular_power = sum(roughness), sum(specular_power)

            # Add noise to surface normals
            surface_normals = surface_normals + sample_param(
                self.normal_noise_range, training=self.training,
                shape=surface_normals.shape, device=device)
            surface_normals = tensor_normalize(surface_normals.unsqueeze(0))

        # Stack all the boards and markers
        placed_markers, placed_marker_alphas = sum(placed_markers), sum(placed_marker_alphas)
        placed_boards, placed_board_alphas = sum(placed_boards), sum(placed_board_alphas)

        # Calculate stacked board + marker, while using diffuse appearance of board
        markers_boards_stacked = placed_boards * (1.0 - placed_marker_alphas) \
            + placed_boards * placed_markers * placed_marker_alphas
        # Calculate final image
        final_image = image * (1.0 - placed_board_alphas) \
            + image * placed_board_alphas * markers_boards_stacked       
        
        if self.render_specular_term:
            # print("-", specular_dirs.unsqueeze(0).shape, view_dirs.permute(2,0,1).unsqueeze(0).shape, surface_normals.shape, roughness.unsqueeze(0).shape, specular_power.shape)
            # Calculate specular term
            specular_term = render_specular(
                specular_dirs.unsqueeze(0), view_dirs.permute(2,0,1).unsqueeze(0),
                surface_normals, roughness.unsqueeze(0))[0]
            final_image = final_image + specular_term * specular_power

        return final_image