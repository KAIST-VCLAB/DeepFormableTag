"""
This code implemented by Andreas Meulueman and Mustafa B. Yaldiz
Copyright (c) (VCLAB, KAIST) All Rights Reserved.
"""
import itertools

import torch
from torch import nn
import torch.nn.functional as F

from detectron2.config import configurable

from .build import INTERMEDIATE_AUGMENTOR_REGISTRY, IntermediateAugmentor
from deepformable.utils import sample_param

# phi(x1, x2) = r^2 * log(r), where r = ||x1 - x2||_2
def compute_partial_repr(input_points, control_points):
    N = input_points.size(0)
    M = control_points.size(0)
    pairwise_diff = input_points.view(N, 1, 2) - control_points.view(1, M, 2)
    # original implementation, very slow
    # pairwise_dist = torch.sum(pairwise_diff ** 2, dim = 2) # square of distance
    pairwise_diff_square = pairwise_diff * pairwise_diff
    del pairwise_diff
    pairwise_dist = pairwise_diff_square[:, :, 0] + pairwise_diff_square[:, :, 1]
    del pairwise_diff_square

    repr_matrix = 0.5 * pairwise_dist * torch.log(pairwise_dist)
    del pairwise_dist
    # fix numerical error for 0 * log(0), substitute all nan with 0
    # mask = repr_matrix != repr_matrix
    repr_matrix.masked_fill_(repr_matrix != repr_matrix, 0)

    return repr_matrix

class TPSGridGen(nn.Module):
    def __init__(self, target_height, target_width, target_control_points):
        super(TPSGridGen, self).__init__()
        assert target_control_points.ndimension() == 2
        assert target_control_points.size(1) == 2
        N = target_control_points.size(0)
        self.num_points = N
        self.height = target_height
        self.width = target_width

        # create padded kernel matrix
        forward_kernel = torch.zeros(N + 3, N + 3)
        target_control_partial_repr = compute_partial_repr(target_control_points, target_control_points)
        forward_kernel[:N, :N].copy_(target_control_partial_repr)
        forward_kernel[:N, -3].fill_(1)
        forward_kernel[-3, :N].fill_(1)
        forward_kernel[:N, -2:].copy_(target_control_points)
        forward_kernel[-2:, :N].copy_(target_control_points.transpose(0, 1))
        # compute inverse matrix
        inverse_kernel = torch.inverse(forward_kernel)

        # create target coordinate matrix
        HW = target_height * target_width
        y, x = torch.meshgrid([torch.arange(0, target_height), 
            torch.arange(0, target_width)])
        x = x.reshape(HW, 1).float() + 0.5
        y = y.reshape(HW, 1).float() + 0.5
        y = y * 2 / (target_height) - 1
        x = x * 2 / (target_width) - 1
        target_coordinate = torch.cat([x, y], dim = 1) # convert from (y, x) to (x, y)
        # print(x.shape)
        del x
        del y
        target_coordinate_partial_repr = compute_partial_repr(target_coordinate, target_control_points)
        target_coordinate_repr = torch.cat([
            target_coordinate_partial_repr, torch.ones(HW, 1), target_coordinate
        ], dim = 1)

        padding_matrix = torch.zeros(3, 2)

        self.inverse_kernel = inverse_kernel
        self.padding_matrix = padding_matrix
        self.target_coordinate_repr = target_coordinate_repr

    @property
    def device(self):
        return self.inverse_kernel.device

    def _apply(self, fn):
        super(TPSGridGen, self)._apply(fn)
        self.inverse_kernel = fn(self.inverse_kernel)
        self.padding_matrix = fn(self.padding_matrix)
        self.target_coordinate_repr = fn(self.target_coordinate_repr)
        return self

    def forward(self, source_control_points):
        assert source_control_points.ndimension() == 3
        assert source_control_points.size(1) == self.num_points
        assert source_control_points.size(2) == 2
        batch_size = source_control_points.size(0)

        Y = torch.cat([source_control_points, (self.padding_matrix.expand(batch_size, 3, 2))], 1)
        mapping_matrix = torch.matmul((self.inverse_kernel), Y)
        source_coordinate = torch.matmul((self.target_coordinate_repr), mapping_matrix)
        return source_coordinate


@INTERMEDIATE_AUGMENTOR_REGISTRY.register()
class TpsAugmentor(IntermediateAugmentor):
    """
    Transformation with thin plate spline
    """
    @configurable
    def __init__(
        self,
        *,
        ctrl_pts_size,
        max_image_size,
        warp_range,
        stop_threshold,
        max_iter,
    ):
        super().__init__(False)

        target_control_points = torch.Tensor(list(itertools.product(
            torch.arange(-1.0, 1.000001, 2.0 / ctrl_pts_size[0]),
            torch.arange(-1.0, 1.000001, 2.0 / ctrl_pts_size[1]),
        ))).float()

        self.warp_range = warp_range
        self.stop_threshold = stop_threshold
        self.max_iter = max_iter
        self.max_image_size = max_image_size
        self.max_image_size_xy = torch.tensor([max_image_size[1], max_image_size[0]])

        self.tps_grid_generator = TPSGridGen(*max_image_size, target_control_points)
        self.target_control_points = target_control_points
    
    def _apply(self, fn):
        super(TpsAugmentor, self)._apply(fn)
        self.tps_grid_generator = self.tps_grid_generator._apply(fn)
        self.target_control_points = fn(self.target_control_points)
        self.max_image_size_xy = fn(self.max_image_size_xy)
        return self
    
    @classmethod
    def from_config(cls, cfg):
        return {
            "ctrl_pts_size": cfg.INTERMEDIATE_AUGMENTOR.TpsTransformer.CTRL_PTS_SIZE,
            "max_image_size": cfg.INTERMEDIATE_AUGMENTOR.MAX_IMAGE_SIZE,
            "warp_range": cfg.INTERMEDIATE_AUGMENTOR.TpsTransformer.WARP_RANGE,
            "stop_threshold": cfg.INTERMEDIATE_AUGMENTOR.TpsTransformer.STOP_THRESHOLD,
            "max_iter": cfg.INTERMEDIATE_AUGMENTOR.TpsTransformer.MAX_ITER,
        }

    def apply_image(self, image):
        return F.grid_sample(image.unsqueeze(0), self.grid, align_corners=False)[0]
    
    def apply_coords(self, coords):
        device = coords.device
        coords = coords * (2.0 / self.max_image_size_xy) - 1.0

        warped_coords0 = coords.clone()
        converged, i = False, 0
        while not converged:
            coords_partial_repr = compute_partial_repr(warped_coords0, self.target_control_points)
            coords_repr = torch.cat([
                coords_partial_repr, torch.ones(coords_partial_repr.shape[0], 1, device=device), warped_coords0
            ], dim = 1)

            Y = torch.cat([self.source_control_points.unsqueeze(0), (self.tps_grid_generator.padding_matrix.expand(1, 3, 2))], 1)
            mapping_matrix = torch.matmul((self.tps_grid_generator.inverse_kernel), Y)
            warped_coords1 = torch.matmul((coords_repr), mapping_matrix)[0]
            coord_dev = warped_coords1 - coords
            warped_coords0 = warped_coords0 - coord_dev
            i+=1
            converged = i > self.max_iter or torch.max(torch.abs(coord_dev)) * max(self.max_image_size) < self.stop_threshold

        if i > self.max_iter:
            print("Failed to converge. l_inf norm is: ", (torch.max(torch.abs(coord_dev)) * max(self.max_image_size)).item())

        return (warped_coords0 + 1.0) * (self.max_image_size_xy / 2.0)

    def generate_params(self, image, gt_instances, strength=None):
        image_size, device = image.shape[-2:], image.device
        self.image_size_xy = torch.tensor(
            [image_size[1], image_size[0]], device=device)
        if device != self.tps_grid_generator.device:
            print(device, "is not equal to", self.tps_grid_generator.device)
            self.to(device)

        ctrl_pts_displacement = sample_param(
            self.warp_range, strength=strength,
            training=self.training, device=device)

        source_control_points = self.target_control_points \
            + (torch.rand(self.target_control_points.size(), device=device) * 2 - 1) * ctrl_pts_displacement
        source_control_points[self.target_control_points <= -1 + ctrl_pts_displacement] = -1 - ctrl_pts_displacement
        source_control_points[self.target_control_points >= 1 - ctrl_pts_displacement] = 1 + ctrl_pts_displacement
        self.source_control_points = source_control_points

        source_coordinate = self.tps_grid_generator(torch.unsqueeze(source_control_points, 0))
        grid_cropped = source_coordinate.view(1, *self.max_image_size, 2)[:,:image_size[0], :image_size[1]]
        # Renormalize the grid
        self.grid = ((grid_cropped + 1.0) * (self.max_image_size_xy / (self.image_size_xy * 2.0))) * 2.0 - 1.0
