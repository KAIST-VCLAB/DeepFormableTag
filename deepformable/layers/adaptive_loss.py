# Copyright (c) Mustafa B. Yaldiz (VCLAB, KAIST) All Rights Reserved.
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from detectron2.utils.comm import get_world_size, is_main_process
from .dist_ops import AllReduce


class AdaptiveLoss(nn.Module):
    """
    This class is implemented to filter the loss values that cause exploding gradients.
    """
    def __init__(
        self,
        n=3.0,
        beta=0.995,
        beta2=0.999,
        loss_type='bce',
        adaptive_on=True,
    ):
        super().__init__()
        self.register_buffer("n", torch.tensor(n, dtype=torch.float32))
        self.register_buffer("beta", torch.tensor(beta, dtype=torch.float32))
        self.register_buffer("beta2", torch.tensor(beta2, dtype=torch.float32))
        self.register_buffer("running_mean", torch.tensor(-1, dtype=torch.float32))
        self.register_buffer("running_std", torch.tensor(-1, dtype=torch.float32))
        if loss_type == 'bce':   
            self.loss = F.binary_cross_entropy_with_logits
        elif loss_type == 'l1':
            self.loss = F.l1_loss
        elif loss_type == 'l2' or loss_type == 'mse':
            self.loss = F.mse_loss
        else:
            raise "Unknown loss type!"
        self.adaptive_on = adaptive_on
    
    def forward(self, input, target):
        if not self.adaptive_on:
            return self.loss(input, target, reduction='sum')
        loss_values = self.loss(input, target, reduction='none')
        
        threshold = self.running_mean + self.n * self.running_std
        #loss_filtered = loss_values[loss_values > threshold].detach()
        loss_filtered = torch.clamp(loss_values[loss_values > threshold], 0, threshold.item())
        loss_passed = loss_values[loss_values <= threshold]
        loss_final = torch.sum(loss_filtered) + torch.sum(loss_passed)

        mean, meansqr = loss_values.mean(), torch.mean(loss_values * loss_values)
        world_size = get_world_size()
        if world_size != 1:
            vec = torch.cat([mean.view(1), meansqr.view(1)])
            mean, meansqr = (AllReduce.apply(vec) * (1.0 / world_size)).detach()
        std = torch.sqrt(meansqr - mean * mean)
        
        if self.running_mean > 0:
            mean_step = min(mean * (1.0 - self.beta), self.running_std * 0.75)
            self.running_mean = self.running_mean * self.beta + mean_step
            std_step =  min(std * (1.0 - self.beta2), self.running_std * 0.25)
            self.running_std = self.running_std * self.beta2 + std_step
        else:
            self.running_mean = mean * 1.5
            self.running_std = std

        # if is_main_process():
        #     print("Mean:", self.running_mean, "Std:", self.running_std)
        #     if len(loss_filtered) != 0:
        #         print("Filtered:", loss_filtered, 
        #             "-- Running_val:", self.running_mean, self.running_std,
        #             "-- Calc val:", mean, std,)
        # loss_final = F.binary_cross_entropy_with_logits(
        #     input, target, reduction='sum')
        return loss_final
