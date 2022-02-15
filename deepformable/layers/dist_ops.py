"""
This code references https://github.com/ag14774/diffdist/blob/b5c17c7354bbbe98b6e8a791ea78614861b4997a/diffdist/
It is primarily used to distribute marker generation task across GPUs.
Copyright (c) Mustafa B. Yaldiz (VCLAB, KAIST) All Rights Reserved.
"""
import torch
import torch.distributed as dist
from torch.autograd import Function

from detectron2.utils.comm import get_world_size, get_rank

class MarkerGatherFunc(Function):
    @staticmethod
    def forward(ctx, markers, rank, group, world_size, backend):
        ctx.backend, ctx.marker_size = backend, len(markers)
        ctx.world_size, ctx.rank, ctx.group = world_size, rank, group
        if world_size == 1:
            return markers
        gather_list = [torch.zeros_like(markers, device=markers.device) for _ in range(world_size)]
        if backend == 'nccl':
            gather_list = [gather_list]
            dist.all_gather_multigpu(gather_list, [markers], group=group)
            gather_list = gather_list[0]
        else:
            dist.all_gather(gather_list, markers, group=group)
            gather_list = [i.to(markers.device) for i in gather_list]
        return torch.cat(gather_list, dim=0)

    @staticmethod
    def backward(ctx, markers_grad):
        if ctx.world_size == 1:
            return markers_grad, None, None, None, None
        if ctx.backend == 'nccl':
            markers_grad = [markers_grad]
            dist.all_reduce_multigpu(markers_grad, group=ctx.group)
            markers_grad = markers_grad[0]
        else:
            dist.all_reduce(markers_grad, group=ctx.group)
        return markers_grad[ctx.marker_size*ctx.rank:ctx.marker_size*(ctx.rank+1)], None, None, None, None


class AllReduce(Function):
    @staticmethod
    def forward(ctx, input):
        input_list = [torch.zeros_like(input) for k in range(dist.get_world_size())]
        # Use allgather instead of allreduce since I don't trust in-place operations ..
        dist.all_gather(input_list, input, async_op=False)
        inputs = torch.stack(input_list, dim=0)
        return torch.sum(inputs, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        dist.all_reduce(grad_output, async_op=False)
        return grad_output