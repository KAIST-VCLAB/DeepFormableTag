"""
Some layers here are copied from facebookresearch/pytorch_GAN_zoo: https://github.com/facebookresearch/pytorch_GAN_zoo
Our marker generator builds on top of those layers.
Copyright (c) Mustafa B. Yaldiz (VCLAB, KAIST) All Rights Reserved.
"""
import logging
import numpy as np
from sklearn.neighbors import KDTree
import math

import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F

import fvcore.nn.weight_init as weight_init
from detectron2.config import configurable
from detectron2.layers import cat, Conv2d
from detectron2.utils.comm import all_gather, get_world_size, get_rank
from .build import MARKER_GENERATOR_REGISTRY, MarkerGenerator

from deepformable.layers import MarkerGatherFunc


def getLayerNormalizationFactor(x):
    size = x.weight.size()
    fan_in = math.prod(size[1:])
    return math.sqrt(2.0 / fan_in)


class PixelNormLayer(nn.Module):
    def __init__(self):
        super(PixelNormLayer, self).__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        x, _ = x
        return x * (((x**2).mean(dim=1, keepdim=True) + self.epsilon).rsqrt())


class NoNormLayer(nn.Module):
    def __init__(self):
        super(NoNormLayer, self).__init__()
    def forward(self, x):
        return x[0]


class ConstrainedLayer(nn.Module):
    def __init__(self,
                 module,
                 equalized=True,
                 lrMul=1.0,
                 initBiasToZero=True):
        super(ConstrainedLayer, self).__init__()

        self.module = module
        self.equalized = equalized

        if initBiasToZero:
            self.module.bias.data.fill_(0)
        if self.equalized:
            self.module.weight.data.normal_(0, 1)
            self.module.weight.data /= lrMul
            self.weight = getLayerNormalizationFactor(self.module) * lrMul

    def forward(self, x):

        x = self.module(x)
        if self.equalized:
            x *= self.weight
        return x


class EqualizedConv2d(ConstrainedLayer):
    def __init__(self,
                 nChannelsPrevious,
                 nChannels,
                 kernelSize,
                 padding=0,
                 bias=True,
                 equalized=False,
                 padding_mode='zeros',
                 **kwargs):
        ConstrainedLayer.__init__(self,
                                  nn.Conv2d(nChannelsPrevious, nChannels,
                                            kernelSize, padding=padding,
                                            bias=bias, padding_mode=padding_mode),
                                  equalized=equalized,
                                  **kwargs)
        if not equalized:
            weight_init.c2_msra_fill(self.module)
        

class EqualizedLinear(ConstrainedLayer):
    def __init__(self,
                 nChannelsPrevious,
                 nChannels,
                 bias=True,
                 equalized=False,
                 **kwargs):
        ConstrainedLayer.__init__(self,
                                  nn.Linear(nChannelsPrevious, nChannels,
                                  bias=bias), equalized=equalized,**kwargs)
        if not equalized:
            weight_init.c2_xavier_fill(self.module)


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim, equalized=False):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = EqualizedLinear(style_dim, in_channel * 2, equalized=equalized)

        self.style.module.bias.data[:in_channel] = 1
        self.style.module.bias.data[in_channel:] = 0

    def forward(self, x):
        x, w = x
        style = self.style(w).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)
        out = self.norm(x)
        return gamma * out + beta


class BlockGenerator(nn.Module):
    def __init__(
        self,
        upsample_type="bilinear",
        upsample_scale=2,
        norm_type="adain",
        activation_type="leaky",
        residual=True,
        input_channels=16,
        conv_dims=[16,16],
        style_dim=128,
        kernel_size=3,
        padding=1,
        equalized=False,
        padding_mode='zeros'
    ):
        super().__init__()
        self.residual = residual
        if upsample_type == 'nearest' or upsample_type == 'bilinear':
            self.upsampler = nn.Upsample(scale_factor=upsample_scale, mode=upsample_type, align_corners=False)
        elif upsample_type == 'transpose':
            self.upsampler = nn.ConvTranspose2d(
                input_channels, input_channels, 3, stride=upsample_scale,
                 padding=1, output_padding=upsample_scale-1)
        else:
            raise "Unknown upsampler!"

        if activation_type.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation_type.lower() == "leaky":
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        else:
            raise "Unknown activation!"

        conv_layers, norm_layers = [], []
        for i in range(len(conv_dims)):
            conv_layers.append(EqualizedConv2d(
                    input_channels, conv_dims[i], kernel_size,
                    padding=padding, equalized=equalized,
                    padding_mode=padding_mode))
            input_channels = conv_dims[i]
            self.add_module("conv{}".format(i + 1), conv_layers[-1])

            if norm_type.lower() == "none":
                norm_layers.append(NoNormLayer())
            elif norm_type.lower() == "pixelnorm":
                norm_layers.append(PixelNormLayer())
            elif norm_type.lower() == "adain":
                norm_layers.append(AdaptiveInstanceNorm(input_channels, style_dim))
            else:
                raise "Unknown normalization layer!"
            self.add_module("norm{}".format(i + 1), norm_layers[-1])

        self.conv_layers, self.norm_layers = conv_layers, norm_layers
    
    def forward(self, x):
        x, w = x
        upsampled_x = x = self.upsampler(x)
        for conv, norm in zip(self.conv_layers, self.norm_layers):
            x = conv(x)
            x = norm((x,w))
            x = self.activation(x)
        if self.residual:
            x = torch.cat([x, upsampled_x], dim=1)
        return x, w


class LinearGenerator(nn.Module):
    def __init__(
        self,
        activation_type="leaky",
        input_channels=36,
        fc_dims=[256,128,128],
        style_on=True,
        initial_size=4,
        equalized=False,
    ):
        super().__init__()
        
        if activation_type.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation_type.lower() == "leaky":
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        else:
            raise "Unknown activation!"
        
        self.initial_size = initial_size
        self.style_on = style_on
        # First fc layer with channel-wise normalization
        self.initial_dim = fc_dims[0]
        assert self.initial_dim % initial_size*initial_size == 0, "Initial fc dim should be resizable to 4,4 feature"
        self.fc_initial = EqualizedLinear(input_channels, self.initial_dim, equalized=equalized)
        self.initial_norm = PixelNormLayer()
        
        fc_layers = []
        input_channels = self.initial_dim
        for i in range(1,len(fc_dims)):
            fc_layers.append(
                EqualizedLinear(input_channels, fc_dims[i], equalized=equalized))
            input_channels = fc_dims[i]
            self.add_module("fc{}".format(i + 1), fc_layers[-1])
        self.fc_layers = fc_layers
    
    def forward(self, x):
        x = self.fc_initial(x)
        x = self.initial_norm((x,None))
        x = self.activation(x)
        w = x

        for fc_layer in self.fc_layers:
            w = fc_layer(w)
            w = self.activation(w)
        
        if self.style_on:
            x = x.view(
                -1, self.initial_dim//(self.initial_size*self.initial_size),
                self.initial_size, self.initial_size)
        else:
            x = w.view(
                -1,
                self.initial_dim//(self.initial_size*self.initial_size),
                self.initial_size, self.initial_size)

        return x, w


class KDTreeClassPredictor:
    """
    Example Usage:
    predictor = KDTreeClassPredictor(self.marker_generator.messages)
    predictor.query(model_out[0]['instances'].decoded_messages_scores)
    """
    def __init__(self, messages, metric='manhattan'):
        if isinstance(messages, torch.Tensor):
            messages = messages.detach().cpu().numpy()
        self.kdt = KDTree(messages, metric=metric)

    def query(self, query_messages):
        if isinstance(query_messages, torch.Tensor):
            query_messages = query_messages.detach().cpu().numpy()
        scores, ids = self.kdt.query(query_messages, k=1)
        return (1.0-scores/query_messages.shape[1])[:,0], ids[:,0]


@MARKER_GENERATOR_REGISTRY.register()
class GeneralizedGenerator(MarkerGenerator):
    @configurable
    def __init__(
        self,
        *,
        border_bits,
        num_classes,
        num_bits,
        initial_size=4,
        conv_hidden_dims=[[16],[12],[8,8]],
        fc_hidden_dims=[256,128,128],
        upsample_type="bilinear",
        upsample_scale=2,
        norm_type="adain",
        activation_type="leaky",
        residual=True,
        vis_period=0,
        equalized=False,
        padding_mode='zeros',
        final_conv_kernel_size=3,
        # out_channels=3,
    ):
        super().__init__(num_classes, num_bits, vis_period)
        self.border_bits = border_bits
        self.initial_size = initial_size
        
        self.world_size = get_world_size()
        self.rank = get_rank()
        assert num_classes % self.world_size == 0, "Number of classes should be divisible to number of workers"
        self.num_markers_per_worker = num_classes // self.world_size
        self.markers = None

        self.group, self.backend = None, None
        if self.world_size != 1:
            self.group = dist.new_group()
            self.backend = dist.get_backend(self.group)
        
        self.linear_generator = LinearGenerator(
            activation_type, num_bits, fc_hidden_dims, style_on=norm_type=="adain", equalized=equalized)
        
        input_channels = fc_hidden_dims[0]//(initial_size*initial_size)
        block_layers = []
        for i in range(len(conv_hidden_dims)):
            conv_dims = conv_hidden_dims[i]
            block_layers.append(
                BlockGenerator(
                    upsample_type, upsample_scale, norm_type,
                    activation_type, residual, input_channels,
                    conv_dims, fc_hidden_dims[-1], equalized=equalized, padding_mode=padding_mode))
            input_channels = conv_dims[-1] + residual * input_channels
        self.block_layers = nn.Sequential(*block_layers)
                
        # Output convolutions
        self.final_conv = Conv2d(input_channels, 3, 
            kernel_size=final_conv_kernel_size, padding=final_conv_kernel_size//2,
            bias=True, activation=torch.sigmoid)
        nn.init.kaiming_normal_(self.final_conv.weight, mode="fan_out", nonlinearity="sigmoid")
        nn.init.constant_(self.final_conv.bias, 0)
        self.register_buffer("_messages", self.message_generator(), False)

    @classmethod
    def from_config(cls, cfg):
        return {
            "border_bits": cfg.MODEL.MARKER_GENERATOR.BORDER_BITS,
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "num_bits": cfg.MODEL.MARKER_GENERATOR.NUM_GENERATION_BITS,
            "initial_size": cfg.MODEL.MARKER_GENERATOR.INITIAL_SIZE,
            "conv_hidden_dims": cfg.MODEL.MARKER_GENERATOR.CONV_DIMS,
            "fc_hidden_dims": cfg.MODEL.MARKER_GENERATOR.FC_DIMS,
            "upsample_type": cfg.MODEL.MARKER_GENERATOR.UPSAMPLE_TYPE,
            "upsample_scale": cfg.MODEL.MARKER_GENERATOR.UPSAMPLE_SCALE,
            "norm_type": cfg.MODEL.MARKER_GENERATOR.NORM_TYPE,
            "activation_type": cfg.MODEL.MARKER_GENERATOR.ACTIVATION_TYPE,
            "residual":cfg.MODEL.MARKER_GENERATOR.RESIDUAL,
            "equalized":cfg.MODEL.MARKER_GENERATOR.EQUALIZED,
            "padding_mode": cfg.MODEL.MARKER_GENERATOR.PADDING_MODE,
            "final_conv_kernel_size": cfg.MODEL.MARKER_GENERATOR.FINAL_CONV_KERNEL_SIZE,
            "vis_period": cfg.VIS_PERIOD,
        }

    @property
    def messages(self):
        return self._messages
    
    @messages.setter
    def messages(self, value):
        if value is not None:
            if isinstance(value, list):
                value = np.array(value)
            if isinstance(value, torch.Tensor):
                self._messages = value
                messages = value.detach().cpu().numpy()
            if isinstance(value, np.ndarray):
                self._messages = torch.tensor(
                    value, device=self.device, dtype=torch.float32)
                messages = value
            self.messages_kdtree = KDTreeClassPredictor(messages)
            self.num_classes = len(value)
        else:
            self._messages = None

    @torch.no_grad()
    def message_generator(self):
        if self.training:
            counter, random_max_trial = 0, 10
            bits_unique = torch.zeros((0,self.num_bits), device=self.device)
            while True:
                bits_batch = torch.randint(2, (self.num_markers_per_worker, self.num_bits), device="cpu")
                bits_remain = torch.cat(all_gather(bits_batch), dim=0).to(self.device)
                bits_unique = torch.cat([bits_unique, bits_remain], dim=0).to(self.device)
                bits_unique = torch.unique(bits_unique, dim=0)[:self.num_classes]
                if len(bits_unique) == self.num_classes:
                    bits_all = bits_unique
                    break
                counter += 1
                if counter == random_max_trial:
                    raise "{} trials could not create enough random bits".format(random_max_trial)
                    
            bits_all = bits_all.to(torch.float32)
            return bits_all
        return self.messages

    def postprocessing(self, pred_instances):
        if self.messages is not None and len(pred_instances) != 0:
            device, dtype = pred_instances.decoded_messages_confidence.device, pred_instances.decoded_messages_confidence.dtype
            # scores, classes = self.messages_kdtree.query(pred_instances.decoded_messages)
            # Paper implementation uses above line instead, 
            # but we believe confidence score metric is more robust this way, it may change test results slightly.
            # And you may need slightly lower threshold for this implementation.
            scores, classes = self.messages_kdtree.query(pred_instances.decoded_messages_confidence)
            pred_instances.pred_classes = torch.tensor(classes, device=device)
            pred_instances.gt_closest_messages = self.messages[pred_instances.pred_classes]
            pred_instances.gt_closest_similarity_score = 1.0 - torch.mean(torch.abs(
                pred_instances.decoded_messages - pred_instances.gt_closest_messages), dim=-1)
            pred_instances.bit_similarity = torch.tensor(scores, device=device, dtype=dtype)
        return pred_instances

    def markers_forward(self, x):
        x = self.linear_generator(x)
        x = self.block_layers(x)
        x = self.final_conv(x[0])
        x = F.pad(x, [self.border_bits]*4)
        return x
    
    def batch_marker_generator(self, gt_classes_batch):
        markers_batch, messages_batch = [], []
        if self.training:
            self.markers = None
            batch_messages = self.messages[self.rank*self.num_markers_per_worker:(self.rank+1)*self.num_markers_per_worker]
            markers = self.markers_forward(batch_messages)
            all_markers = MarkerGatherFunc.apply(
                markers, self.rank, self.group, self.world_size, self.backend)
            
            for classes in gt_classes_batch:
                messages = self.messages[classes]
                messages_batch.append(messages)
                markers_batch.append(all_markers[classes])
        else:
            markers = self.markers
            if markers is None:
                # gt_classes = cat([c for c in gt_classes_batch], dim=0)
                # unique_classes, inverse_classes = torch.unique(gt_classes, return_inverse=True)
                # selected_messages = self.messages[unique_classes]
                # markers = self.markers_forward(selected_messages)
                markers = self.markers_forward(self.messages)
                self.markers = markers
            
            for classes in gt_classes_batch:
                messages = self.messages[classes]
                messages_batch.append(messages)
                markers_batch.append(markers[classes])
                # markers_batch.append(markers[inverse_classes[:len(classes)]])
                # inverse_classes = inverse_classes[len(classes):]
        return markers_batch, messages_batch, {}