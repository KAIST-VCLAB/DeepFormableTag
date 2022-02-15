# Copyright (c) Mustafa B. Yaldiz (VCLAB, KAIST) All Rights Reserved.
import torch
from torch import nn
import numpy as np

from detectron2.utils.registry import Registry
from detectron2.utils.events import get_event_storage
from abc import ABCMeta, abstractmethod

MARKER_GENERATOR_REGISTRY = Registry("MARKER_GENERATOR")  # noqa F401 isort:skip
MARKER_GENERATOR_REGISTRY.__doc__ = """
Registry for the marker generator architecture
"""


def build_marker_generator(cfg):
    """
    Build the marker generator pipeline, defined by ``cfg.MODEL.MARKER_GENERATOR``.
    """
    marker_generator = cfg.MODEL.MARKER_GENERATOR.NAME
    generator = MARKER_GENERATOR_REGISTRY.get(marker_generator)(cfg)
    generator.to(torch.device(cfg.MODEL.DEVICE))
    return generator


class MarkerGenerator(nn.Module, metaclass=ABCMeta):
    def __init__(self, num_classes, num_bits, vis_period=0):
        super().__init__()
        self.num_classes = num_classes
        self.num_bits = num_bits
        self.vis_period = vis_period
        self.register_buffer("gamma", torch.tensor((1/2.2), dtype=torch.float32), False)
        # self.register_buffer("messages", torch.zeros(num_classes, num_bits, dtype=torch.float32), False)

    @property
    def device(self):
        return self.gamma.device

    @torch.no_grad()
    def get_markers_numpy(self, classes):
        if isinstance(classes, int):
            classes = [classes]
        classes = torch.tensor(classes, device=self.device)
        markers_batch, _, _ = self.batch_marker_generator([classes])
        markers = markers_batch[0] ** self.gamma
        markers = markers.permute(0,2,3,1)[...,[2,1,0]].cpu().numpy()
        if len(markers) == 1:
            return markers[0]
        return markers

    @abstractmethod
    def message_generator(self):
        """
        This method should save the internal variables required for batch_marker_generator
        """
        pass

    @abstractmethod
    def batch_marker_generator(self, gt_classes_batch):
        """
        This method returns markers and messages in list form
        """
        pass

    def postprocessing(self, pred_instances):
        return pred_instances

    def visualize_training(self):
        if self.training and self.vis_period != 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                if self.messages is not None:
                    messages = torch.zeros(2, self.num_bits, dtype=torch.float32)
                    messages[0, range(0,self.num_bits,2)] = 1.0
                    messages[1, range(1,self.num_bits,2)] = 1.0
                    old_messages = self.messages[:2].clone()
                    self.messages[:2] = messages
                    markers = np.uint8(self.get_markers_numpy([0, 1]) * 255)
                    self.messages[:2] = old_messages
                    marker1, marker2 = markers[0].transpose(2,0,1), markers[1].transpose(2,0,1)
                else:
                    # TODO: What is this? :)
                    markers = np.uint8(self.get_markers_numpy([0, 1]) * 255)
                    marker1, marker2 = markers[0].transpose(2,0,1), markers[1].transpose(2,0,1)
                storage.put_image("Marker (101010..)", marker1)
                storage.put_image("Marker (010101..)", marker2)

    def forward(self, batch_instances):
        with torch.no_grad():
            self.visualize_training()
            self.messages = self.message_generator()
            if self.messages is not None:
                self.messages = self.messages.to(self.device)
        
        gt_classes_batch = [i.gt_classes for i in batch_instances]
        markers_batch, messages_batch, loss = self.batch_marker_generator(gt_classes_batch)
        
        for messages, instances in zip(messages_batch, batch_instances):
            instances.gt_message = messages

        return markers_batch, loss