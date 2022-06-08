# Copyright (c) Mustafa B. Yaldiz (VCLAB, KAIST) All Rights Reserved.
import logging
import numpy as np
import cv2
from cv2 import aruco

import torch
from torch import nn
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.utils.comm import all_gather, is_main_process
from detectron2.utils.events import get_event_storage

from .build import MARKER_GENERATOR_REGISTRY, MarkerGenerator
from deepformable.utils import get_aruco_dict


@MARKER_GENERATOR_REGISTRY.register()
class FixedGenerator(MarkerGenerator):
    @configurable
    def __init__(
        self,
        *,
        marker_size,
        border_bits,
        num_classes,
        init_method="uniform",
        init_std=1.4,
        out_channels=3,
        vis_period=0,
    ):
        super().__init__(num_classes, 1, vis_period)
        self.marker_size = marker_size
        self.border_bits = border_bits
        generation_size = self.marker_size - self.border_bits * 2
        self.markers = nn.Parameter(
            torch.empty(
                self.num_classes, out_channels, generation_size, generation_size, 
                requires_grad=True))
        if init_method == "uniform":
            nn.init.uniform(self.markers, -init_std, init_std)
        elif init_method == "kaiming":
            nn.init.kaiming_normal_(self.markers, mode="fan_out", nonlinearity="sigmoid")
        else:
            nn.init.normal_(self.markers, std=init_std)

        nn.init.normal_(self.markers, std=init_std)
        self.activation = nn.Sigmoid()

    @classmethod
    def from_config(cls, cfg):
        return {
            "marker_size": cfg.MODEL.MARKER_GENERATOR.MARKER_SIZE[0],
            "border_bits": cfg.MODEL.MARKER_GENERATOR.BORDER_BITS,
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "init_method": cfg.MODEL.MARKER_GENERATOR.INIT_METHOD.lower(),
            "init_std": cfg.MODEL.MARKER_GENERATOR.INIT_STD,
            "vis_period": cfg.VIS_PERIOD,
        }

    def message_generator(self):
        return None

    def batch_marker_generator(self, gt_classes_batch):
        markers = self.activation(self.markers)
        markers = F.pad(markers, [self.border_bits]*4)    # Add padding

        markers_batch, messages_batch = [], []
        for classes in gt_classes_batch:
            markers_batch.append(markers[classes])
            messages_batch.append(torch.zeros(len(classes)))
        
        return markers_batch, messages_batch, {}


@MARKER_GENERATOR_REGISTRY.register()
class ArucoGenerator(MarkerGenerator):
    @configurable
    def __init__(
        self,
        *,
        aruco_dict,
        border_bits,
        num_classes,
        shuffling,
        vis_period=0
    ):
        super().__init__(num_classes, aruco_dict.markerSize * aruco_dict.markerSize, vis_period)
        marker_size = aruco_dict.markerSize + 2 * border_bits
        self.aruco_dict = aruco_dict
        self.border_bits = border_bits
        self.num_markers = len(aruco_dict.bytesList)
        self.marker_size = marker_size
        self.shuffling = shuffling
        self.detect_params = aruco.DetectorParameters_create()
        self.subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)
        
        markers_binary = []
        for i in range(self.num_markers):
            marker = aruco_dict.drawMarker(i, marker_size, borderBits=1)
            markers_binary.append(marker[1:-1, 1:-1].reshape(-1))
            
        # self.register_buffer("markers", torch.tensor(
        #     markers, dtype=torch.float32).view(num_classes, 3, marker_size, marker_size)/255.0)
        self.register_buffer("markers_binary", torch.tensor(markers_binary, dtype=torch.float32)/255.0, False)
        self.register_buffer("messages", self.markers_binary[:num_classes], False)

    @classmethod
    def from_config(cls, cfg):
        aruco_dict = get_aruco_dict(cfg.MODEL.MARKER_GENERATOR.ARUCO_DICT)
        shuffling = cfg.MODEL.ROI_DECODER_HEAD.DECODER_ON
        return {
            "aruco_dict": aruco_dict,
            "border_bits": cfg.MODEL.MARKER_GENERATOR.BORDER_BITS,
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "shuffling": shuffling,
            "vis_period": cfg.VIS_PERIOD,
        }

    def visualize_training(self):
        if self.training and self.vis_period != 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                marker = np.uint8(self.get_markers_numpy(0) * 255)
                marker = marker.transpose(2, 0, 1)
                storage.put_image("Generated Marker", marker)
    
    def message_generator(self):
        if self.shuffling and self.training:
            message_indexes = None
            if is_main_process():
                message_indexes = torch.randperm(self.num_markers)[:self.num_classes]
            message_indexes = all_gather(message_indexes)[0].to(self.device)
            return self.markers_binary[message_indexes]
        return self.markers_binary[:self.num_classes]

    def batch_marker_generator(self, gt_classes_batch):
        markers_batch, messages_batch = [], []
        for classes in gt_classes_batch:
            messages = self.messages[classes]
            messages_batch.append(messages)
            
            markers = torch.repeat_interleave(
                messages.view(-1, 1, self.aruco_dict.markerSize, self.aruco_dict.markerSize), 3, dim=1)
            markers = F.pad(markers, [self.border_bits]*4)
            markers_batch.append(markers)
        
        return markers_batch, messages_batch, {}
    
    def recognize(self, img):
        marker_corners, ids, _ = cv2.aruco.detectMarkers(
            img, self.aruco_dict, parameters=self.detect_params)
        if len(img.shape) == 3:
            gray_undistorted = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray_undistorted = img
        for corners in marker_corners:
            cv2.cornerSubPix(gray_undistorted, corners,
                                winSize=(3, 3),
                                zeroZone=(-1, -1),
                                criteria=self.subpix_criteria)

        marker_corners = np.array(marker_corners).reshape(-1,4,2)+0.5
        ids = ids.reshape(-1) if ids is not None else np.ones(len(marker_corners))
        return marker_corners, ids