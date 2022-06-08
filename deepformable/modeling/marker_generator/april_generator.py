# Copyright (c) Mustafa B. Yaldiz (VCLAB, KAIST) All Rights Reserved.
import logging
import numpy as np
import cv2
from cv2 import aruco
from apriltag import apriltag

from detectron2.config import configurable

from .build import MARKER_GENERATOR_REGISTRY
from deepformable.utils import get_aruco_dict
from .aruco_generator import ArucoGenerator


@MARKER_GENERATOR_REGISTRY.register()
class AprilGenerator(ArucoGenerator):
    @configurable
    def __init__(
        self,
        *,
        april_dict,
        border_bits,
        num_classes,
        shuffling,
        vis_period=0
    ):
        super().__init__(
            aruco_dict=april_dict, 
            border_bits=border_bits, 
            num_classes=num_classes, 
            shuffling=shuffling, 
            vis_period=vis_period)
        self.detector = apriltag("tag36h11")
        

    @classmethod
    def from_config(cls, cfg):
        april_dict = get_aruco_dict(cfg.MODEL.MARKER_GENERATOR.ARUCO_DICT, default=aruco.DICT_APRILTAG_36h11)
        shuffling = cfg.MODEL.ROI_DECODER_HEAD.DECODER_ON
        return {
            "april_dict": april_dict,
            "border_bits": cfg.MODEL.MARKER_GENERATOR.BORDER_BITS,
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "shuffling": shuffling,
            "vis_period": cfg.VIS_PERIOD,
        }

    def recognize(self, img):
        if len(img.shape) == 3:
            img_ = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif len(img.shape) == 2:
            img_ = img

        ### opencv
        # marker_corners, ids, _ = cv2.aruco.detectMarkers(img, cv2.aruco.getPredefinedDictionary(
        #     cv2.aruco.DICT_APRILTAG_36H11))
        ###

        ### AprilRobotics
        detections = self.detector.detect(img_)
        marker_corners = []
        ids = []
        for i in detections:
            corners = np.array(i['lb-rb-rt-lt'])
            corners_ = corners.copy()

            corners_[0] = corners[1]
            corners_[1] = corners[0]
            corners_[2] = corners[3]
            corners_[3] = corners[2]

            marker_corners.append(np.array([corners_], dtype=np.float32))
            ids.append(i["id"])
        ids = np.array(ids)
        ###

        marker_corners = np.array(marker_corners).reshape(-1,4,2)
        return marker_corners, ids