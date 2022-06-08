# Copyright (c) Mustafa B. Yaldiz (VCLAB, KAIST) All Rights Reserved.
import copy
import logging
from typing import List, Optional, Union

import numpy as np
import cv2
import torch

from detectron2.config import configurable
from detectron2.structures import Boxes, BoxMode, Instances, PolygonMasks
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from detectron2.data import DatasetMapper as DetectronMapper

from deepformable.utils import is_polygon_intersects, marker_placer
from deepformable.utils.board_utils import image_placer

def calc_additional_annotation(ann, transforms, sampling_size=8):
    if "segmentation" in ann:
        # We assume there are 4 segmentation points
        polygons = [np.asarray(p).reshape(-1, 2) for p in ann["segmentation"]]
        # Pytorch sampling convention
        pts_src = np.array([[-1,-1],[1,-1],[1,1],[-1,1]])
        pts_dest = transforms.apply_polygons(polygons)[0].reshape(-1,2)

        # homography
        h = cv2.findHomography(pts_src, pts_dest)[0]
        ann["homography"] = h

        # Add middle points of segmentation
        s = np.linspace(-1, 1, 3)
        seg_obj = np.stack(np.meshgrid(s, s, 1), axis=2).reshape(-1,3)
        seg_screen = np.matmul(seg_obj[[0,1,2,5,8,7,6,3]], h.T)
        seg_screen = (seg_screen / np.expand_dims(seg_screen[:,-1], -1))[:,:-1]
        ann["segmentation"] = [seg_screen.reshape(-1)]

        # Compute sampling locations
        # sx = np.linspace(-1,1,sampling_size[0]+1)[:-1] + (1/sampling_size[0])
        # sy = np.linspace(-1,1,sampling_size[1]+1)[:-1] + (1/sampling_size[1])
        sx = sy = np.linspace(-1,1,sampling_size+1)[:-1] + (1/sampling_size)
        sample_obj = np.stack(np.meshgrid(sx, sy, 1), axis=2).reshape(-1, 3)
        sample_screen = np.matmul(sample_obj, h.T)
        ann["sample_locs"] = (sample_screen / np.expand_dims(sample_screen[:,-1], -1))[:,:-1]

        # Recompute bounding boxes
        ann["bbox"] = [*list(seg_screen.min(axis=0)), *list(seg_screen.max(axis=0))]
        ann["bbox_mode"] = BoxMode.XYXY_ABS

@torch.no_grad()
def annotations_to_instances(
    image_shape, annos
):
    instances = Instances(image_shape)
    if not len(annos):
        return instances

    instances.gt_boxes = Boxes([i["bbox"] for i in annos])
    instances.gt_segm  = torch.tensor([obj["segmentation"][0] for obj in annos], dtype=torch.float32).view(-1,8,2)
    instances.gt_classes = torch.tensor([obj["category_id"] for obj in annos], dtype=torch.int64)
    
    if "sample_locs" in annos[0]:
        instances.gt_sample_locs = torch.tensor([obj["sample_locs"] for obj in annos], dtype=torch.float32)
    if "homography" in annos[0]:
        instances.homography = torch.tensor([obj["homography"] for obj in annos], dtype=torch.float32)
    if "normal" in annos[0]:
        instances.normal = torch.tensor([obj["normal"] for obj in annos], dtype=torch.float32)
    if "view_dir" in annos[0]:
        instances.view_dir = torch.tensor([obj["view_dir"] for obj in annos], dtype=torch.float32)
    if "refl_dir" in annos[0]:
        instances.refl_dir = torch.tensor([obj["refl_dir"] for obj in annos], dtype=torch.float32)
    if "brightness_max" in annos[0]:
        instances.brightness_max = torch.tensor([obj["brightness_max"] for obj in annos], dtype=torch.float32).view(-1,1)
    if "avg_color" in annos[0]:
        instances.avg_color = torch.tensor([obj["avg_color"] for obj in annos], dtype=torch.float32)
    if "board_index" in annos[0]:
        instances.board_index = torch.tensor([obj["board_index"] for obj in annos], dtype=torch.int64)

    return instances


def filter_instances(
    instances,
    box_threshold=0,
    max_instances=0,
):
    image_shape = instances.image_size
    
    filter_mask = instances.gt_boxes.nonempty(threshold=box_threshold)
    filter_mask = filter_mask & (instances.gt_boxes.tensor > 0).all(dim=1)
    filter_mask = filter_mask \
                    & (instances.gt_boxes.tensor < torch.tensor([*image_shape, *image_shape][::-1])).all(dim=1)
    instances = instances[filter_mask]
    if max_instances != 0 and len(instances) > max_instances:
        instances = instances[:max_instances]
    return instances


class DeepformableMapper:
    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        num_classes: int,
        sampling_size: int,
        gamma: float,
        reg_p,
        p_reg_rand,
        placement_marker_min_max,
        box_threshold,
        max_instances,
    ):
        self.is_train = is_train
        self.augmentations = augmentations
        self.image_format = image_format
        self.num_classes = num_classes
        self.sampling_size = sampling_size
        self.gamma = gamma
        self.reg_p, self.p_reg_rand = reg_p, p_reg_rand
        self.placement_marker_min_max = placement_marker_min_max
        self.box_threshold = box_threshold
        self.max_instances = max_instances

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        r = {
            "is_train": is_train,
            "augmentations": utils.build_augmentation(cfg, is_train),
            "image_format": cfg.INPUT.FORMAT,
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "sampling_size": cfg.MODEL.ROI_TRANSFORM_HEAD.TRANSFORMER_RESOLUTION,
            "gamma": cfg.RENDERER.GAMMA,
            "max_instances": cfg.INPUT.MAX_MARKERS_PER_IMAGE,
        }
        if is_train:
            r['reg_p'] = [0.2, 0.3, 0.2, 0.15, 0, 0.15]
            r['p_reg_rand'] = [0.55, 0.35, 0.1]
            r['placement_marker_min_max'] = cfg.INPUT.PLACEMENT_MARKER_MINMAX
            r['box_threshold'] = cfg.INPUT.FILTER_BOX_THRESHOLD
        else:
            r['reg_p'] = [0,0,0,0,1,0]
            r['p_reg_rand'] = [1,0,0]
            r['placement_marker_min_max'] = cfg.INPUT.MARKER_TEST_SIZE, cfg.INPUT.MARKER_TEST_SIZE+1
            r['box_threshold'] = cfg.INPUT.FILTER_BOX_THRESHOLD_TEST
        return r
    
    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        # Get the camera
        cam = dataset_dict.pop("camera")
        mtx, dist = np.array(cam["mtx"]), np.array(cam["dist"])

        board_annotations = dataset_dict.pop("annotations")
        
        # Add markers if not in annotations
        if "markers_world" not in board_annotations[0]:
            for ann in board_annotations:
                # ann["markers_world"], ann["marker_ids"] = image_placer(
                #     ann["board_world"][3][:2],
                #     num_classes=self.num_classes)
                markers_world, ann["marker_ids"] = marker_placer(
                    ann["board_world"][3][:2],
                    *self.placement_marker_min_max,
                    num_classes=self.num_classes, 
                    p_reg=self.reg_p, p_reg_rand=self.p_reg_rand)
                ann["markers_world"]  = markers_world[:,[0,1,3,2]]

        # Create marker annotations
        annotations = []
        for board_id, ann in enumerate(board_annotations):
            rvec, tvec = np.array(ann.pop("rvec")), np.array(ann.pop("tvec"))
            markers_screen = cv2.projectPoints(
                np.array(ann.pop("markers_world")).reshape(-1,3),
                rvec, tvec, mtx, dist)[0].reshape(-1,4,2) + 0.5 # Insert 0.5 offset from opencv
            for marker_screen, marker_id in zip(markers_screen, ann.pop("marker_ids")):
                annotations.append({
                    "iscrowd": 0,
                    "segmentation": [marker_screen.flatten().tolist()],
                    "category_id": marker_id,
                    "board_index": board_id})

        # if self.process_renderer_params:
        for ann in board_annotations:
            ann["brightness_max"] = (ann["brightness_max"] / 255.0) ** self.gamma
            ann["avg_color"] = (np.array(ann["avg_color"]) / 255.0) ** self.gamma

        # Apply augmentations here
        aug_input = T.StandardAugInput(image)
        transforms = aug_input.apply_augmentations(self.augmentations)
        
        # Calculate viewing directions
        # if self.process_renderer_params:
        inv_mtx = np.linalg.inv(mtx)
        image_shape = image.shape
        sx = np.linspace(0,image_shape[0]-1,image_shape[0])
        sy = np.linspace(0,image_shape[1]-1,image_shape[1])
        viewdir_screen = np.stack(np.meshgrid(sy, sx, 1), axis=2).reshape(image_shape[0], image_shape[1], 3)
        viewdir_world = np.matmul(viewdir_screen, inv_mtx.T)
        viewdir_world /= -np.expand_dims(np.linalg.norm(viewdir_world, axis=2), -1)
        dataset_dict["view_dir"] = torch.as_tensor(
            transforms.apply_image(viewdir_world).copy(), dtype=torch.float32)
        
        # Prepare image as tensor
        image = aug_input.image
        image_shape = image.shape[:2]  # h, w
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        # Add homography and more detailed segmentation points
        for ann in (annotations + board_annotations):
            calc_additional_annotation(ann, transforms, self.sampling_size)

        # Since evaluation uses recomputed gt, set height and width to image_shape
        dataset_dict['height'], dataset_dict["width"] = image_shape
        
        dataset_dict["instances"] = filter_instances(
            annotations_to_instances(image_shape, annotations), 
            self.box_threshold, self.max_instances)
        
        # Return None if no instances, this way mapper will try again.
        if len(dataset_dict["instances"]) == 0:
            return None
        dataset_dict["board_instances"] = annotations_to_instances(image_shape, board_annotations)
        
        return dataset_dict

class DetectronMapperWAnn(DetectronMapper):
    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        
        # Save gt_segm which might be used later
        # TODO: figure out when segmentation turn into segmentation mask
        gt_segm = torch.as_tensor([d['segmentation'][0] for d in dataset_dict['annotations']]).reshape(-1,4,2)

        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        # Will use the modified annotations after augmentations
        # if not self.is_train:
        #     # USER: Modify this if you want to keep them for some reason.
        #     dataset_dict.pop("annotations", None)
        #     dataset_dict.pop("sem_seg_file_name", None)
        #     return dataset_dict

        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

        # Since evaluation uses recomputed gt, set height and width to image_shape
        dataset_dict['height'], dataset_dict["width"] = image_shape

        return dataset_dict