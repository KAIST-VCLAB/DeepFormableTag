# Copyright (c) Mustafa B. Yaldiz (VCLAB, KAIST) All Rights Reserved.
import cv2
from cv2 import aruco
import random
import matplotlib.pyplot as plt
import json
import numpy as np
import argparse
import math
from pathlib import Path
from typing import Union
from copy import deepcopy
import datetime
import os
import random
from tqdm.notebook import tqdm

import torch
import torch.nn.functional as F

import detectron2
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.events import EventStorage
from detectron2.config import configurable

from deepformable.modeling import MarkerRendererDiffrast
from deepformable.modeling import MarkerRenderer

import deepformable
from deepformable.utils import DeepformableVisualizer
from deepformable.data import (
    register_deepformable_dataset, DeepformableMapper,
    DetectronMapperWAnn
)
from deepformable.utils import (
    get_cfg, convert_mapped_instances,
    DeepformableVisualizer,
    marker_metadata_loader
)
from deepformable.modeling import MarkerRendererDiffrast, IntermediateAugmentor
from deepformable.engine import DeepformableTrainer


def make_config(
    config_path="/host/configs/deepformable-main.yaml",
    weights="/Data/Models/deepformable_model.pth",
):
    # Setup Config
    cfg = get_cfg()
    cfg.OUTPUT_DIR = "/root"
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = weights
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 720, 736, 768, 800, 960)

    # cfg.INTERMEDIATE_AUGMENTOR.AUG_LIST = [
    #     "PerspectiveAugmentor", "TpsAugmentor", "RadialDistortionAugmentor",
    #     "DefocusBlurAugmentor", "MotionBlurAugmentor", "HueShiftAugmentor", 
    #     "BrightnessAugmentor", "NoiseAugmentor", "GammaAugmentor", 
    #     "GammaCorrector", "JPEGAugmentor"]
    # cfg.INTERMEDIATE_AUGMENTOR.EXEC_PROBA_LIST = [0.4, 0.5, 0.3, 0.4, 0.4, 0.4, 0.4, 0.45, 0.3, 1.0, 0.35]
    # cfg.INTERMEDIATE_AUGMENTOR.AUG_LIST = ["HueShiftAugmentor", "GammaCorrector"]
    # cfg.INTERMEDIATE_AUGMENTOR.EXEC_PROBA_LIST = [1.0] * len(cfg.INTERMEDIATE_AUGMENTOR.AUG_LIST)

    # cfg.RENDERER.SHADING_METHOD = "diffuse"
    return cfg


def register_datasets(cfg, data_root=Path("/Data/Datasets")):
    # Register datasets
    cur_data_root = data_root / "train-reduced"
    # cur_data_root = data_root / "test-inpainted"
    register_deepformable_dataset(
        "deepformable-rendered-train", {},
        str(cur_data_root / "annotations.json"),
        str(cur_data_root),
        load_markers=False)

    # Test1: rendered test
    cur_data_root = data_root / "test-inpainted"
    register_deepformable_dataset(
        "deepformable-rendered-test1", {},
        str(cur_data_root / "annotations.json"),
        str(cur_data_root),
        load_markers=False)
    register_deepformable_dataset(
        "deepformable-rendered-aug-test1", {},
        str(cur_data_root / "annotations.json"),
        str(cur_data_root),
        load_markers=False)

    # Test2: real-flat test
    cur_data_root = data_root / "test-realworld/flat"
    register_deepformable_dataset(
        "deepformable_flat-real-load_markers-test2", {},
        str(cur_data_root / "annotations.json"),
        str(cur_data_root),
        load_markers=True)
    register_deepformable_dataset(
        "deepformable_flat-real-load_markers-aug-test2", {},
        str(cur_data_root / "annotations.json"),
        str(cur_data_root),
        load_markers=True)

    # Test2: real-deformation test
    cur_data_root = data_root / "test-realworld/deformation"
    register_deepformable_dataset(
        "deepformable_deformation-real-load_markers-test3", {},
        str(cur_data_root / "annotations.json"),
        str(cur_data_root),
        load_markers=True)
    register_deepformable_dataset(
        "deepformable_deformation-real-load_markers-aug-test3", {},
        str(cur_data_root / "annotations.json"),
        str(cur_data_root),
        load_markers=True)

    # Load metadata
    if not marker_metadata_loader(cfg, data_root / "test-realworld/marker_config.json"):
        print("Failed to load marker metadata")

if __name__ == "__main__":
    enable_aug = True
    show_labels = True
    config_path = "configs/aruco-learnable-mit.yaml"
    
    cfg = make_config(config_path, weights="")
    register_datasets(cfg)
    dataset_names = cfg.DATASETS.TRAIN + cfg.DATASETS.TEST

    # Load datasets
    datasets, datasets_metadata, datasets_mapper = [], [], []
    for dataset_name in dataset_names:
        dataset = DatasetCatalog.get(dataset_name)
        is_train = "train" in dataset_name
        print(f"{dataset_name} length: {len(dataset)}")
        datasets.append(dataset)
        datasets_metadata.append(MetadataCatalog.get(dataset_name))
        mapper = DeepformableMapper(cfg, is_train) \
            if "rendered" in dataset_name else DetectronMapperWAnn(cfg, is_train)
        datasets_mapper.append(mapper)

    trainer = DeepformableTrainer(cfg, False)
    trainer.resume_or_load(resume=False)
    self = trainer.model
    show_with_renderers = {}
    # show_with_renderers = {cfg.RENDERER.NAME: self.renderer}
    # show_with_renderers = {
    #     "nvdiffrast": MarkerRendererDiffrast(cfg).to(self.device), 
    #     # "homography": MarkerRenderer(cfg).to(self.device),
    # }

    print("Input dataset index:")
    for index, dataset_name in enumerate(dataset_names):
        print(f" - {index}): {dataset_name}")
    
    d_idx = 0 # int(input())
    dataset, metadata = datasets[d_idx], datasets_metadata[d_idx]
    dataset_name, mapper = dataset_names[d_idx], datasets_mapper[d_idx]
    visualizer = DeepformableVisualizer(metadata)

    while(1):
        d = random.sample(dataset, 1)[0]
        dataset_dict = mapper(d)

        if dataset_dict is None:
            continue

        converted_dict = convert_mapped_instances(dataset_dict)
        img = dataset_dict['image'].permute(1,2,0).cpu().numpy()

        # %matplotlib inline 
        fig = plt.figure(figsize=(12,8))
        vis_out = visualizer.draw_instance_predictions(img, converted_dict)
        cv2.imshow("original", vis_out.get_image())
        
        with EventStorage(), torch.no_grad():
            data = self.carry_to_gpu([deepcopy(dataset_dict)])
            
            for d in data:
                d["image"] = (d["image"] / 255.0) ** self.gamma

            markers_batch, marker_loss = self.marker_generator(
                [d["instances"] for d in data])

            for renderer_name, renderer in show_with_renderers.items():
                d, markers = data[0], markers_batch[0]
                image = (renderer(d, markers) ** (1/self.gamma)) * 255.0
                image = image.permute(1,2,0).cpu().numpy()

                if not show_labels:
                    converted_dict = {}
                vis_out = visualizer.draw_instance_predictions(image, converted_dict)
                cv2.imshow(renderer_name, vis_out.get_image())

            for d, markers in zip(data, markers_batch):
                d["image"] = self.renderer(d, markers)

            # data, marker_loss = self.render_data(data)
            
            # Apply Augmentations
            for d in data:
                if enable_aug:
                    probabilities = torch.rand(self.aug_prob.shape, device=self.device)
                    indexes = (probabilities < self.aug_prob).nonzero(as_tuple=True)[0].tolist()
                    selected_augmentations = [self.intermediate_augmentations[i] for i in indexes]
                
                    for aug in selected_augmentations:
                        d["image"], d["instances"] = aug(d["image"], d["instances"])
                    d["instances"] = IntermediateAugmentor.fix_instances(d["instances"])
                    
                    print(selected_augmentations)
                else:
                    d["image"] = d["image"] ** (1/self.gamma)
                d["image"] = d["image"] * 255.0    
                
            image_np = data[0]["image"].permute(1,2,0).cpu().numpy()
            instances = data[0]["instances"]

            fig = plt.figure(figsize=(12,8))
            converted_dict = convert_mapped_instances(data[0])
            # converted_dict = {}
            if not show_labels:
                converted_dict = {}
            vis_out = visualizer.draw_instance_predictions(image_np, converted_dict)
            cv2.imshow("final_image", vis_out.get_image())

        k = cv2.waitKey(0)
        # Esc key to stop
        if k==27:    
            break
        else:
            continue
    