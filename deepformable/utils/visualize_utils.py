"""
This code implements new visualizers and modifies detectron2's demo.py.
Copyright (c) Mustafa B. Yaldiz (VCLAB, KAIST) All Rights Reserved.
"""
from copy import deepcopy
import atexit
import bisect
import multiprocessing as mp
from typing import List
from collections import deque
import numpy as np
import random
import pycocotools.mask as mask_util

import cv2
import torch

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, GenericMask
from detectron2.structures import Instances
from detectron2.utils.colormap import _COLORS
from detectron2.utils.visualizer import Visualizer, _create_text_labels


def convert_mapped_instances(dataset_dict):
    instances = deepcopy(dataset_dict["instances"].to("cpu"))
    if instances.has("gt_segm") and instances.gt_segm.size(0) != 0:
        gt_segm_np = instances.gt_segm.view(instances.gt_segm.size(0), 1, -1).numpy()
        generic_mask = GenericMask(gt_segm_np.tolist(), *instances.image_size)
        instances.pred_masks = [generic_mask.polygons_to_mask([p]) for p in generic_mask.polygons]
        instances.pred_corners = instances.gt_segm[:,[0,2,4,6]]
    if instances.has("gt_masks"):
        instances.pred_masks = [
                mask_util.decode(mask_util.merge(
                    mask_util.frPyObjects(p, *instances.image_size)))[:, :] for p in instances.gt_masks.polygons]
        instances.pred_corners = torch.tensor(instances.gt_masks.polygons).view(-1,4,2)
    if instances.has("gt_sample_locs"):
        instances.pred_sample_locations = instances.gt_sample_locs

    instances.pred_boxes = instances.gt_boxes
    instances.pred_classes = instances.gt_classes
    instances.scores = torch.ones(len(instances.pred_classes))
    return instances


def random_colors(N, rgb=False, maximum=255):
    indices = []
    while len(indices) != N:
        indices.extend(
            random.sample(range(len(_COLORS)), min(N-len(indices), len(_COLORS))))
    ret = [_COLORS[i] * maximum for i in indices]
    if not rgb:
        ret = [x[::-1] for x in ret]
    return np.array(ret)

class DeepformableVisualizer:
    def __init__(self, metadata=None, num_classes=10):
        
        metadata = MetadataCatalog.get("__unused") if metadata is None else metadata
        self.marker_text = metadata.get("thing_classes", [f"{i}" for i in range(num_classes)])
        num_classes = len(self.marker_text)
        
        self.box_colors = random_colors(num_classes, maximum=1.0)
        self.segm_colors = random_colors(num_classes, maximum=1.0)

        self.box_rg_colors = np.array([
            1.0, 0.05, 0.05, 0.05, 1.0, 0.1]
        ).reshape(-1,3)

        self.segm_rg_colors = np.array([
            1.0, 0.1, 0.0, 0.0, 1.0, 0.05]
        ).reshape(-1,3)

        self.corner_colors = np.array([
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            0.0, 1.0, 0.0, 0.8, 0.3, 0.9
        ]).reshape(-1,3)

        self.sample_locations_color = np.array([0.1, 0.1, 0.1])
    
    def draw_instance_predictions(
        self,
        frame,
        predictions: Instances,
        draw_mask: bool = False,
        draw_bbox: bool = True,
        draw_corners: bool = True,
        draw_sample_locations: bool = True,
        use_closest_similarity_score: bool = True,
        draw_scale: float = 1.0,
        draw_circle_radius: float = 2.0,
        color_redgreen_threshold: float = 0.0,
    ):
        frame_visualizer = Visualizer(frame, None, scale=draw_scale)
        
        if len(predictions) == 0:
            return frame_visualizer.get_output()
        
        if use_closest_similarity_score and predictions.has("gt_closest_similarity_score"):
            predictions.scores = predictions.gt_closest_similarity_score

        boxes = predictions.pred_boxes.tensor.numpy() if predictions.has("pred_boxes") else None
        scores = predictions.scores.numpy() if predictions.has("scores") else np.ones(len(predictions))
        classes = predictions.pred_classes.numpy() if predictions.has("pred_classes") else np.zeros_like(scores, dtype=int)
        corners = predictions.pred_corners.numpy() if predictions.has("pred_corners") else None
        sample_locs = predictions.pred_sample_locations.view(-1,2).numpy() if predictions.has("pred_sample_locations") else None

        # If prediction classes are larger than the number of marker classes,
        # create a new color and text for each class.
        max_class = max(classes) + 1
        if max_class > len(self.box_colors):
            self.box_colors = np.concatenate([self.box_colors, random_colors(max_class - len(self.box_colors), maximum=1.0)])
            self.segm_colors = np.concatenate([self.segm_colors, random_colors(max_class - len(self.segm_colors), maximum=1.0)])
            self.marker_text = self.marker_text + [f"{i+len(self.marker_text)}" for i in range(max_class - len(self.marker_text))]

        labels = _create_text_labels(classes, scores, self.marker_text)

        if draw_bbox and boxes is not None:
            if color_redgreen_threshold == 0.0:
                assigned_colors = self.box_colors[classes]
            else:
                indexes = (scores > color_redgreen_threshold).astype(np.int)
                assigned_colors = self.segm_rg_colors[indexes]
            
            frame_visualizer.overlay_instances(
                boxes=boxes,
                labels=labels,
                assigned_colors=assigned_colors,
                alpha=0.5)

        if draw_mask and corners is not None:
            gt_segm_np = corners.reshape(corners.shape[0], 1, -1)
            generic_mask = GenericMask(gt_segm_np.tolist(), *predictions.image_size)
            pred_masks = [generic_mask.polygons_to_mask([p]) for p in generic_mask.polygons]
            
            frame_visualizer.overlay_instances(
                masks=pred_masks,
                labels=None if draw_bbox else labels,
                assigned_colors=self.segm_colors[classes],
                alpha=0.5)
            
        if draw_corners and corners is not None:
            for corners_i in corners:
                for idx, coord in enumerate(corners_i):
                    frame_visualizer.draw_circle(
                        coord, color=self.corner_colors[idx], 
                        radius=draw_circle_radius)

        if draw_sample_locations and sample_locs is not None:
            for coord in sample_locs:
                frame_visualizer.draw_circle(
                        coord, color=self.sample_locations_color, radius=0.4)

        return frame_visualizer.get_output()

class ModifiedPredictor(DefaultPredictor):
    def __init__(self, cfg, metadata=None):
        super().__init__(cfg)
        if metadata is not None:
            self.metadata = metadata
        
        if self.model.marker_generator and self.metadata and "messages" in self.metadata.as_dict():
            self.model.marker_generator.messages = self.metadata.messages
        

class VisualizationDemo(object):
    def __init__(self, cfg, parallel=False):
        self.cfg = cfg.clone()
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")

        parallel = parallel and not cfg.MODEL.DEVICE == "cpu"
        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = ModifiedPredictor(cfg)
        
        self.visualizer = DeepformableVisualizer(self.metadata)
    
    def process_predictions(self, frame, predictions):
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        vis_output = None
        if "instances" in predictions:
            vis_output = self.visualizer.draw_instance_predictions(
                frame, predictions["instances"].to(self.cpu_device),
                draw_mask=self.cfg.DEMO.DRAW_MASK, draw_bbox=self.cfg.DEMO.DRAW_BBOX,
                draw_corners=self.cfg.DEMO.DRAW_CORNERS, 
                color_redgreen_threshold=self.cfg.DEMO.COLOR_REDGREEN_THRESHOLD)
        return vis_output

    def run_on_image(self, image):
        predictions = self.predictor(image)
        return predictions, self.process_predictions(image, predictions)

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break
    
    def run_on_video(self, video):
        frame_gen = self._frame_from_video(video)
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()

            for cnt, frame in enumerate(frame_gen):
                frame_data.append(frame)
                self.predictor.put(frame)

                if cnt >= buffer_size:
                    frame = frame_data.popleft()
                    predictions = self.predictor.get()
                    vis_output = self.process_predictions(frame, predictions)
                    yield cv2.cvtColor(vis_output.get_image(), cv2.COLOR_RGB2BGR)

            while len(frame_data):
                frame = frame_data.popleft()
                predictions = self.predictor.get()
                vis_output = self.process_predictions(frame, predictions)
                yield cv2.cvtColor(vis_output.get_image(), cv2.COLOR_RGB2BGR)
        else:
            for frame in frame_gen:
                vis_output = self.process_predictions(frame, self.predictor(frame))
                yield cv2.cvtColor(vis_output.get_image(), cv2.COLOR_RGB2BGR)

class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput a little bit when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            self.metadata = MetadataCatalog.get(
                cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused")
            super().__init__()

        def run(self):
            predictor = ModifiedPredictor(self.cfg, self.metadata)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5