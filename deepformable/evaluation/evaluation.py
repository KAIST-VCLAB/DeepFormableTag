"""
This code is modified from detectron2 implementation, 
changes are logged in comments. 
Copyright (c) Mustafa B. Yaldiz (VCLAB, KAIST) All Rights Reserved.
"""
import os
import copy
import logging
import itertools
import datetime
from collections import OrderedDict

import numpy as np
import torch

import pycocotools.mask as mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.config import CfgNode
from detectron2.structures import Boxes, BoxMode, PolygonMasks, RotatedBoxes
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.visualizer import GenericMask
from detectron2.utils.file_io import PathManager

try:
    from detectron2.evaluation.fast_eval_api import COCOeval_opt
except ImportError:
    COCOeval_opt = COCOeval


def instances_to_coco_json(instances, img_id):
    """
    This method is modified from detectron2 implementation,
    we convert corners into prediction masks below.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()

    # Convert corners into prediction masks for evaluation
    if instances.has("pred_corners"):
        pred_segm_np = instances.pred_corners.view(instances.pred_corners.size(0), 1, -1).numpy()
        generic_mask = GenericMask(pred_segm_np.tolist(), *instances.image_size)
        instances.pred_masks = [generic_mask.polygons_to_mask([p]) for p in generic_mask.polygons]

    has_mask = instances.has("pred_masks")
    if has_mask:
        # use RLE to encode the masks, because they are too large and takes memory
        # since this evaluator stores outputs of the entire dataset
        rles = [
            mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            for mask in instances.pred_masks
        ]
        for rle in rles:
            # "counts" is an array encoded by mask_util as a byte-stream. Python3's
            # json writer which always produces strings cannot serialize a bytestream
            # unless you decode it. Thankfully, utf-8 works out (which is also what
            # the pycocotools/_mask.pyx does).
            rle["counts"] = rle["counts"].decode("utf-8")

    has_keypoints = instances.has("pred_keypoints")
    if has_keypoints:
        keypoints = instances.pred_keypoints

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
        }
        if has_mask:
            result["segmentation"] = rles[k]
        if has_keypoints:
            # In COCO annotations,
            # keypoints coordinates are pixel indices.
            # However our predictions are floating point coordinates.
            # Therefore we subtract 0.5 to be consistent with the annotation format.
            # This is the inverse of data loading logic in `datasets/coco.py`.
            keypoints[k][:, :2] -= 0.5
            result["keypoints"] = keypoints[k].flatten().tolist()
        results.append(result)
    return results


def inputs_to_coco_json(anns_per_image, img_id):# unmap the category mapping ids for COCO        
    coco_annotations = []
    for index in range(len(anns_per_image)):
        annotation = anns_per_image[index]
        # create a new dict with only COCO fields
        coco_annotation = {}

        if annotation.has("gt_segm"):
            gt_segm_np = annotation.gt_segm.view(annotation.gt_segm.size(0), 1, -1).numpy()
            annotation.gt_masks = PolygonMasks(gt_segm_np.tolist())

        # COCO requirement: XYWH box format for axis-align and XYWHA for rotated
        bbox = annotation.gt_boxes.tensor.numpy()[0]
        if isinstance(bbox, np.ndarray):
            if bbox.ndim != 1:
                raise ValueError(f"bbox has to be 1-dimensional. Got shape={bbox.shape}.")
            bbox = bbox.tolist()
        if len(bbox) not in [4, 5]:
            raise ValueError(f"bbox has to has length 4 or 5. Got {bbox}.")
        # from_bbox_mode = annotation["bbox_mode"]
        to_bbox_mode = BoxMode.XYWH_ABS if len(bbox) == 4 else BoxMode.XYWHA_ABS
        bbox = BoxMode.convert(bbox, BoxMode.XYXY_ABS, to_bbox_mode)
        
        if annotation.has("gt_masks"):
            area = annotation.gt_masks.area()[0].item()
            # polygons = np.array(annotation.gt_masks.polygons).reshape(-1,2)
            # min_val = polygons.min(axis=0)
            # dif = polygons.max(axis=0) - min_val
            # bbox = [*list(min_val), *list(dif)]
            coco_annotation["segmentation"] = [annotation.gt_masks.polygons[0][0]]
        else:
            bbox_xy = BoxMode.convert(bbox, to_bbox_mode, BoxMode.XYXY_ABS)
            area = Boxes([bbox_xy]).area()[0].item()
        
        # # COCO requirement: instance area
        # if "segmentation" in annotation:
        #     # Computing areas for instances by counting the pixels
        #     segmentation = annotation["segmentation"]
        #     # TODO: check segmentation type: RLE, BinaryMask or Polygon
        #     if isinstance(segmentation, list):
        #         polygons = PolygonMasks([segmentation])
        #         area = polygons.area()[0].item()
        #     elif isinstance(segmentation, dict):  # RLE
        #         area = mask_util.area(segmentation).item()
        #     else:
        #         raise TypeError(f"Unknown segmentation type {type(segmentation)}!")
        # else:
        #     # Computing areas using bounding boxes
        #     if to_bbox_mode == BoxMode.XYWH_ABS:
        #         bbox_xy = BoxMode.convert(bbox, to_bbox_mode, BoxMode.XYXY_ABS)
        #         area = Boxes([bbox_xy]).area()[0].item()
        #     else:
        #         area = RotatedBoxes([bbox]).area()[0].item()

        # if "keypoints" in annotation:
        #     keypoints = annotation["keypoints"]  # list[int]
        #     for idx, v in enumerate(keypoints):
        #         if idx % 3 != 2:
        #             # COCO's segmentation coordinates are floating points in [0, H or W],
        #             # but keypoint coordinates are integers in [0, H-1 or W-1]
        #             # For COCO format consistency we substract 0.5
        #             # https://github.com/facebookresearch/detectron2/pull/175#issuecomment-551202163
        #             keypoints[idx] = v - 0.5
        #     if "num_keypoints" in annotation:
        #         num_keypoints = annotation["num_keypoints"]
        #     else:
        #         num_keypoints = sum(kp > 0 for kp in keypoints[2::3])

        # COCO requirement:
        #   linking annotations to images
        #   "id" field must start with 1
        # coco_annotation["id"] = len(coco_annotations) + 1
        coco_annotation["image_id"] = img_id
        coco_annotation["bbox"] = bbox
        coco_annotation["area"] = float(area)
        coco_annotation["iscrowd"] = 0 # int(annotation.get("iscrowd", 0))
        coco_annotation["category_id"] = int(annotation.gt_classes.item())

        # # Add optional fields
        # if "keypoints" in annotation:
        #     coco_annotation["keypoints"] = keypoints
        #     coco_annotation["num_keypoints"] = num_keypoints

        # if "segmentation" in annotation:
        #     seg = coco_annotation["segmentation"] = annotation["segmentation"]
        #     if isinstance(seg, dict):  # RLE
        #         counts = seg["counts"]
        #         if not isinstance(counts, str):
        #             # make it json-serializable
        #             seg["counts"] = counts.decode("ascii")

        coco_annotations.append(coco_annotation)
    return coco_annotations

class DeepformableEvaluator(COCOEvaluator):
    def __init__(
        self,
        dataset_name,
        tasks=None,
        distributed=True,
        output_dir=None,
        *,
        max_dets_per_image=None,
        use_fast_impl=True,
        kpt_oks_sigmas=(),
    ):
        self._logger = logging.getLogger(__name__)
        self._distributed = distributed
        self._output_dir = output_dir

        if use_fast_impl and (COCOeval_opt is COCOeval):
            self._logger.info("Fast COCO eval is not built. Falling back to official COCO eval.")
            use_fast_impl = False
        self._use_fast_impl = use_fast_impl

        # COCOeval requires the limit on the number of detections per image (maxDets) to be a list
        # with at least 3 elements. The default maxDets in COCOeval is [1, 10, 100], in which the
        # 3rd element (100) is used as the limit on the number of detections per image when
        # evaluating AP. COCOEvaluator expects an integer for max_dets_per_image, so for COCOeval,
        # we reformat max_dets_per_image into [1, 10, max_dets_per_image], based on the defaults.
        if max_dets_per_image is None:
            max_dets_per_image = [1, 10, 100]
        else:
            max_dets_per_image = [1, 10, max_dets_per_image]
        self._max_dets_per_image = max_dets_per_image

        if tasks is not None and isinstance(tasks, CfgNode):
            kpt_oks_sigmas = (
                tasks.TEST.KEYPOINT_OKS_SIGMAS if not kpt_oks_sigmas else kpt_oks_sigmas
            )
            self._logger.warn(
                "COCO Evaluator instantiated using config, this is deprecated behavior."
                " Please pass in explicit arguments instead."
            )
            self._tasks = None  # Infering it from predictions should be better
        else:
            self._tasks = tasks

        self._cpu_device = torch.device("cpu")

        self._coco_api = None
        self._metadata = MetadataCatalog.get(dataset_name)
        # Instead of applying below, we will use inputs converted to COCO at the end
        # if not hasattr(self._metadata, "json_file"):
        #     if output_dir is None:
        #         raise ValueError(
        #             "output_dir must be provided to COCOEvaluator "
        #             "for datasets not in COCO format."
        #         )
        #     self._logger.info(f"Trying to convert '{dataset_name}' to COCO format ...")

        #     cache_path = os.path.join(output_dir, f"{dataset_name}_coco_format.json")
        #     self._metadata.json_file = cache_path
        #     convert_to_coco_json(dataset_name, cache_path, allow_cached=allow_cached_coco)

        # json_file = PathManager.get_local_path(self._metadata.json_file)
        # with contextlib.redirect_stdout(io.StringIO()):
        #     self._coco_api = COCO(json_file)

        # Test set json files do not contain annotations (evaluation must be
        # performed using the COCO evaluation server).
        # self._do_evaluation = "annotations" in self._coco_api.dataset
        # Below line is modified.
        self._do_evaluation = True
        if self._do_evaluation:
            self._kpt_oks_sigmas = kpt_oks_sigmas

    def reset(self):
        self._predictions = []
        self._inputs = []
        self._coco_api = None
    
    def process(self, inputs, outputs):
        """
        This method overrides the default process method of COCOEvaluator
        to call the custom instances_to_coco_json method.
        """
        for inp, outp in zip(inputs, outputs):
            image_id = inp["image_id"]
            prediction = {"image_id": image_id}
            input_dict = {"image_id": image_id}

            if "instances" in outp:
                instances = outp["instances"].to(self._cpu_device)
                # prediction["instances_detectron"] = instances
                prediction["instances"] = instances_to_coco_json(instances, image_id)
                # Move input instances to CPU too.
                instances = outp["gt_instances"].to(self._cpu_device)
                # input_dict["instances_detectron"] = instances
                input_dict["instances"] = inputs_to_coco_json(instances, image_id)
                # input_dict["image"] = outp["image"].to(self._cpu_device)
                for key in ["height", "width", "file_name"]:
                    input_dict[key] = inp[key]
    
            if "proposals" in outp:
                prediction["proposals"] = outp["proposals"].to(self._cpu_device)
            # Not the number of predictions, just checking if any instance is there
            if len(prediction) > 1: 
                self._predictions.append(prediction)
                # Append the inputs which might be modified by augmentations
                self._inputs.append(input_dict)

    def _tasks_from_predictions(self, predictions):
        return {"segm"}
    
    def evaluate(self, img_ids=None):
        """
        Args:
            img_ids: a list of image IDs to evaluate on. 
            Default to None for the whole dataset
        """
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            inputs = comm.gather(self._inputs, dst=0)
            predictions = list(itertools.chain(*predictions))
            inputs = list(itertools.chain(*inputs))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions
            inputs = self._inputs

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        # Bundle inputs into COCO format
        self._logger.info("Converting dataset dicts into COCO format")
        # First: Fix input category_id's into original COCO format
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()}
            reverse_id_mapper = lambda contiguous_id: reverse_id_mapping[contiguous_id]  # noqa
        else:
            reverse_id_mapper = lambda contiguous_id: contiguous_id  # noqa

        categories = [
            {"id": reverse_id_mapper(id), "name": name}
            for id, name in enumerate(self._metadata.thing_classes)
        ]
        # Second populate lists of annotations
        coco_images, coco_annotations = [], []
        for image_id, input_dict in enumerate(inputs):
            image_id = input_dict["image_id"]
            coco_image = {
                "id": image_id,
                "width": int(input_dict["width"]),
                "height": int(input_dict["height"]),
                "file_name": str(input_dict["file_name"]),
            }
            coco_images.append(coco_image)

            annotations = input_dict.get("instances", [])
            for ann in annotations:
                ann["id"] = len(coco_annotations) + 1
                ann["category_id"] = int(reverse_id_mapper(ann["category_id"]))
                coco_annotations.append(ann)
        info = {
            "date_created": str(datetime.datetime.now()),
            "description": "Automatically generated COCO json file for Detectron2.",
        }
        # Finally bundle them into dictionary
        coco_dict = {"info": info, "images": coco_images, "categories": categories, "licenses": None}
        if len(coco_annotations) > 0:
            coco_dict["annotations"] = coco_annotations
        self._logger.info(
            "Input conversion finished, "
            f"#images: {len(coco_images)}, #annotations: {len(coco_annotations)}"
        )

        # # Save predictions and gt
        # if self._output_dir:
        #     PathManager.mkdirs(self._output_dir)
        #     file_path = os.path.join(self._output_dir, "instances_predictions.pth")
        #     with PathManager.open(file_path, "wb") as f:
        #         torch.save(predictions, f)

        #     file_path = os.path.join(self._output_dir, "gt_instances.pth")
        #     with PathManager.open(file_path, "wb") as f:
        #         torch.save(inputs, f)

        # Create coco dataset with a dictionary generated
        coco_api = COCO()
        coco_api.dataset = coco_dict
        coco_api.createIndex()
        self._coco_api = coco_api
        
        self._results = OrderedDict()
        if "proposals" in predictions[0]:
            self._eval_box_proposals(predictions)
        if "instances" in predictions[0]:
            self._eval_predictions(predictions, img_ids=img_ids)
        # Copy so the caller can do whatever with results

        return copy.deepcopy(self._results)