"""
This code is modified from detectron2 implementation, 
changes are logged in comments. 
Copyright (c) Mustafa B. Yaldiz (VCLAB, KAIST) All Rights Reserved.
"""
import numpy as np
import pycocotools.mask as mask_util

from detectron2.structures import BoxMode
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.visualizer import GenericMask


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

class DeepformableEvaluator(COCOEvaluator):
    def process(self, inputs, outputs):
        """
        This method overrides the default process method of COCOEvaluator
        to call the custom instances_to_coco_json method.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_coco_json(instances, input["image_id"])
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)
            if len(prediction) > 1:
                self._predictions.append(prediction)