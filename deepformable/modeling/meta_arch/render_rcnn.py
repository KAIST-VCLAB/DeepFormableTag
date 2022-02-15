"""
This code is modified from detectron2 implementation, 
changes are logged in comments. 
Copyright (c) Mustafa B. Yaldiz (VCLAB, KAIST) All Rights Reserved.
"""
from __future__ import division
from typing import List, Dict, Optional
import numpy as np

import torch
from torch.nn import functional as F

from detectron2.utils.events import get_event_storage
from detectron2.utils.visualizer import Visualizer
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.modeling import GeneralizedRCNN, META_ARCH_REGISTRY
from detectron2.structures import Instances, Boxes
from detectron2.layers import batched_nms

from deepformable.modeling.marker_generator import build_marker_generator
from .utils import ImageList


@META_ARCH_REGISTRY.register()
class GeneralizedRCNN_RenderInput(GeneralizedRCNN):
    """
    GeneralizedRCNN modified for following functionalities:
    - Marker generation network is added, which might be used in postprocessing
    - Preprocessing inputs the rendered image.
    - Postprocessing is modified to process additional network outputs.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.test_score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        self.test_nms_thresh  = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        self.test_topk_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
        self.test_sort_instances = cfg.TEST.SORT_INSTANCES
        self.test_apply_nms = cfg.TEST.APPLY_NMS
        self.nms_score_criteria = str(cfg.TEST.NMS_SCORE_CRITERIA).lower()
        self.marker_postprocessing = cfg.TEST.MARKER_POSTPROCESSING
        self.add_module("marker_generator", build_marker_generator(cfg))

    def visualize_training(self, batched_inputs, proposals):
        """
        Modified to support training with rendering,
        Check detectron2 class for the original.
        """
        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["rendered_image"].detach()
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes.to("cpu"))
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch
    
    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Modified to support gradients for the image
        """
        if "rendered_image" in batched_inputs[0]:
            images = [(x["rendered_image"] - self.pixel_mean) / self.pixel_std for x in batched_inputs]
        else:
            images = [(x["image"].to(self.device) - self.pixel_mean) / self.pixel_std for x in batched_inputs]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def resize_create_instances(
        self, result: dict, output_height: int, output_width: int
    ):
        if isinstance(output_width, torch.Tensor):
            # This shape might (but not necessarily) be tensors during tracing.
            # Converts integer tensors to float temporaries to ensure true
            # division is performed when computing scale_x and scale_y.
            output_width_tmp = output_width.float()
            output_height_tmp = output_height.float()
            new_size = torch.stack([output_height, output_width])
        else:
            new_size = (output_height, output_width)
            output_width_tmp = output_width
            output_height_tmp = output_height
 
        scale_x, scale_y = (
            output_width_tmp / result["image_shape"][1],
            output_height_tmp / result["image_shape"][0],
        )

        pred_instances = Instances(new_size)
        corners = result["corners"]
        if corners.shape[0] == 0:
            pred_instances.pred_corners = corners
            return pred_instances
        decoded_messages = result.get("decoded_messages", None)
        obj_scores = result["obj_scores"]
        sample_locations = result.get("sample_locations", None)

        if decoded_messages is not None:
            scores = torch.sigmoid(obj_scores).squeeze()        
        else:
            scores = F.softmax(obj_scores, dim=-1)[:, :-1]
        
        if decoded_messages is not None:
            pred_instances.decoded_messages_confidence = decoded_messages
            pred_instances.decoded_messages = torch.round(decoded_messages)
            pred_instances.decoded_messages_avg_confidence = 1.0 - torch.mean(
                torch.abs(pred_instances.decoded_messages_confidence-decoded_messages), dim=1)
        else:
            # pred_instances.pred_classes = filter_inds[:, 1]
            # TODO: Fix this!!
            pass
        pred_instances.objectness_scores = scores

        # Scale corners and sample_locations
        scale_tensor = torch.tensor([scale_x, scale_y], device=corners.device)
        corners = corners * scale_tensor
        if sample_locations is not None:
            sample_locations = sample_locations * scale_tensor

        # Recalculate boxes
        min_c, max_c = torch.min(corners, dim=1)[0], torch.max(corners, dim=1)[0]
        boxes = torch.cat([min_c, max_c], dim=1)
        valid_mask = torch.isfinite(boxes).all(dim=1)
        
        # Add predictions to the instances, filter valid ones
        pred_instances.pred_boxes = Boxes(boxes)
        pred_instances.pred_corners = corners
        if sample_locations is not None:
            pred_instances.pred_sample_locations = sample_locations
        pred_instances = pred_instances[valid_mask]

        return pred_instances

    def apply_nms_instances(self, pred_instances):
        # Apply NMS from config settings provided
        if pred_instances.has("decoded_messages"):
            classes_temp = torch.zeros_like(pred_instances.scores, device=pred_instances.scores.device)
            keep = batched_nms(
                pred_instances.pred_boxes.tensor, pred_instances.scores, classes_temp, self.test_nms_thresh)
        else: # Case where classes are predicted from model directly
            keep = batched_nms(
                pred_instances.pred_boxes.tensor, pred_instances.scores, pred_instances.pred_classes, self.test_nms_thresh)
        return pred_instances[keep]
    
    def postprocess_single(self, result: dict, output_height: int, output_width: int):
        pred_instances = self.resize_create_instances(result, output_height, output_width)
        if self.marker_postprocessing:
            pred_instances = self.marker_generator.postprocessing(pred_instances)

        # Select the score wrt self.nms_score_criteria
        scores = pred_instances.objectness_scores
        if pred_instances.has("decoded_messages"):
            if "bit_similarity" in self.nms_score_criteria and pred_instances.has("bit_similarity"):
                scores = pred_instances.bit_similarity
            elif "message_confidence" in self.nms_score_criteria and pred_instances.has("decoded_messages_avg_confidence"):
                scores = pred_instances.decoded_messages_avg_confidence
            elif "mc_obj_product" in self.nms_score_criteria and pred_instances.has("decoded_messages_avg_confidence"):
                scores = scores * pred_instances.decoded_messages_avg_confidence
            elif "mc_obj_bs_product" in self.nms_score_criteria \
                and pred_instances.has("decoded_messages_avg_confidence")\
                and pred_instances.has("bit_similarity"):
                scores = scores * pred_instances.decoded_messages_avg_confidence * pred_instances.bit_similarity
        pred_instances.scores = scores

        if self.test_apply_nms:
            pred_instances = self.apply_nms_instances(pred_instances)

        # Filter by test score threshold
        pred_instances = pred_instances[pred_instances.scores > self.test_score_thresh]

        # Sort instances by score, FYI: NMS alreadys sorts them in decreasing order
        if (self.test_sort_instances or self.test_topk_per_image > 0) and (not self.test_apply_nms):
            sorted_idx = pred_instances.scores.argsort(descending=True)
            pred_instances = pred_instances[sorted_idx]

        # Index topk predictions only
        if self.test_topk_per_image > 0:
            pred_instances = pred_instances[:self.test_topk_per_image]

        return pred_instances
    
    def postprocess(self, instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes):
        # Rescale the output instances to the target size.
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = self.postprocess_single(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.
        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.
        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return self.postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results