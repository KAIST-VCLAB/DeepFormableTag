# Copyright (c) Mustafa B. Yaldiz (VCLAB, KAIST) All Rights Reserved.
import numpy as np
from typing import List, Dict
import torch
from torch import nn

import detectron2
from detectron2.structures import ImageList, Instances, Boxes
from detectron2.modeling import META_ARCH_REGISTRY

from ..marker_generator import build_marker_generator


@META_ARCH_REGISTRY.register()
class ClassicalDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.add_module("marker_generator", build_marker_generator(cfg))
        self.test_topk_per_image = 0
        self.test_sort_instances = False
        self.test_apply_nms = False
        self.nms_score_criteria = "none"
        self.marker_postprocessing = False

    def inference(self, images):
        results = []
        for img in images:
            img = img.permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
            marker_corners, ids = self.marker_generator.recognize(img)
            result = {
                "corners": torch.tensor(marker_corners, dtype=torch.float32),
                "image_shape": img.shape[:2],
                "obj_scores": torch.as_tensor(np.ones(len(marker_corners)), dtype=torch.float32)}
            if len(marker_corners) != 0:
                result["pred_classes"] = torch.as_tensor(ids, dtype=torch.int64)
            results.append(result)
            
        return results

    def postprocess_single(self, result: dict, output_height: int, output_width: int):
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
      
        pred_instances.scores = result["obj_scores"]
        pred_instances.pred_classes = result["pred_classes"]

        # Scale corners and sample_locations
        scale_tensor = torch.tensor([scale_x, scale_y], device=corners.device)
        corners = corners * scale_tensor

        # Recalculate boxes
        min_c, max_c = torch.min(corners, dim=1)[0], torch.max(corners, dim=1)[0]
        boxes = torch.cat([min_c, max_c], dim=1)
        valid_mask = torch.isfinite(boxes).all(dim=1)
        
        # Add predictions to the instances, filter valid ones
        pred_instances.pred_boxes = Boxes(boxes)
        pred_instances.pred_corners = corners
        pred_instances = pred_instances[valid_mask]

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

    def forward(self, batched_inputs, do_postprocess=True):
        images = ImageList.from_tensors([x["image"] for x in batched_inputs], 1)
        results = self.inference(images)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return self.postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results