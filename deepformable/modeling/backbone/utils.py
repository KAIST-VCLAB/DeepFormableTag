
"""
This code modifies FPN implementation from detectron2 to output stem features.
Copyright (c) Mustafa B. Yaldiz (VCLAB, KAIST) All Rights Reserved.
"""
import torch
import torch.nn.functional as F

from detectron2.modeling.backbone.fpn import FPN as FPN_detectron2

class FPN(FPN_detectron2):
    def forward(self, x):
        # Reverse feature maps into top-down order (from low to high resolution)
        bottom_up_features = self.bottom_up(x)
        x = [bottom_up_features[f] for f in self.in_features[::-1]]
        results = []
        prev_features = self.lateral_convs[0](x[0])
        results.append(self.output_convs[0](prev_features))
        for features, lateral_conv, output_conv in zip(
            x[1:], self.lateral_convs[1:], self.output_convs[1:]
        ):
            top_down_features = F.interpolate(prev_features, scale_factor=2, mode="nearest")
            lateral_features = lateral_conv(features)
            prev_features = lateral_features + top_down_features
            if self._fuse_type == "avg":
                prev_features /= 2
            results.insert(0, output_conv(prev_features))

        if self.top_block is not None:
            top_block_in_feature = bottom_up_features.get(self.top_block.in_feature, None)
            if top_block_in_feature is None:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        out = dict(zip(self._out_features, results))
        
        # -- MODIFICATION: Make sure out includes stem features added in output of backbone --
        for key in bottom_up_features.keys():
            if key not in self.in_features and key in self.bottom_up.output_shape():
                out[key] = bottom_up_features[key]
        
        return out
