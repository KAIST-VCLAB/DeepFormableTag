# Copyright (c) Mustafa B. Yaldiz (VCLAB, KAIST) All Rights Reserved.
from .marker_roi_heads import MarkerROIHeads
from .naive_transform_head import NaiveTransformHead, ROI_TRANSFORM_HEAD_REGISTRY
from .corner_head import ROI_CORNER_HEAD_REGISTRY, CornerHead, CornerHeadV2
from .decoder_head import DecoderHead, ROI_DECODER_HEAD_REGISTRY
from .transformer_head import SpatialTransformerHead, SpatialTransformerHeadV2
