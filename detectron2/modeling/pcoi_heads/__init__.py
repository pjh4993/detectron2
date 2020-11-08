# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#from .box_head import PCOI_BOX_HEAD_REGISTRY, build_pcoi_box_head

"""
from .keypoint_head import ROI_KEYPOINT_HEAD_REGISTRY, build_keypoint_head, BaseKeypointRCNNHead
from .mask_head import ROI_MASK_HEAD_REGISTRY, build_mask_head, BaseMaskRCNNHead
"""

from .pcoi_heads import (
    PCOI_HEADS_REGISTRY,
    PCOIHeads,
    build_pcoi_heads,
)
"""
Res5ROIHeads,
StandardROIHeads,
build_roi_heads,
select_foreground_proposals,
"""


__all__ = list(globals().keys())
