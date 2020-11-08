# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import inspect
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, nonzero_tuple
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

from ..backbone.resnet import BottleneckBlock, make_stage
from ..matcher import Matcher
from ..poolers import ROIPooler
from ..proposal_generator.proposal_utils import add_ground_truth_to_proposals
from ..sampling import subsample_labels

class PCOIHeads(torch.nn.Module):
    """
    PCOIHeads perform all per-cluster computation in an R-CNN.

    part-cluster given to PCOIHedas have already matched with ground truth.
    each element of part-cluster contains several features from Part proposal generator and some relative locale information btw features.

    1. Box-head features                (N x (k x k) x C_box)
    2. Cls-head features                (N x (k x k) x C_cls)
    3. locale information btw features  (N x 2)

    Typical logics to do in PCOIHeads is

    1. Feature fusion btw Box-head feature and locale info / Cls-head feature and locale info
    2. make per-cluster Box regression / classification with head

    It can have many variants, implemented as subclasses of this class.
    This base class contains the typical logic part.
    """

    @configurable
    def __init__(
        self,
        *,
        num_classes,
        box_feature_fusion_network,
        cls_feature_fusion_network,
        box_head,
        cls_head,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            num_classes (int): number of classes. Used to label background proposals.
        """
        super().__init__()
        self.num_classes = num_classes
        self.box_feature_fusion_network = box_feature_fusion_network
        self.cls_feature_fusion_network = cls_feature_fusion_network
        self.box_head = box_head

    @classmethod
    def from_config(cls, cfg):
        box_feature_fusion_network = build_locale_fusion_head(cfg.MODEL.PCOI_HEADS.LOCALE_FUSION_HEAD)
        cls_feature_fusion_network = build_locale_fusion_head(cfg.MODEL.PCOI_HEADS.LOCALE_FUSION_HEAD)
        box_head = build_box_head(cfg.MODEL.PCOI_HEADS.BOX_HEAD)
        return {
            "num_classes": cfg.MODEL.PCOI_HEADS.NUM_CLASSES,
            "box_feature_fusion_network": box_feature_fusion_network,
            "cls_feature_fusion_network": cls_feature_fusion_network,
            "box_head": box_head,
        }

    def forward(
        self,
        images: ImageList,
        part_clusters: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        Args:
            images (ImageList):

            part_clusters (list[Instances]): length `N` list of `Instances`. The i-th
                `Instances` contains part clusters for the i-th input image,
                with fields "box_features", "cls_features" and "rel_locale_info".

            targets (list[Instances], optional): length `N` list of `Instances`. The i-th
                `Instances` contains the ground-truth per-cluster annotations
                for the i-th input image.  Specify `targets` during training only.
                It may have the following fields:

                - gt_boxes: the bounding box of each instance.
                - gt_classes: the label for each instance with a category ranging in [0, #class].
                - gt_masks: PolygonMasks or BitMasks, the ground-truth masks of each instance.
                - gt_keypoints: NxKx3, the groud-truth keypoints for each instance.

        Returns:
            list[Instances]: length `N` list of `Instances` containing the
            detected instances. Returned during inference only; may be [] during training.

            dict[str->Tensor]:
            mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        raise NotImplementedError()

