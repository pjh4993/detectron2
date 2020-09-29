# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from torch.autograd.function import Function

from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.structures import Boxes, Instances, pairwise_iou, ImageList
from detectron2.utils.events import get_event_storage

from ..box_regression import Box2BoxTransform
from ..matcher import Matcher
from ..poolers import ROIPooler
from .box_head import build_box_head
from .fast_rcnn import FastRCNNOutputLayers, fast_rcnn_inference, FastRCNNOutputs
from .roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads



__all__ = ["FCOSROIHeads", "FCOSRCNNOutputLayers"]

@ROI_HEADS_REGISTRY.register()
class FCOSROIHeads(StandardROIHeads):
    """
    ROIHeads for FCOSRPN
    in_features from Box / Class subnet of FCOSRPN
    No keypoint, segment head yet
    """

    @configurable
    def __init__(
        self,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_head : nn.Module,
        class_head : nn.Module,
        box_predictor: nn.Module,
        **kwargs,

    ):
        """
        NOTE: this interface is experimental.

        Args:
            box_pooler (ROIPooler): pooler that extracts region features from given boxes
            box_predictors (list[nn.Module]): box predictor 
        """
        super().__init__(
            box_in_features=box_in_features,
            box_pooler=box_pooler,
            box_head=box_head,
            box_predictor=box_predictor,
            **kwargs,
        )
        self.class_head = class_head
        self.proposal_append_gt = False

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        return ret

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features              = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution        = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales            = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio           = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type              = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        pooled_shape = ShapeSpec(
            channels=in_channels, width=pooler_resolution, height=pooler_resolution
        )
        box_head = build_box_head(cfg, pooled_shape)
        class_head = build_box_head(cfg, pooled_shape)

        box_predictor = FCOSRCNNOutputLayers(
            cfg, box_head.output_shape
        )

        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head" : box_head,
            "class_head" : class_head,
            "box_predictor": box_predictor,
        }

    def forward(
        self,
        images: ImageList,
        box_features: Dict[str, torch.Tensor],
        class_features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            pred_instances, losses = self._forward_box(box_features, class_features ,proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.

            return pred_instances, losses
        else:
            pred_instances = self._forward_box(box_features, class_features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            return pred_instances, {}

    def _forward_box(
        self, box_features: Dict[str, torch.Tensor], class_features: Dict[str, torch.Tensor], proposals: List[Instances], 
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        box_features = [box_features[f] for f in self.box_in_features]
        box_features = self.box_pooler(box_features, [x.proposal_boxes for x in proposals], [x.level for x in proposals])
        box_features = self.box_head(box_features)

        class_features = [class_features[f] for f in self.box_in_features]
        class_features = self.box_pooler(class_features, [x.proposal_boxes for x in proposals], [x.level for x in proposals])
        class_features = self.class_head(class_features)

        predictions = self.box_predictor(box_features, class_features)

        del box_features, class_features

        if self.training:
            losses = self.box_predictor.losses(predictions, proposals)
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances, losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances

class FCOSRCNNOutputLayers(FastRCNNOutputLayers):
    """
    Two linear layers for predicting FCOS R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """

    def forward(self, box_subnet, class_subnet):
        """
        Args:
            x: per-region features of shape (N, ...) for N bounding boxes to predict.

        Returns:
            (Tensor, Tensor):
            First tensor: shape (N,K+1), scores for each of the N box. Each row contains the
            scores for K object categories and 1 background class.

            Second tensor: bounding box regression deltas for each box. Shape is shape (N,Kx4),
            or (N,4) for class-agnostic regression.
        """
        if box_subnet.dim() > 2:
            box_subnet = torch.flatten(box_subnet, start_dim=1)

        if class_subnet.dim() > 2:
            class_subnet = torch.flatten(class_subnet, start_dim=1)

        scores = self.cls_score(class_subnet)
        proposal_deltas = self.bbox_pred(box_subnet)
        return scores, proposal_deltas