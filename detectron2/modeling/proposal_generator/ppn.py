# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict, List, Optional, Tuple, Union
from detectron2.modeling.meta_arch.retinanet import permute_to_N_HWA_K
import torch
import torch.nn.functional as F
from fvcore.nn import giou_loss, smooth_l1_loss
from torch import nn

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, cat
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.utils.registry import Registry

from ..anchor_generator import build_anchor_generator
from ..box_regression import Box2BoxTransform
from ..matcher import Matcher
from ..sampling import subsample_labels
from .build import PROPOSAL_GENERATOR_REGISTRY
from .proposal_utils import find_top_ppn_proposals

PPN_HEAD_REGISTRY = Registry("PPN_HEAD")
PPN_HEAD_REGISTRY.__doc__ = """
Registry for RPN heads, which take feature maps and perform
objectness classification and bounding box regression for anchors.

The registered object will be called with `obj(cfg, input_shape)`.
The call should return a `nn.Module` object.
"""


"""
Shape shorthand in this module:

    N: number of images in the minibatch
    L: number of feature maps per image on which RPN is run
    A: number of cell anchors (must be the same for all feature maps)
    Hi, Wi: height and width of the i-th feature map
    B: size of the box parameterization

Naming convention:

    objectness: refers to the binary classification of an anchor as object vs. not object.

    deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransform`), or 5d for rotated boxes.

    pred_objectness_logits: predicted objectness scores in [-inf, +inf]; use
        sigmoid(pred_objectness_logits) to estimate P(object).

    gt_labels: ground-truth binary classification labels for objectness

    pred_anchor_deltas: predicted box2box transform deltas

    gt_anchor_deltas: ground-truth box2box transform deltas
"""


def build_ppn_head(cfg, input_shape):
    """
    Build an PPN head defined by `cfg.MODEL.PPN.HEAD_NAME`.
    """
    name = cfg.MODEL.PPN.HEAD_NAME
    return PPN_HEAD_REGISTRY.get(name)(cfg, input_shape)

def permute_to_N_HWA_K(tensor, K):
    """
    Transpose/reshape a tensor from (N, (Ai x K), H, W) to (N, (HxWxAi), K)
    """
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N,-1, K)  # Size=(NHWA,K)
    return tensor

def meshgrid_to_HW(tensor):
    N, _, H, W = tensor.shape
    grid_x, grid_y = torch.meshgrid(torch.arange(H), torch.arange(W))
    mesh = torch.cat((grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)), dim=-1)
    return mesh.reshape(-1, 2)

@PPN_HEAD_REGISTRY.register()
class StandardPPNHead(nn.Module):
    """
    Standard PPN classification and regression heads.
    Uses a 3x3 conv to produce a shared hidden state from which one 1x1 conv predicts
    objectness logits for each anchor and a second 1x1 conv predicts bounding-box deltas
    specifying how to deform each anchor into an object proposal.
    """

    @configurable
    def __init__(self, *, in_channels: int, num_anchors: int, box_dim: int = 4, id_dim: int = 1, prior: float = 0.01, num_classes: int = 80):
        """
        NOTE: this interface is experimental.

        Args:
            in_channels (int): number of input feature channels. When using multiple
                input features, they must have the same number of channels.
            num_anchors (int): number of anchors to predict for *each spatial position*
                on the feature map. The total number of anchors for each
                feature map will be `num_anchors * H * W`.
            box_dim (int): dimension of a box, which is also the number of box regression
                predictions to make for each anchor. An axis aligned box has
                box_dim=4, while a rotated box has box_dim=5.
        """
        super().__init__()

        # 3x3 conv for each hidden representation
        self.cls_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.box_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.id_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

        # 1x1 conv for predicting objectness logits
        self.logits_pred = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=1, stride=1)
        # 1x1 conv for predicting box2box transform deltas
        self.reg_pred = nn.Conv2d(in_channels, num_anchors * box_dim, kernel_size=1, stride=1)
        # 1x1 conv for predicting id vector
        self.id_vec_pred = nn.Conv2d(in_channels, num_anchors * id_dim, kernel_size=1, stride=1)

        self.output_shape = {'cls': in_channels, 'box': in_channels, 'pred_id_vec': id_dim, 'locale': 2}

        for l in [self.cls_conv, self.box_conv, self.id_conv,
                  self.logits_pred, self.reg_pred, self.id_vec_pred,]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)

        # @pjh3974 : Do prior things like Focal loss
 

    @classmethod
    def from_config(cls, cfg, input_shape):
        # Standard RPN is shared across levels:
        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        # RPNHead should take the same input as anchor generator
        # NOTE: it assumes that creating an anchor generator does not have unwanted side effect.
        anchor_generator = build_anchor_generator(cfg, input_shape)
        num_anchors = anchor_generator.num_anchors
        box_dim = anchor_generator.box_dim
        assert (
            len(set(num_anchors)) == 1
        ), "Each level must have the same number of anchors per spatial position"

        # @pjh3974 : Add PPN.ID_DIM, NUM_CLASSES to defaults.py
        ret = {
            "in_channels": in_channels, 
            "num_anchors": num_anchors[0], 
            "box_dim": box_dim,
            "id_dim": cfg.MODEL.PPN.ID_DIM,
            "num_classes": cfg.MODEL.PPN.NUM_CLASSES,
        }

        return ret

    def forward(self, features: List[torch.Tensor]):
        """
        Args:
            features (list[Tensor]): list of feature maps

        Returns:
            logits_pred (list[Tensor]): 
                the predicted class logits for all anchors. A is the number of cell anchors.
                Element i is a tensor of shape (N, A, Hi, Wi) representing
            list[Tensor]: A list of L elements. Element i is a tensor of shape
                (N, A*box_dim, Hi, Wi) representing the predicted "deltas" used to transform anchors
                to proposals.
        """
        logits_pred = []
        reg_pred = []
        id_vec_pred = []
        self.cls_subnet = []
        self.box_subnet = []

        for x in features:

            cls_subnet = F.relu(self.cls_conv(x))
            box_subnet = F.relu(self.box_conv(x))
            id_subnet = F.relu(self.id_conv(x))

            self.cls_subnet.append(cls_subnet)
            self.box_subnet.append(box_subnet)

            logits_pred.append(self.logits_pred(cls_subnet))
            reg_pred.append(self.reg_pred(box_subnet))
            id_vec_pred.append(self.id_vec_pred(id_subnet))

        return logits_pred, reg_pred, id_vec_pred

    def subnet_shape(self):
        return self.output_shape

@PROPOSAL_GENERATOR_REGISTRY.register()
class PPN(nn.Module):
    """
    Part Proposal Network.
    """

    @configurable
    def __init__(
        self,
        *,
        in_features: List[str],
        head: nn.Module,
        anchor_generator: nn.Module,
        anchor_matcher: Matcher,
        box2box_transform: Box2BoxTransform,
        batch_size_per_image: int,
        min_box_size: float = 0.0,
        anchor_boundary_thresh: float = -1.0,
        box_reg_loss_type: str = "giou",
        smooth_l1_beta: float = 0.0,
        num_classes = 80,
        id_dim = 1,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            in_features (list[str]): list of names of input features to use
            head (nn.Module): a module that predicts logits and regression deltas
                for each level from a list of per-level features
            anchor_generator (nn.Module): a module that creates anchors from a
                list of features. Usually an instance of :class:`AnchorGenerator`
            anchor_matcher (Matcher): label the anchors by matching them with ground truth.
            box2box_transform (Box2BoxTransform): defines the transform from anchors boxes to
                instance boxes
            batch_size_per_image (int): number of anchors per image to sample for training
            min_box_size (float): remove proposal boxes with any side smaller than this threshold,
                in the unit of input image pixels
            anchor_boundary_thresh (float): legacy option
            loss_weight (float|dict): weights to use for losses. Can be single float for weighting
                all rpn losses together, or a dict of individual weightings. Valid dict keys are:
                    "loss_rpn_cls" - applied to classification loss
                    "loss_rpn_loc" - applied to box regression loss
            box_reg_loss_type (str): Loss type to use. Supported losses: "smooth_l1", "giou".
            smooth_l1_beta (float): beta parameter for the smooth L1 regression loss. Default to
                use L1 loss. Only used when `box_reg_loss_type` is "smooth_l1"
        """
        super().__init__()
        self.in_features = in_features
        self.ppn_head = head
        self.anchor_generator = anchor_generator
        self.anchor_matcher = anchor_matcher
        self.box2box_transform = box2box_transform
        self.batch_size_per_image = batch_size_per_image
        # Map from self.training state to train/test settings
        self.min_box_size = float(min_box_size)
        self.anchor_boundary_thresh = anchor_boundary_thresh
        self.box_reg_loss_type = box_reg_loss_type
        self.smooth_l1_beta = smooth_l1_beta
        self.num_classes = num_classes
        self.id_dim = id_dim

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        in_features = cfg.MODEL.PPN.IN_FEATURES
        ret = {
            "in_features": in_features,
            "min_box_size": cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE,
            "batch_size_per_image": cfg.MODEL.PPN.BATCH_SIZE_PER_IMAGE,
            "anchor_boundary_thresh": cfg.MODEL.PPN.BOUNDARY_THRESH,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.PPN.BBOX_REG_WEIGHTS),
            "box_reg_loss_type": cfg.MODEL.PPN.BBOX_REG_LOSS_TYPE,
            "smooth_l1_beta": cfg.MODEL.PPN.SMOOTH_L1_BETA,
            "num_classes": cfg.MODEL.PPN.NUM_CLASSES,
            "id_dim": cfg.MODEL.PPN.ID_DIM,
        }

        ret["anchor_generator"] = build_anchor_generator(cfg, [input_shape[f] for f in in_features])
        ret["anchor_matcher"] = Matcher(
            cfg.MODEL.RPN.IOU_THRESHOLDS, cfg.MODEL.RPN.IOU_LABELS, allow_low_quality_matches=True
        )
        ret["head"] = build_ppn_head(cfg, [input_shape[f] for f in in_features])
        return ret

    def losses(
        self, 
        locations, 
        logits_pred, 
        reg_pred, 
        id_vec_pred, 
        gt_instances, 
        top_feats=None):

        return NotImplementedError

    def label_and_sample_anchors(
        self,
        anchors,
        gt_instances,
    ):

        return NotImplementedError

    def predict_proposals(
        self, 
        locations, 
        box_head_feature, 
        cls_head_feature, 
        id_vec_pred,
        image_sizes, 
    ):

        return NotImplementedError

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        gt_instances: Optional[List[Instances]] = None,
    ):
        """
        Args:
            images (ImageList): input images of length `N`
            features (dict[str, Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            gt_instances (list[Instances], optional): a length `N` list of `Instances`s.
                Each `Instances` stores ground-truth instances for the corresponding image.

        Returns:
            proposals: list[Instances]: contains fields "proposal_boxes", "objectness_logits"
            loss: dict[Tensor] or None
        """

        # @pjh3974 : pls put num_classes, id_dim to ppn
        features = [features[f] for f in self.in_features]
        anchors = self.anchor_generator(features)

        logits_pred, reg_pred, id_vec_pred = self.ppn_head(features)

        # Transpose the Hi*Wi*A dimension to the middle:
        logits_pred = [
            # (N, A*C, Hi, Wi) -> (N, A, C, Hi, Wi) -> (N, Hi, Wi, A, C) -> (N, Hi*Wi*A, C)
            permute_to_N_HWA_K(x, self.num_classes)
            for x in logits_pred
        ]
        reg_pred = [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            permute_to_N_HWA_K(x, self.anchor_generator.box_dim)
            for x in reg_pred
        ]
        id_vec_pred = [
            # (N, A, Hi, Wi) ->  (N, Hi, Wi, A) -> (N, Hi*Wi*A,1)
            permute_to_N_HWA_K(x, self.id_dim)
            for x in id_vec_pred
        ]

        locale_info = [
            meshgrid_to_HW(x)
            for x in self.ppn_head.box_subnet
        ]

        if self.training:
            assert gt_instances is not None, "PPN requires gt_instances in training!"

            losses = self.losses(anchors, logits_pred, reg_pred, id_vec_pred, gt_instances)

            part_proposals = self.predict_part_proposals(
                locale_info, self.ppn_head.box_subnet, self.ppn_head.cls_subnet, logits_pred, id_vec_pred, images.image_sizes
            )
        else:
            losses = {}

            part_proposals = self.predict_part_proposals(
                locale_info, self.ppn_head.box_subnet, self.ppn_head.cls_subnet, logits_pred, id_vec_pred, images.image_sizes
            )

        return part_proposals, losses

    def output_shape(self):
        return self.ppn_head.subnet_shape()