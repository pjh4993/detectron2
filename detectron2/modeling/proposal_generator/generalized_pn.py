from collections import defaultdict
import logging
import math
import numpy as np
from typing import Dict, List, Tuple
from numpy.lib.arraysetops import isin
import torch
from fvcore.nn import giou_loss, sigmoid_focal_loss_jit, smooth_l1_loss
from torch import Tensor, nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import ShapeSpec, batched_nms, cat, get_norm, nonzero_tuple, permute_to_N_HWA_K, image_first_tensor, level_first_list
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou, levelwiseTensor
from detectron2.utils.events import get_event_storage

from ..anchor import build_anchor_generator, build_anchor_matcher
from ..box_coder import build_box_coder
from .build import PROPOSAL_GENERATOR_REGISTRY, build_proposal_generator_head

__all__ = ["GeneralizedPN"]

@PROPOSAL_GENERATOR_REGISTRY.register()
class GeneralizedPN(nn.Module):
    """
    Generalized Proposal network. Use configuration to detailed implementation.
    Designed to produce Classification and Regression result only.
    Do additional thing at Instance generator.
    """

    @configurable
    def __init__(
        self,
        *,
        # Head parameters
        head,
        head_in_features,
        num_classes,
        # Anchor parameters
        anchor_generator,
        box_coder,
        anchor_matcher,
        # Loss parameters
        focal_loss_alpha=0.25,
        focal_loss_gamma=2.0,
        smooth_l1_beta=0.1,
        box_reg_loss_type="smooth_l1",
        # Input parameters
        pixel_mean,
        pixel_std,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            # Head parameters:
            head (nn.Module): a module that predicts logits and regression deltas
                for each level from a list of per-level features
            head_in_features (Tuple[str]): Names of the input feature maps to be used in head
            num_classes (int): number of classes. Used to label background proposals.

            # Anchor parameters:
            anchor_generator (nn.Module): a module that creates anchors from a
                list of features. Usually an instance of :class:`AnchorGenerator`
            box_coder: defines the transform from anchors boxes to
                instance boxes
            anchor_matcher (Matcher): label the anchors by matching them with ground truth.

            # Loss parameters:
            focal_loss_alpha (float): focal_loss_alpha
            focal_loss_gamma (float): focal_loss_gamma
            smooth_l1_beta (float): smooth_l1_beta
            box_reg_loss_type (str): Options are "smooth_l1", "giou"

            # Input parameters
            pixel_mean (Tuple[float]):
                Values to be used for image normalization (BGR order).
                To train on images of different number of channels, set different mean & std.
                Default values are the mean pixel value from ImageNet: [103.53, 116.28, 123.675]
            pixel_std (Tuple[float]):
                When using pre-trained models in Detectron1 or any MSRA models,
                std has been absorbed into its conv1 weights, so the std needs to be set 1.
                Otherwise, you can use [57.375, 57.120, 58.395] (ImageNet std)
        """
        super().__init__()

        # Head
        self.head = head
        self.head_in_features = head_in_features
        self.num_classes = num_classes

        # Anchors
        self.anchor_generator = anchor_generator
        self.box_coder = box_coder
        self.anchor_matcher = anchor_matcher

        # Loss parameters:
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.smooth_l1_beta = smooth_l1_beta
        self.box_reg_loss_type = box_reg_loss_type

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        feature_shapes = [input_shape[f] for f in cfg.MODEL.PROPOSAL_GENERATOR.IN_FEATURES]
        head = GeneralizedPNHead(cfg, feature_shapes)
        anchor_generator = build_anchor_generator(cfg, feature_shapes)
        box_coder = build_box_coder(cfg)
        anchor_matcher = build_anchor_matcher(cfg)
        return {
            # Head parameters
            "head": head,
            "num_classes": cfg.MODEL.PROPOSAL_GENERATOR.NUM_CLASSES,
            "head_in_features": cfg.MODEL.PROPOSAL_GENERATOR.IN_FEATURES,
            # Anchor paramters
            "anchor_generator": anchor_generator,
            "box_coder": box_coder,
            "anchor_matcher": anchor_matcher,
            # Loss parameters:
            "focal_loss_alpha": cfg.MODEL.PROPOSAL_GENERATOR.FOCAL_LOSS_ALPHA,
            "focal_loss_gamma": cfg.MODEL.PROPOSAL_GENERATOR.FOCAL_LOSS_GAMMA,
            "smooth_l1_beta": cfg.MODEL.PROPOSAL_GENERATOR.SMOOTH_L1_LOSS_BETA,
            "box_reg_loss_type": cfg.MODEL.PROPOSAL_GENERATOR.BBOX_REG_LOSS_TYPE,
            # Training parameters
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, features, gt_instances):
        """
        Args:
            images : 
            features : 
            gt_instances : 
        Returns:
            in training, dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
            in inference, the standard output format, described in :doc:`/tutorials/models`.
        """
        features = [features[f] for f in self.head_in_features]
        anchors = self.anchor_generator(features)
        preds = self.head(features)
        pred_logits = levelwiseTensor(preds['CLS'], self.num_classes)
        pred_regs = levelwiseTensor(preds['BBOX'], K=4)
        pred_boxes = self.box_coder.decode(anchors, pred_regs)

        # Transpose the Hi*Wi*A dimension to the middle:
        losses = {}

        if self.training:
            assert not torch.jit.is_scripting(), "Not supported"
            gt_labels, gt_boxes, cls_pos_mask, reg_pos_mask = self.sample_positive(anchors, gt_instances, pred_logits, pred_boxes)
            losses = self.losses(anchors, pred_logits, gt_labels, pred_boxes, gt_boxes, cls_pos_mask, reg_pos_mask)

        proposal_results = self.inference(pred_logits, pred_boxes)

        return proposal_results, losses

    def losses(self, anchors, pred_logits, gt_labels, pred_boxes, gt_boxes, cls_pos_mask, reg_pos_mask):
        """
        Args:
            anchors (list[Boxes]): a list of #feature level Boxes
            gt_labels, gt_boxes: see output of :meth:`RetinaNet.sample_positive`.
                Their shapes are (N, R) and (N, R, 4), respectively, where R is
                the total number of anchors across levels, i.e. sum(Hi x Wi x Ai)
            pred_logits, pred_regs: both are list[Tensor]. Each element in the
                list corresponds to one level and has shape (N, Hi * Wi * Ai, K or 4).
                Where K is the number of classes used in `pred_logits`.

        Returns:
            dict[str, Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_cls" and "loss_box_reg"
        """
        num_images = gt_labels[0].shape[0]
        gt_labels = gt_labels.image_first_tensor()
        cls_pos_mask = cls_pos_mask.image_first_tensor()
        reg_pos_mask = reg_pos_mask.image_first_tensor()
        pred_logits = pred_logits.image_first_tensor()
        pred_boxes = pred_boxes.image_first_tensor()
        gt_boxes = gt_boxes.image_first_tensor()

        # need reduce sum for training in multiple GPU
        num_cls_pos_anchors = cls_pos_mask.sum().item()
        num_reg_pos_anchors = reg_pos_mask.sum().item()

        get_event_storage().put_scalar("num_cls_pos_anchors", num_cls_pos_anchors / num_images)
        get_event_storage().put_scalar("num_reg_pos_anchors", num_reg_pos_anchors / num_images)

        # classification and regression loss
        gt_labels_target = F.one_hot(gt_labels[cls_pos_mask], num_classes=self.num_classes)

        loss_cls = sigmoid_focal_loss_jit(
            pred_logits[cls_pos_mask],
            gt_labels_target.to(pred_logits.dtype),
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        )

        if self.box_reg_loss_type == "smooth_l1":
            loss_box_reg = smooth_l1_loss(
                pred_boxes[reg_pos_mask],
                gt_boxes[reg_pos_mask],
                beta=self.smooth_l1_beta,
                reduction="sum",
            )
        elif self.box_reg_loss_type == "giou":
            loss_box_reg = giou_loss(
                pred_boxes[reg_pos_mask,:], gt_boxes[reg_pos_mask,:], reduction="sum"
            )
        else:
            raise ValueError(f"Invalid bbox reg loss type '{self.box_reg_loss_type}'")

        return {
            "loss_cls": loss_cls / num_cls_pos_anchors,
            "loss_box_reg": loss_box_reg / num_reg_pos_anchors,
        }

    @torch.no_grad()
    def sample_positive(self, anchors, gt_instances, pred_logits=None, pred_boxes=None):
        """
        Args:
            anchors (list[Boxes]): A list of #feature level Boxes.
                The Boxes contains anchors of this image on the specific feature level.
            gt_instances (list[Instances]): a list of N `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.

        Returns:
            list[Tensor]:
                List of #img tensors. i-th element is a vector of labels whose length is
                the total number of anchors across all feature maps (sum(Hi * Wi * A)).
                Label values are in {-1, 0, ..., K}, with -1 means ignore, and K means background.
            list[Tensor]:
                i-th element is a Rx4 tensor, where R is the total number of anchors across
                feature maps. The values are the matched gt boxes for each anchor.
                Values are undefined for those anchors not labeled as foreground.
        """
        gt_labels = []
        matched_gt_boxes = []
        cls_pos_masks = []
        reg_pos_masks = []

        gt_labels_per_image = []
        matched_gt_boxes_per_image = []
        cls_pos_masks_per_image = []
        reg_pos_masks_per_image = []
        for gt_per_image in gt_instances:
            matched_idxs, cls_mask_per_image, reg_mask_per_image = self.anchor_matcher(gt_per_image.gt_boxes, anchors, pred_logits, pred_boxes)

            if len(gt_per_image) > 0:
                matched_gt_boxes_i = gt_per_image.gt_boxes.tensor[matched_idxs]
                gt_labels_i = gt_per_image.gt_classes[matched_idxs]
            else:
                matched_gt_boxes_i = torch.zeros_like(anchors.tensor)
                gt_labels_i = torch.zeros_like(matched_idxs)

            gt_labels_per_image.append(gt_labels_i)
            matched_gt_boxes_per_image.append(matched_gt_boxes_i)
            cls_pos_masks_per_image.append(cls_mask_per_image)
            reg_pos_masks_per_image.append(reg_mask_per_image)
        
        len_lvl = len(gt_labels[0])

        for lvl in range(len_lvl):
            gt_labels_per_level = []
            matched_gt_boxes_per_level = []
            cls_pos_per_level = []
            reg_pos_per_level = []
            for i in range(len(gt_instances)):
                gt_labels_per_level.append(gt_labels_per_image[i][lvl])
                matched_gt_boxes_per_level.append(matched_gt_boxes_per_image[i][lvl])
                cls_pos_per_level.append(cls_pos_masks_per_image[i][lvl])
                reg_pos_per_level.append(reg_pos_masks_per_image[i][lvl])

            gt_labels.append(cat(gt_labels_per_level,dim=0))
            matched_gt_boxes.append(cat(matched_gt_boxes_per_level, dim=0))
            cls_pos_masks.append(cat(cls_pos_per_level, dim=0))
            reg_pos_masks.append(cat(reg_pos_per_level, dim=0))

        gt_labels = levelwiseTensor(gt_labels, K=1)
        matched_gt_boxes = levelwiseTensor(matched_gt_boxes, K=4)
        cls_pos_masks = levelwiseTensor(cls_pos_masks, K=1)
        reg_pos_masks = levelwiseTensor(reg_pos_masks, K=1)

        return gt_labels, matched_gt_boxes, cls_pos_masks, reg_pos_masks
    
    def inference(
        self,
        pred_logits: Tensor,
        pred_boxes: Tensor,
        cls_pos_mask: Tensor,
        reg_pos_mask: Tensor,
    ):
        """
        Arguments:
            pred_logits, pred_anchor_deltas: Tensor, one per level. Each
                has shape (N, Hi * Wi * Ai, K or 4)
        Returns:
            results (List[Instances]): a list of #images elements.
        """

        return {"CLS": pred_logits, "BBOX": pred_boxes, "CLS_POS": cls_pos_mask, "REG_POS": reg_pos_mask}

class GeneralizedPNHead(nn.Module):
    """
    The head used in GeneralizedPN for object classification and box regression.
    It has two subnets for the two tasks, with a common structure but separate parameters.
    """

    @configurable
    def __init__(
        self,
        *,
        input_shape: List[ShapeSpec],
        num_classes,
        num_anchors,
        head_config,
        norm="",
        prior_prob=0.01,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (List[ShapeSpec]): input shape
            num_classes (int): number of classes. Used to label background proposals.
            num_anchors (int): number of generated anchors
            head_config (Dict): configuration about head structure
            norm (str or callable):
                    Normalization for conv layers except for the two output layers.
                    See :func:`detectron2.layers.get_norm` for supported types.
            prior_prob (float): Prior weight for computing bias
        """
        super().__init__()

        if norm == "BN" or norm == "SyncBN":
            logger = logging.getLogger(__name__)
            logger.warn("Shared norm does not work well for BN, SyncBN, expect poor results")

        subnets = {}
        heads = {}
        for k, configs in head_config:
            subnets[k] = [] 
            in_channels = input_shape[0].channels
            out_channels = configs.INTER_DIM

            if k == 'CLS':
                out_dim = num_anchors * num_classes
            elif k == 'BBOX':
                out_dim = num_anchors * 4
            else:
                out_dim = 1

            num_convs = configs.NUM_CONVS
            for i in range(num_convs):
                subnets[k].append(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
                )
                in_channels = out_channels
                if norm:
                    subnets[k].append(get_norm(norm, out_channels))
                subnets[k].append(nn.ReLU())

            subnets[k] = nn.Sequential(*subnets[k])

            heads[k] = nn.Conv2d(
                out_channels, out_dim, kernel_size=3, stride=1, padding=1
            )
        
        for k, module in subnets.items() + heads.items():
            for layer in module.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)
        

        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(heads['CLS'].bias, bias_value)

        self.subnets = subnets
        self.heads = heads

    @classmethod
    def from_config(cls, cfg, input_shape: List[ShapeSpec]):
        num_anchors = cfg.MODEL.ANCHOR.NUM_ANCHORS

        return {
            "input_shape": input_shape,
            "num_classes": cfg.MODEL.PROPOSAL_GENERATOR.NUM_CLASSES,
            "num_anchors": num_anchors,
            "subnet_config": cfg.MODEL.PROPOSAL_GEERATOR.SUBNET_CONFIG,
            "head_config": cfg.MODEL.PROPOSAL_GEERATOR.HEAD_CONFIG,
            "prior_prob": cfg.MODEL.PROPOSAL_GENERATOR.PRIOR_PROB,
            "norm": cfg.MODEL.PROPOSAL_GENERATOR.NORM,
        }

    def forward(self, features: List[Tensor]):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            outputs (Dict[List[Tensor]]): #lvl tensors, each list contains outputs of heads such as cls, bbox and etc.
                Tensor per each list has shape (N, AxK, Hi, Wi).
        """
        outputs = defaultdict(list)
        for feature in features:
            tower = {}
            for k in list(self.subnets.keys()):
                tower = self.subnets[k](feature)
                outputs[k].append(self.heads[k](tower))

        return outputs