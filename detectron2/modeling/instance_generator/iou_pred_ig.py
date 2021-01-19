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
from detectron2.layers import ShapeSpec, batched_nms, cat, get_norm, nonzero_tuple, permute_to_N_HWA_K
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from torch.nn.modules.loss import BCEWithLogitsLoss

from .build import INSTANCE_GENERATOR_REGISTRY, build_instance_generator

@INSTANCE_GENERATOR_REGISTRY.register()
class IoUPredIG(nn.Module):
    """
    Instance generator with iou prediction.
    Predict per feature iou prediction with pred Box and target gt box.
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
        # Inference parameters
        test_score_thresh=0.05,
        test_topk_candidates=1000,
        test_nms_thresh=0.5,
        max_detections_per_image=100,
        # Input parameters
        pixel_mean,
        pixel_std,
    ):
        super.__init__()

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

        # Inference parameters:
        self.test_score_thresh = test_score_thresh
        self.test_topk_candidates = test_topk_candidates
        self.test_nms_thresh = test_nms_thresh
        self.max_detections_per_image = max_detections_per_image

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))


    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        feature_shapes = [input_shape[f] for f in cfg.MODEL.PROPOSAL_GENERATOR.IN_FEATURES]
        head = IoUPredIGHead(cfg, feature_shapes)
        return {
            # Head parameters
            "head": head,
            "num_classes": cfg.MODEL.PROPOSAL_GENERATOR.NUM_CLASSES,
            "head_in_features": cfg.MODEL.PROPOSAL_GENERATOR.IN_FEATURES,
            # Loss parameters:
            "focal_loss_alpha": cfg.MODEL.PROPOSAL_GENERATOR.FOCAL_LOSS_ALPHA,
            "focal_loss_gamma": cfg.MODEL.PROPOSAL_GENERATOR.FOCAL_LOSS_GAMMA,
            "smooth_l1_beta": cfg.MODEL.PROPOSAL_GENERATOR.SMOOTH_L1_LOSS_BETA,
            "box_reg_loss_type": cfg.MODEL.PROPOSAL_GENERATOR.BBOX_REG_LOSS_TYPE,
            # Inference parameters:
            "test_score_thresh": cfg.MODEL.PROPOSAL_GENERATOR.SCORE_THRESH_TEST,
            "test_topk_candidates": cfg.MODEL.PROPOSAL_GENERATOR.TOPK_CANDIDATES_TEST,
            "test_nms_thresh": cfg.MODEL.PROPOSAL_GENERATOR.NMS_THRESH_TEST,
            "max_detections_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            # Training parameters
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, images, features, proposal_results, gt_instances):
        """
        Args:
            images : 
            features : 
            proposal_results :
            gt_instances : 
        Returns:
            in training, dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
            in inference, the standard output format, described in :doc:`/tutorials/models`.
        """
        features = [features[f] for f in self.head_in_features]
        preds = self.head(features)
        pred_ious = preds['IOU']

        # Transpose the Hi*Wi*A dimension to the middle:
        pred_ious = cat([permute_to_N_HWA_K(x, self.num_classes) for x in pred_ious], dim=1)

        losses = {}

        if self.training:
            assert not torch.jit.is_scripting(), "Not supported"
            gt_ious, iou_pos_mask = self.sample_positive(proposal_results, gt_instances)
            losses = self.losses(pred_ious, gt_ious, iou_pos_mask)

        instance_results = self.inference(proposal_results, pred_ious, images.image_sizes)

        return instance_results, losses

    def sample_positive(self, proposal_results, gt_instances):
        reg_pos_mask = proposal_results['REG_POS']
        pred_boxes = proposal_results['BBOX']

        gt_ious = []
 
        for gt_per_image, reg_pos_per_image, pred_boxes_per_image in zip(gt_instances, reg_pos_mask, pred_boxes):
            gt_iou_per_image = pairwise_iou(gt_per_image.gt_boxes.tensor[reg_pos_per_image], pred_boxes_per_image)
            gt_ious.append(gt_iou_per_image)

        return gt_ious, reg_pos_mask

    def inference(self, proposal_results, pred_ious, image_sizes):
        """
        Arguments:
            pred_logits, pred_anchor_deltas: list[Tensor], one per level. Each
                has shape (N, Hi * Wi * Ai, K or 4)
            image_sizes (List[(h, w)]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """

        results: List[Instances] = []
        pred_logits, pred_boxes = proposal_results['CLS'], proposal_results['BBOX']

        for img_idx, image_size in enumerate(image_sizes):
            pred_logits_per_image = [x[img_idx] for x in pred_logits]
            pred_boxes_per_image = [x[img_idx] for x in pred_boxes]
            pred_ious_per_image = [x[img_idx] for x in pred_ious]
            results_per_image = self.inference_single_image(
                pred_logits_per_image, pred_boxes_per_image, pred_ious_per_image, image_size
            )
            results.append(results_per_image)
        return results

    def inference_single_image(
        self,
        pred_logits: List[Tensor],
        pred_boxes: List[Tensor],
        pred_ious: List[Tensor],
        image_size: Tuple[int, int],
    ):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Arguments:
            anchors (list[Boxes]): list of #feature levels. Each entry contains
                a Boxes object, which contains all the anchors in that feature level.
            box_cls (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (H x W x A, K)
            box_delta (list[Tensor]): Same shape as 'box_cls' except that K becomes 4.
            image_size (tuple(H, W)): a tuple of the image height and width.

        Returns:
            Same as `inference`, but for only one image.
        """
        boxes_all = []
        scores_all = []
        class_idxs_all = []

        # Iterate over every feature level
        for box_cls_i, box_reg_i, box_iou_i in zip(pred_logits, pred_boxes, pred_ious):
            # (HxWxAxK,)
            predicted_prob = box_cls_i.flatten().sigmoid_()

            # Apply two filtering below to make NMS faster.
            # 1. Keep boxes with confidence score higher than threshold
            keep_idxs = predicted_prob > self.test_score_thresh
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = nonzero_tuple(keep_idxs)[0]

            # 2. Keep top k top scoring boxes only
            num_topk = min(self.test_topk_candidates, topk_idxs.size(0))
            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, idxs = predicted_prob.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[idxs[:num_topk]]

            anchor_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes

            box_reg_i = box_reg_i[anchor_idxs]
            anchors_i = anchors_i[anchor_idxs]
            # predict boxes
            predicted_boxes = self.box2box_transform.apply_deltas(box_reg_i, anchors_i.tensor)

            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)

        boxes_all, scores_all, class_idxs_all = [
            cat(x) for x in [boxes_all, scores_all, class_idxs_all]
        ]
        keep = batched_nms(boxes_all, scores_all, class_idxs_all, self.test_nms_thresh)
        keep = keep[: self.max_detections_per_image]

        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes_all[keep])
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]
        return result

    def losses(self, pred_ious, gt_ious, iou_pos_mask):

        num_images = len(gt_ious)
        gt_ious = torch.stack(gt_ious) # (N, R)
        iou_pos_mask = torch.stack(iou_pos_mask)
        num_iou_pos_anchors = iou_pos_mask.sum().item()
        get_event_storage().put_scalar("num_cls_pos_anchors", num_iou_pos_anchors / num_images)

        if self.iou_pred_loss_type == "smooth_l1":
            loss_iou = smooth_l1_loss(
                pred_ious[iou_pos_mask,:],
                gt_ious[iou_pos_mask,:],
                beta=self.smooth_l1_beta,
                reduction="sum",
            )
            pass

        elif self.iou_pred_loss_type == "bce":
            loss_iou = BCEWithLogitsLoss(
                pred_ious[iou_pos_mask,:], gt_ious[iou_pos_mask,:], reduction="sum"
            )

        else:
            raise ValueError(f"Invalid bbox reg loss type '{self.box_reg_loss_type}'")

        return {
            "loss_iou": loss_iou / num_iou_pos_anchors,
        }

class IoUPredIGHead(nn.Module):
    """
    The head used in IoUPredIG for predict IoU per feature.
    It has one subnets for the iou prediction.
    """

    @configurable
    def __init__(
        self,
        *,
        input_shape: List[ShapeSpec],
        num_anchors,
        head_config,
        norm="",
        prior_prob=0.01,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (List[ShapeSpec]): input shape
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
            out_dim = configs.OUT_DIM * num_anchors
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
        torch.nn.init.constant_(heads['IOU'].bias, bias_value)

        self.subnets = subnets
        self.heads = heads

    @classmethod
    def from_config(cls, cfg, input_shape: List[ShapeSpec]):
        num_anchors = cfg.MODEL.ANCHOR.NUM_ANCHORS

        return {
            "input_shape": input_shape,
            "num_anchors": num_anchors,
            "subnet_config": cfg.MODEL.INSTANCE_GEERATOR.SUBNET_CONFIG,
            "head_config": cfg.MODEL.INSTANCE_GEERATOR.HEAD_CONFIG,
            "prior_prob": cfg.MODEL.INSTANCE_GENERATOR.PRIOR_PROB,
            "norm": cfg.MODEL.INSTANCE_GENERATOR.NORM,
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