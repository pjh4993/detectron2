# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import itertools
import logging
from typing import Dict, List, Optional, Tuple, Union
import torch
import numpy as np
import torch
from fvcore.nn import sigmoid_focal_loss_jit, smooth_l1_loss
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import math


from detectron2.layers import ShapeSpec, batched_nms_rotated, batched_nms, cat, IOULoss, Scale
from detectron2.structures import Instances, RotatedBoxes, pairwise_iou_rotated, Boxes, ImageList
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.utils.events import get_event_storage
from detectron2.data.detection_utils import convert_image_to_rgb

from ..postprocessing import detector_postprocess
from ..matcher import FCOSMatcher
from ..anchor_generator import build_anchor_generator
from ..box_regression import Box2BoxTransformRotated, Box2BoxTransform
from .build import PROPOSAL_GENERATOR_REGISTRY
from .rpn import RPN

logger = logging.getLogger(__name__)


def permute_to_N_HWA_K(tensor, K):
    """
    Transpose/reshape a tensor from (N, (Ai x K), H, W) to (N, (HxWxAi), K)
    """
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)  # Size=(N,HWA,K)
    return tensor

@PROPOSAL_GENERATOR_REGISTRY.register()
class FCOSRPN(nn.Module):
    """
    FCOS Region Proposal Network described in :paper:`FCOS`.
    """

    def __init__(self, cfg, input_shape):
        super().__init__()
        # fmt: off
        self.num_classes              = cfg.MODEL.FCOS.NUM_CLASSES
        self.in_features              = cfg.MODEL.FCOS.IN_FEATURES
        # Loss parameters:
        self.focal_loss_alpha         = cfg.MODEL.FCOS.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma         = cfg.MODEL.FCOS.FOCAL_LOSS_GAMMA
        self.smooth_l1_loss_beta      = cfg.MODEL.FCOS.SMOOTH_L1_LOSS_BETA
        # Inference parameters:
        self.score_threshold          = cfg.MODEL.FCOS.SCORE_THRESH_TEST
        self.topk_candidates          = cfg.MODEL.FCOS.TOPK_CANDIDATES_TEST
        self.nms_threshold            = cfg.MODEL.FCOS.NMS_THRESH_TEST
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
        # Vis parameters
        self.vis_period               = cfg.VIS_PERIOD
        self.input_format             = cfg.INPUT.FORMAT
        # fmt: on
 
        feature_shapes = [input_shape[f] for f in self.in_features]
        self.head = FCOSHead(cfg, feature_shapes)
        self.anchor_generator = build_anchor_generator(cfg, feature_shapes)

        # Matching and loss
        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS, fpn_stride=cfg.MODEL.FCOS.FPN_STRIDE)
        self.anchor_matcher = FCOSMatcher(
            cfg.MODEL.FCOS.SCALE_PER_LEVEL, cfg.MODEL.FCOS.FPN_STRIDE, 1.5
        )

        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

        #need to change here / check what happens at init stage

    """
    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        ret = {
            "in_features": in_features,
            "min_box_size": cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE,
            "nms_thresh": cfg.MODEL.RPN.NMS_THRESH,
            "batch_size_per_image": cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE,
            "positive_fraction": cfg.MODEL.RPN.POSITIVE_FRACTION,
            "loss_weight": {
                "loss_rpn_cls": cfg.MODEL.RPN.LOSS_WEIGHT,
                "loss_rpn_loc": cfg.MODEL.RPN.BBOX_REG_LOSS_WEIGHT * cfg.MODEL.RPN.LOSS_WEIGHT,
            },
            "anchor_boundary_thresh": cfg.MODEL.RPN.BOUNDARY_THRESH,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS, fpn_stride=cfg.MODEL.FCOS.FPN_STRIDE),
            "box_reg_loss_type": cfg.MODEL.RPN.BBOX_REG_LOSS_TYPE,
            "smooth_l1_beta": cfg.MODEL.RPN.SMOOTH_L1_BETA,
        }

        ret["pre_nms_topk"] = (cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN, cfg.MODEL.RPN.PRE_NMS_TOPK_TEST)
        ret["post_nms_topk"] = (cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN, cfg.MODEL.RPN.POST_NMS_TOPK_TEST)

        ret["anchor_generator"] = build_anchor_generator(cfg, [input_shape[f] for f in in_features])
        ret["anchor_matcher"] = FCOSMatcher(
            cfg.MODEL.FCOS.SCALE_PER_LEVEL, cfg.MODEL.FCOS.FPN_STRIDE, 1.5
        )
        ret["head"] = FCOSHead(cfg, [input_shape[f] for f in in_features])
        return ret
    """
    def visualize_training(self, batched_inputs, results):
        """
        A function used to visualize ground truth images and final network predictions.
        It shows ground truth bounding boxes on the original image and up to 20
        predicted object bounding boxes on the original image.

        Args:
            batched_inputs (list): a list that contains input to the model.
            results (List[Instances]): a list of #images elements.
        """
        from detectron2.utils.visualizer import Visualizer

        assert len(batched_inputs) == len(
            results
        ), "Cannot visualize inputs and results of different sizes"
        storage = get_event_storage()
        max_boxes = 20

        image_index = 0  # only visualize a single image
        img = batched_inputs[image_index]["image"]
        img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
        v_gt = Visualizer(img, None)
        v_gt = v_gt.overlay_instances(boxes=batched_inputs[image_index]["instances"].gt_boxes)
        anno_img = v_gt.get_image()
        processed_results = detector_postprocess(results[image_index], img.shape[0], img.shape[1])
        predicted_boxes = processed_results.proposal_boxes.tensor.detach().cpu().numpy()

        v_pred = Visualizer(img, None)
        v_pred = v_pred.overlay_instances(boxes=predicted_boxes[0:max_boxes])
        prop_img = v_pred.get_image()
        vis_img = np.vstack((anno_img, prop_img))
        vis_img = vis_img.transpose(2, 0, 1)
        vis_name = f"FCOSSMP  :  Top: GT bounding boxes; Bottom: {max_boxes} Highest Scoring Results"
        storage.put_image(vis_name, vis_img)

    @property
    def device(self):
        return self.pixel_mean.device

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        #images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def forward(
        self,
        batched_inputs,
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

        #images_ = self.preprocess_image(batched_inputs)

        features = [features[f] for f in self.in_features]
        self.feature_size = [f.shape[2] * f.shape[3] for f in features]
        anchors = self.anchor_generator(features)

        pred_logits, pred_densebox_regress, pred_centerness = self.head(features)

        # Transpose the Hi*Wi*A dimension to the middle:
        pred_logits = [permute_to_N_HWA_K(x, self.num_classes) for x in pred_logits]
        pred_densebox_regress = [permute_to_N_HWA_K(x, 4) for x in pred_densebox_regress]
        pred_centerness = [permute_to_N_HWA_K(x, 1) for x in pred_centerness]
        self.feature_num_per_level = [[features[i].shape[2] ,features[i].shape[3]] for i in range(len(features))]

        if self.training:
            assert gt_instances is not None, "Instance annotations are missing in training!"
            gt_labels, gt_boxes = self.label_anchors(anchors, gt_instances)
            losses = self.losses(anchors, pred_logits, gt_labels, pred_densebox_regress, gt_boxes, pred_centerness)
            # images.image_sizes might not exist
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    results = self.inference(
                        anchors, pred_logits, pred_densebox_regress, pred_centerness ,images.image_sizes
                    )
                    self.visualize_training(batched_inputs, results)

        else:
            losses = {}

        results = self.inference(anchors, pred_logits, pred_densebox_regress, pred_centerness, images.image_sizes)
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            results, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            #r = detector_postprocess(results_per_image, height, width)
            results = Instances((height, width), **results_per_image.get_fields())
            processed_results.append(results)

        return processed_results, losses

    def losses(self, anchors, pred_logits, gt_labels, pred_densebox_regress, gt_boxes, pred_centerness):
        """
        Args:
            anchors (list[Boxes]): a list of #feature level Boxes
            gt_labels, gt_boxes: see output of :meth:`RetinaNet.label_anchors`.
                Their shapes are (N, R) and (N, R, 4), respectively, where R is
                the total number of anchors across levels, i.e. sum(Hi x Wi x Ai)
            pred_logits, pred_densebox_regress: both are list[Tensor]. Each element in the
                list corresponds to one level and has shape (N, Hi * Wi * Ai, K or 4).
                Where K is the number of classes used in `pred_logits`.

        Returns:
            dict[str, Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_cls" and "loss_box_reg"
        """

        num_images = pred_logits[0].shape[0]
        box_cls_flatten = []
        box_regression_flatten = []
        centerness_flatten = []
        labels_flatten = []
        reg_targets_flatten = []

        for l in range(len(gt_labels)):
           box_cls_flatten.append(pred_logits[l].reshape(-1, self.num_classes))
           box_regression_flatten.append(pred_densebox_regress[l].reshape(-1, 4))
           labels_flatten.append(gt_labels[l])
           reg_targets_flatten.append(gt_boxes[l])
           centerness_flatten.append(pred_centerness[l].reshape(-1,1))

        box_cls_flatten = torch.cat(box_cls_flatten, dim=0)
        box_regression_flatten = torch.cat(box_regression_flatten, dim=0)
        centerness_flatten = torch.cat(centerness_flatten, dim=0)
        labels_flatten = torch.cat(labels_flatten, dim=0)
        reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0)

        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)
        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]

        num_pos_anchors = pos_inds.numel()

        get_event_storage().put_scalar("num_pos_anchors", num_pos_anchors / num_images)


        labels_flatten[labels_flatten < 0] = self.num_classes
        #change target to one hot vector
        gt_labels_target = F.one_hot(labels_flatten, num_classes=self.num_classes+1)[
            :, :-1
        ]  # no loss for the last (background) class

        loss_cls = sigmoid_focal_loss_jit(
            box_cls_flatten,
            gt_labels_target.to(pred_logits[0].dtype),
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / num_pos_anchors

        if pos_inds.numel() > 0:
            centerness_targets = self.compute_centerness(reg_targets_flatten)
            sum_centerness_targets = centerness_targets.sum()

            loss_box_reg = IOULoss(loss_type="giou")(
                box_regression_flatten,
                reg_targets_flatten,
                centerness_targets 
            ) / sum_centerness_targets

            loss_centerness = nn.BCEWithLogitsLoss(reduction="sum")(
                centerness_flatten.squeeze(),
                centerness_targets
            ) / num_pos_anchors
        else:
            loss_box_reg = box_regression_flatten.sum()
            loss_centerness = centerness_flatten.sum()
        return {
            "loss_cls": loss_cls ,
            "loss_box_reg": loss_box_reg ,
            "loss_centerness" : loss_centerness 
        }

    def compute_centerness(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)
        
    @torch.no_grad()
    def label_anchors(self, anchors, gt_instances):
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
        #anchors = Boxes.cat(anchors)  # Rx4

        gt_labels, matched_gt_boxes  = self.anchor_matcher(gt_instances, anchors)

        return gt_labels, matched_gt_boxes

    def inference(self, anchors, pred_logits, pred_densebox_regress, pred_centerness, image_sizes):
        """
        Arguments:
            anchors (list[Boxes]): A list of #feature level Boxes.
                The Boxes contain anchors of this image on the specific feature level.
            pred_logits, pred_densebox_regress: list[Tensor], one per level. Each
                has shape (N, Hi * Wi * Ai, K or 4)
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        results = []
        for img_idx, image_size in enumerate(image_sizes):
            pred_logits_per_image = [x[img_idx] for x in pred_logits]
            deltas_per_image = [x[img_idx] for x in pred_densebox_regress]
            pred_centerness_per_image = [x[img_idx] for x in pred_centerness]
            results_per_image = self.inference_single_image(
                anchors, pred_logits_per_image, deltas_per_image, pred_centerness_per_image ,tuple(image_size)
            )
            results.append(results_per_image)
        return results

    def inference_single_image(self, anchors, box_cls, box_delta, centerness ,image_size):
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
        level_all = []
        anchor_all = []

        # Iterate over every feature level
        for box_cls_i, box_reg_i, anchors_i, centerness_i, curr_level in zip(box_cls, box_delta, anchors, centerness, torch.arange(len(self.feature_size),dtype=torch.long)):
            # (HxWxAxK,)
            box_cls_i = box_cls_i.flatten().sigmoid_().view(-1, self.num_classes).permute(1,0)
            centerness_i = centerness_i.flatten().sigmoid_()
            box_cls_i = (box_cls_i * centerness_i).permute(1,0).flatten()

            # Keep top k top scoring indices only.
            num_topk = min(self.topk_candidates, box_reg_i.size(0))
            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = box_cls_i.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]
 
            # filter out the proposals with low confidence score
            keep_idxs = predicted_prob > self.score_threshold
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            anchor_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes

            box_reg_i = box_reg_i[anchor_idxs]
            anchors_i = anchors_i[anchor_idxs]
            box_level_i = torch.ones((anchors_i.tensor.shape[0]), device=box_reg_i.device) * curr_level
            # predict boxes
            predicted_boxes = self.box2box_transform.apply_densebox_deltas(box_reg_i, anchors_i.get_centers().repeat(1,2), curr_level)

            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)
            level_all.append(box_level_i)
            anchor_all.append(anchors_i.get_centers())

        boxes_all, scores_all, class_idxs_all, level_all, anchor_all = [
            cat(x) for x in [boxes_all, scores_all, class_idxs_all, level_all, anchor_all]
        ]
        keep = batched_nms(boxes_all, scores_all, class_idxs_all, self.nms_threshold)
        keep = keep[: self.max_detections_per_image]

        result = Instances(image_size)
        result.proposal_boxes = Boxes(boxes_all[keep])
        result.objectness_logits = scores_all[keep]
        #result.pred_classes = class_idxs_all[keep]
        #result.level = level_all[keep]
        #result.anchor = anchor_all[keep]
        return result

class FCOSHead(nn.Module):
    """
    The head used in RetinaNet for object classification and box regression.
    It has two subnets for the two tasks, with a common structure but separate parameters.
    """

    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        # fmt: off
        in_channels      = input_shape[0].channels
        num_classes      = cfg.MODEL.FCOS.NUM_CLASSES
        num_convs        = cfg.MODEL.FCOS.NUM_CONVS
        prior_prob       = cfg.MODEL.FCOS.PRIOR_PROB
        # fmt: on
        self.centerness_on_cls = cfg.MODEL.FCOS.CENTERNESS_ON_CLS
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDE

        cls_subnet = []
        bbox_subnet = []
        for _ in range(num_convs):
            cls_subnet.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            cls_subnet.append(nn.GroupNorm(32, in_channels))
            cls_subnet.append(nn.ReLU())
            bbox_subnet.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            bbox_subnet.append(nn.GroupNorm(32, in_channels))
            bbox_subnet.append(nn.ReLU())

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.cls_score = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1, padding=1
        )
        self.bbox_pred = nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1)
        self.centerness_pred = nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1)

        # Initialization
        for modules in [self.cls_subnet, self.bbox_subnet, self.cls_score, self.bbox_pred, self.centerness_pred]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)


        # Use prior in model initialization to improve stability
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(self.cls_score.bias, bias_value)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])
        #self.scales = nn.ModuleList([Scale_grouping(in_channels, 1 ,init_value=1.0) for _ in range(5)])

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            logits (list[Tensor]): #lvl tensors, each has shape (N, AxK, Hi, Wi).
                The tensor predicts the classification probability
                at each spatial position for each of the A anchors and K object
                classes.
            bbox_reg (list[Tensor]): #lvl tensors, each has shape (N, Ax4, Hi, Wi).
                The tensor predicts 4-vector (dx,dy,dw,dh) box
                regression values for every anchor. These values are the
                relative offset between the anchor and the ground truth box.
            centerness (list[Tensor]) : #lvl tensors, each has shape (N, Ax1, Hi, Wi)
        """
        logits = []
        bbox_reg = []
        centerness = []
        for l ,feature in enumerate(features):
            cls_subnet = self.cls_subnet(feature)
            logits.append(self.cls_score(cls_subnet))
            
            box_subnet = self.bbox_subnet(feature)
            bbox_reg.append(F.relu(self.scales[l](self.bbox_pred(box_subnet))))
            #bbox_reg.append(F.relu(self.scales[l](box_subnet) * self.bbox_pred(box_subnet)))

            if self.centerness_on_cls:
                centerness.append(self.centerness_pred(cls_subnet))
            else:
                centerness.append(self.centerness_pred(box_subnet))
	
        return logits, bbox_reg, centerness

