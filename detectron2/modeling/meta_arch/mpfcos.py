# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
import numpy as np
from typing import List
import torch
from fvcore.nn import sigmoid_focal_loss_jit, smooth_l1_loss
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable


from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import ShapeSpec, batched_nms, cat, IOULoss, Scale, Scale_grouping
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from sklearn import mixture

from ..anchor_generator import build_anchor_generator
from ..backbone import build_backbone
from ..box_regression import Box2BoxTransform
from ..matcher import MPFCOSMatcher
from ..postprocessing import detector_postprocess
from .build import META_ARCH_REGISTRY
import time
from torchviz import make_dot

__all__ = ["MPFCOS"]


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


@META_ARCH_REGISTRY.register()
class MPFCOS(nn.Module):
    """
    Implement MPFCOS in :paper:`MPFCOS`.
    """

    def __init__(self, cfg):
        super().__init__()
        # fmt: off
        self.num_classes = cfg.MODEL.MPFCOS.NUM_CLASSES
        self.in_features = cfg.MODEL.MPFCOS.IN_FEATURES
        # Loss parameters:
        self.focal_loss_alpha = cfg.MODEL.MPFCOS.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma = cfg.MODEL.MPFCOS.FOCAL_LOSS_GAMMA
        self.smooth_l1_loss_beta = cfg.MODEL.MPFCOS.SMOOTH_L1_LOSS_BETA
        # Inference parameters:
        self.score_threshold = cfg.MODEL.MPFCOS.SCORE_THRESH_TEST
        self.topk_candidates = cfg.MODEL.MPFCOS.TOPK_CANDIDATES_TEST
        self.nms_threshold = cfg.MODEL.MPFCOS.NMS_THRESH_TEST
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
        # Vis parameters
        self.vis_period = cfg.VIS_PERIOD
        self.input_format = cfg.INPUT.FORMAT
        # fmt: on
        self.backbone = build_backbone(cfg)
        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.in_features]
        self.head = MPFCOSHead(cfg, feature_shapes)
        self.box_pred = nn.Conv2d(feature_shapes[0].channels, feature_shapes[0].channels, kernel_size=1, stride=1, padding=0)
        self.stride = 4

        #----------------------------------------- Need to find usage ------------------

        self.anchor_generator = build_anchor_generator(cfg, feature_shapes)

        # Matching and loss
        self.box2box_transform = Box2BoxTransform(
            weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS, fpn_stride=cfg.MODEL.MPFCOS.FPN_STRIDE)
        self.anchor_matcher = MPFCOSMatcher(
            cfg.MODEL.MPFCOS.SCALE_PER_LEVEL, cfg.MODEL.MPFCOS.FPN_STRIDE, 1.5
        )

        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

    @property
    def device(self):
        return self.pixel_mean.device

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
        predicted_boxes = processed_results.pred_boxes.tensor.detach().cpu().numpy()

        v_pred = Visualizer(img, None)
        v_pred = v_pred.overlay_instances(boxes=predicted_boxes[0:max_boxes])
        prop_img = v_pred.get_image()
        vis_img = np.vstack((anno_img, prop_img))
        vis_img = vis_img.transpose(2, 0, 1)
        vis_name = f"Top: GT bounding boxes; Bottom: {max_boxes} Highest Scoring Results"
        storage.put_image(vis_name, vis_img)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        images = self.preprocess_image(batched_inputs)
        self.image_tensor = images.tensor
        features = self.backbone(images.tensor)
        features = [features[f] for f in self.in_features]
        self.feature_size = [[f.shape[2] , f.shape[3]] for f in features]

        anchors = self.anchor_generator(features)
        gt_instances = None

        if self.training:
            assert "instances" in batched_inputs[0], "Instance annotations are missing in training!"
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        pred_logits, pred_box_subnet, pred_emb = self.head(features, gt_instances)

        pred_logits = [permute_to_N_HWA_K(x, self.num_classes) for x in pred_logits]
        pred_box_subnet = [permute_to_N_HWA_K(x, x.shape(-1)) for x in pred_box_regress]
        pred_emb = [permute_to_N_HWA_K(x, 1) for x in pred_emb]
        # ------------------------------------------- crop ------------------------------
        B, C, _, _ = pred_logits.shape

        self.peak_threshold = pred_logits.view(B, C, -1).mean(dim=2).detach()
        peak_idx_mask = pred_logits > self.peak_threshold
        peak_idx = (pred_logits > self.peak_threshold).nonzero()
        
        if gt_id is None:
            pass
            # need per batch , per class clustering
            """
            for batch_id in range(B):
                peak_idx_batch = peak_idx[(peak_idx[:,0] == batch_id).nonzero().squeeze(1)]
                peak_emb_batch = pred_emb[batch_id, peak_idx_batch[:, 1], peak_idx_batch[:, 2], peak_idx_batch[:, 3]]
                peak_emb_numpy = peak_emb_batch.unsqueeze(-1).cpu().numpy()
                group_model = mixture.BayesianGaussianMixture(n_components=peak_idx_batch.shape[0], covariance_type='spherical', max_iter=3).fit(peak_emb_numpy)
                group_label = group_model.predict(peak_emb_numpy)
                unique_label = np.unique(group_label)        
            """
        else:
            for batch_id in range(B):

            gt_id = gt_id * peak_idx_mask
            instance_label = torch.unique(gt_id)
            instance_list = {lb : {'group' : (gt_id == lb).nonzero(), 'label' : gt_label[lb]} for lb in instance_label if lb != 0}

        #stage 2. predict box from instance_list

        # ------------------------------------------- crop ------------------------------

        if self.training:
            gt_labels, peak_in_gt_idx = self.compute_logit_target(pred_logits, gt_instances)
            gt_box, pred_box_regress = self.box_regress_from_peak(pred_box_subnet ,peak_in_gt_idx, gt_instances)

            losses = self.losses(gt_labels, pred_logits, gt_box, pred_box_regress, peak_in_gt_idx ,pred_emb)

            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    results = self.inference(
                        pred_logits, pred_box_subnet, pred_emb, images.image_sizes
                    )
                    self.visualize_training(batched_inputs, results)

            return losses
        else:
            pred_logits, pred_box_subnet, pred_emb = self.head(features)
            results = self.inference(pred_logits, pred_box_subnet, pred_emb, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def losses(self, gt_labels, pred_logits, gt_box, pred_box_regress, peak_in_gt_idx, pred_emb):
        """
        Args:

        """

        num_images = pred_logits[0].shape[0]
        box_cls_flatten = []
        box_regression_flatten = []
        centerness_flatten = []
        labels_flatten = []
        reg_targets_flatten = []
        weight_flatten = []
        reg_id_flatten = []

        for l in range(len(gt_labels)):
            box_cls_flatten.append(pred_logits[l].reshape(-1, self.num_classes))
            box_regression_flatten.append(pred_densebox_regress[l].reshape(-1, 4))
            labels_flatten.append(gt_labels[l])
            reg_targets_flatten.append(gt_boxes[l])
            centerness_flatten.append(pred_centerness[l].reshape(-1, 1))
            reg_id_flatten.append(gt_id[l])

        box_cls_flatten = torch.cat(box_cls_flatten, dim=0)
        box_regression_flatten = torch.cat(box_regression_flatten, dim=0)
        centerness_flatten = torch.cat(centerness_flatten, dim=0)
        labels_flatten = torch.cat(labels_flatten, dim=0)
        reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0)
        reg_id_flatten = torch.cat(reg_id_flatten, dim=0).to(box_cls_flatten.device)

        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)
        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        reg_id_pos_flatten = reg_id_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]

        num_pos_anchors = pos_inds.numel()

        get_event_storage().put_scalar("num_pos_anchors", num_pos_anchors / num_images)

        labels_flatten[labels_flatten < 0] = self.num_classes
        # change target to one hot vector
        gt_labels_target = F.one_hot(labels_flatten, num_classes=self.num_classes + 1)[
            :, :-1
        ]  # no loss for the last (background) class

        loss_cls = sigmoid_focal_loss_jit(
            box_cls_flatten,
            gt_labels_target.to(pred_logits[0].dtype),
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / num_pos_anchors
        """

        instance_mask = [reg_id_flatten == _id for _id in torch.unique(reg_id_flatten)]

        loss_cls = sigmoid_focal_loss_jit(
            box_cls_flatten,
            gt_labels_target.to(pred_logits[0].dtype),
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
        ) 

        loss_cls = sum([(loss_cls * mask.unsqueeze(1)).mean() for mask in instance_mask ])

        """
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

            """
            if pos_inds.numel() > 0:

                instance_mask = [reg_id_flatten == _id for _id in torch.unique(reg_id_pos_flatten)]

                centerness_targets = self.compute_centerness(reg_targets_flatten)
                sum_centerness_targets = centerness_targets.sum()

                loss_box_reg = IOULoss(loss_type="giou",reduction=None)(
                    box_regression_flatten,
                    reg_targets_flatten,
                    centerness_targets 
                )
                loss_box_reg = sum([(loss_box_reg * mask.unsqueeze(1)).mean() for mask in instance_mask ])

                loss_centerness = nn.BCEWithLogitsLoss()(
                    centerness_flatten.squeeze(),
                    centerness_targets
                )
                loss_centerness = sum([(loss_centerness * mask.unsqueeze(1)).mean() for mask in instance_mask ])
            """
        else:
            loss_box_reg = box_regression_flatten.sum()
            loss_centerness = centerness_flatten.sum()

        """
        loss_cls = loss_cls * 1e4
        loss_box_reg = loss_box_reg * 1e3
        loss_centerness = loss_centerness * 1e3
        """
        return {
            "loss_cls": loss_cls,
            "loss_box_reg": loss_box_reg,
            "loss_centerness": loss_centerness
        }

    def compute_centerness(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
            (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)


    @torch.no_grad()
    def compute_gt_logit_mask(self, gt_box, feature_size):
        gt_box = gt_box // self.stride


    @torch.no_grad()
    def compute_logit_target(self, pred_logits, gt_instances):
        """
        Args:
           pred_logits Tensor(B, C, H, W) : predicted log logit from MPFCOSHead
           gt_instances (list[Instances]): a list of N `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.

        Returns:
            Tensor(B, C, H, W) : target score respond to pred_logit
        """
        # Stage 1. Reflect gt_instance label with gaussian distribution
        target_logit = torch.zeros_like(pred_logits)

        for image_id in range(len(gt_instances)):
            curr_instance = gt_instances[image_id]
            instance_box = curr_instance.get_field('gt_boxes').tensor()
            instance_label = curr_instance.get_field('gt_classes').tensor()
            curr_target_logit = []
            for instance_id in range(curr_instance.num_instances):
                curr_target_logit.append()
                target_logit[image_id, instance_label[instance_id]] = torch.max(self.compute_gt_logit_mask(instance_box[instance_id]), target_logit[image_id, instance_label[instance_id]])


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
                anchors, pred_logits_per_image, deltas_per_image, pred_centerness_per_image, tuple(
                    image_size)
            )
            results.append(results_per_image)
        return results

    def inference_single_image(self, anchors, box_cls, box_delta, centerness, image_size):
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
        for box_cls_i, box_reg_i, anchors_i, centerness_i, curr_level in zip(box_cls, box_delta, anchors, centerness, torch.arange(len(self.feature_size), dtype=torch.long)):
            # (HxWxAxK,)
            box_cls_i = box_cls_i.flatten().sigmoid_().view(-1, self.num_classes).permute(1, 0)
            centerness_i = centerness_i.flatten().sigmoid_()
            box_cls_i = (box_cls_i * centerness_i).permute(1, 0).flatten()

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
            box_level_i = torch.ones(
                (anchors_i.tensor.shape[0]), device=box_reg_i.device) * curr_level
            # predict boxes
            predicted_boxes = self.box2box_transform.apply_densebox_deltas(
                box_reg_i, anchors_i.get_centers().repeat(1, 2), curr_level)

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
        result.pred_boxes = Boxes(boxes_all[keep])
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]
        result.level = level_all[keep]
        result.anchor = anchor_all[keep]
        return result

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        images.tensor = Variable(images.tensor, requires_grad=True)
        return images


class MPFCOSHead(nn.Module):
    """
    The head used in MPFCOS for object classification and embedding vector for group peak response.
    It has two subnets for the two tasks, with a common structure but separate parameters.
    """

    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        # fmt: off
        in_channels = input_shape[0].channels
        num_classes = cfg.MODEL.MPFCOS.NUM_CLASSES
        num_convs = cfg.MODEL.MPFCOS.NUM_CONVS
        prior_prob = cfg.MODEL.MPFCOS.PRIOR_PROB
        emb_size = cfg.MODEL.MPFCOS.EMB_SIZE
        self.peak_threshold = torch.nn.Parameter(torch.ones(1))
        # fmt: on

        cls_subnet = []
        emb_subnet = []
        box_subnet = []
        for _ in range(num_convs):
            cls_subnet.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            cls_subnet.append(nn.GroupNorm(32, in_channels))
            cls_subnet.append(nn.ReLU())
            emb_subnet.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            emb_subnet.append(nn.GroupNorm(32, in_channels))
            emb_subnet.append(nn.ReLU())
            box_subnet.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            box_subnet.append(nn.GroupNorm(32, in_channels))
            box_subnet.append(nn.ReLU())

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.emb_subnet = nn.Sequential(*emb_subnet)
        self.box_subnet = nn.Sequential(*box_subnet)
        self.cls_score = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1, padding=1
        )
        self.emb_pred = nn.Conv2d(in_channels, emb_size, kernel_size=3, stride=1, padding=1)
        self.box_reg = nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1)

        # Initialization
        for modules in [self.cls_subnet, self.emb_subnet, self.cls_score, self.emb_pred, self.box_subnet, self.box_reg]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

        # Use prior in model initialization to improve stability
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(self.cls_score.bias, bias_value)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(1)])

    def forward(self, features, gt_instances=None):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            logits (list[Tensor]): #lvl tensors, each has shape (N, AxK, Hi, Wi).
                The tensor predicts the classification probability
                at each spatial position for each of the A anchors and K object
                classes.
            emb (list[Tensor]): #lvl tensors, each has shape (N, Axk, Hi, Wi).
                The tensor predicts k-size embedding vector
                regression values for every anchor. These values are the
                relative offset between the anchor and the ground truth box.
        """

        # Stage 1. pred logit and emb , group peak from logit with emb
        # if model is in training we use gt_id to group peak
        # else we will cluster emb tag with BayesianGMM
        # Output is pred_logit, pred_emb, instance_list
        pred_logits = []
        pred_emb = []
        pred_box_subnet = []
        for l, feature in enumerate(features):
            cls_subnet = self.cls_subnet(feature)
            pred_logits.append(self.cls_score(cls_subnet))

            emb_subnet = self.emb_subnet(feature)
            pred_emb.append(self.emb_pred(emb_subnet))

            pred_box_subnet.append(self.box_subnet(feature))


        return pred_logits, pred_box_subnet, pred_emb



