# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
import numpy as np
from typing import List , Dict
import torch
from fvcore.nn import sigmoid_focal_loss_jit, smooth_l1_loss
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import fvcore.nn.weight_init as weight_init


from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import ShapeSpec, batched_nms, cat, IOULoss, Scale, Scale_grouping, Conv2d, get_norm
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage

from ..anchor_generator import build_anchor_generator
from ..backbone import build_backbone
from ..box_regression import Box2BoxTransform
from ..postprocessing import detector_postprocess
from .build import META_ARCH_REGISTRY
import time

__all__ = ["IRNet"]


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

def to_onehot(labels, n_categories, dtype=torch.float64):
    batch_size = len(labels)
    one_hot_labels = torch.zeros(size=(1, n_categories), dtype=dtype, device=labels.device)
    for i, label in enumerate(labels):
        # Subtract 1 from each LongTensor because your
        # indexing starts at 1 and tensor indexing starts at 0
        one_hot_labels[0] = one_hot_labels[0].scatter_(dim=0, index=label, value=1.)
    return one_hot_labels

@META_ARCH_REGISTRY.register()
class IRNet(nn.Module):
    """
    Implement IRNet in :paper:`IRNet`.
    """

    def __init__(self, cfg):
        super().__init__()
        # fmt: off
        self.num_classes              = cfg.MODEL.IRNET.NUM_CLASSES
        self.in_features              = cfg.MODEL.IRNET.IN_FEATURES
        self.mode                     = cfg.MODEL.IRNET.MODE
        self.threshold                = cfg.MODEL.IRNET.THRESHOLD
        self.focal_loss_alpha         = cfg.MODEL.IRNET.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma         = cfg.MODEL.IRNET.FOCAL_LOSS_GAMMA
        self.cls_feature              = cfg.MODEL.IRNET.CLS_FEATURE
        self.fg_threshold             = cfg.MODEL.IRNET.FORE_GROUND_THRESHOLD
        self.bg_threshold             = cfg.MODEL.IRNET.BACK_GROUND_THRESHOLD
        """
        # Loss parameters:
        self.focal_loss_alpha         = cfg.MODEL.IRNET.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma         = cfg.MODEL.IRNET.FOCAL_LOSS_GAMMA
        self.smooth_l1_loss_beta      = cfg.MODEL.IRNET.SMOOTH_L1_LOSS_BETA
        """

        # Vis parameters
        self.vis_period               = cfg.VIS_PERIOD
        self.input_format             = cfg.INPUT.FORMAT
        # fmt: on
 
        self.backbone = build_backbone(cfg)

        backbone_shape = self.backbone.output_shape()
        feature_shapes = {f_key : f_item  for f_key, f_item in backbone_shape.items() if f_key in self.in_features}
        self.classification_head = Conv2d(
            feature_shapes['res5'].channels, 
            self.num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        weight_init.c2_xavier_fill(self.classification_head)

        if self.mode != "classification":
            self.displacement_head = IRNetHead(cfg, feature_shapes, cfg.MODEL.IRNET.DISP_HEAD)
            self.class_boundary_head = IRNetHead(cfg, feature_shapes, cfg.MODEL.IRNET.CLR_BND_HEAD)
            self.peak_response_head = lambda x: x

        # Matching and loss

        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))


    @property
    def device(self):
        return self.pixel_mean.device


    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        features = {f : features[f] for f in self.in_features}
        if self.mode == "classification":
            features = features[self.cls_feature]
            N, C, _, _ = features.shape
            pred_logits = self.classification_head(F.adaptive_avg_pool2d(features, (1,1)))
            pred_logits = pred_logits.flatten()
            if self.training:
                assert "instances" in batched_inputs[0]
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                gt_labels_list = [torch.unique(gt_inst.gt_classes) for gt_inst in gt_instances]
                gt_labels =  torch.cat([to_onehot(gt_label, self.num_classes) for gt_label in gt_labels_list], dim=0).flatten()
                loss = sigmoid_focal_loss_jit(
                    pred_logits,
                    gt_labels.to(pred_logits[0].dtype),
                    alpha=self.focal_loss_alpha,
                    gamma=self.focal_loss_gamma,
                    reduction="sum",
                ) / N
 
                if self.vis_period > 0:
                    storage = get_event_storage()
                    if storage.iter % self.vis_period == 0:
                        pred_cls = pred_logits.sigmoid() > self.threshold
                        precision = gt_labels[pred_cls].sum() / (pred_cls.nonzero().shape[0] + 1e-5)
                        recall = gt_labels[pred_cls].sum() / gt_labels.nonzero().shape[0]
                        storage.put_scalar("precision", precision)
                        storage.put_scalar("recall", recall)
                        storage.put_scalar("f1 score", 2 * precision * recall / (precision + recall))

                return {
                    "loss_cls" : loss
                }
            else:
                results = []
                for idx, image_size in enumerate(images.image_sizes):
                    pred_classes = (pred_logits.sigmoid() > self.threshold).nonzero().flatten()
                    result = []
                    for idx in pred_classes:
                        result.append({"category_id" : idx})
                    results.append({"classification" : result})
                return results
                
        elif self.mode == "IRNet":
            class_response_map = self.classification_head(features[self.cls_feature]).sigmoid()
            peak_response_map = self.peak_response_head(class_response_map)
            displacement_map = self.displacement_head(features)
            class_boundary_map = self.class_boundary_head(features)

            #mask class response map with fg, bg threshold
            fg_CAM = class_response_map * (class_response_map > self.fg_threshold)
            bg_CAM = class_response_map * (class_response_map < self.bg_threshold)

            return 1

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

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        images.tensor = Variable(images.tensor, requires_grad=True)
        return images

 
class IRNetHead(nn.Module):
    """
    The head used in RetinaNet for object classification and box regression.
    It has two subnets for the two tasks, with a common structure but separate parameters.
    """

    def __init__(self, cfg, input_shape, path_aggregation):
        super().__init__()
        # fmt: off
        num_classes         = cfg.MODEL.IRNET.NUM_CLASSES
        out_channels        = cfg.MODEL.IRNET.OUT_CHANNELS
        use_bias            = True
        norm                = cfg.MODEL.IRNET.NORM
        output_connection   = []
        strides = [in_shape.stride for in_shape in input_shape]
        result = input_shape.copy()
        path_agg = []
        
        for output_path in path_aggregation:
            input_key = output_path["input_key"]
            out_channels = output_path["out_channel"]
            kernel_size = output_path["kernel"]
            stride = output_path["stride"]
            out_key = output_path["out_key"]

            in_channels = sum([result[in_key].channels for in_key in input_key])
            if out_key != "last":
                out_norm = get_norm(norm, out_channels)
            else:
                out_norm = None
                use_bias = False
            out_conv = Conv2d(
                in_channels, 
                out_channels,
                kernel_size = kernel_size,
                stride = stride,
                bias = use_bias,
                norm = out_norm
            )
            weight_init.c2_xavier_fill(out_conv)
            module_name = "path_agg_{}_{}".format('-'.join(input_key),out_key)
            self.add_module(module_name, out_conv)
            path_agg.append(out_conv)
            result[out_key] = {'channels':out_channels}

        self.path_agg_convs = path_agg
        self.path_aggregation = path_aggregation
    
    def forward(self, features):
        for output_path, target_conv in zip(self.path_aggregation, self.path_agg_convs):
            input_key = output_path["input_key"]
            out_key = output_path["out_key"]
            sample = None if 'sample' in output_path.keys() else output_path['sample']
            in_channels = [features[in_key].channels for in_key in input_key]
            for i in range(len(sample)):
                if sample[i] == 1:
                    continue
                in_channels[i] = F.interpolate(in_channels[i], scale_factor=sample[i], mode="nearest")
            in_feature = torch.cat(in_channels, dim=1)
            if out_key != "last":
                out_feature = F.relu(target_conv(in_feature))
            features[out_key] = out_feature
        
        return features['last']

