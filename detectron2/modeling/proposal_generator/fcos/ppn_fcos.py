import logging
from numpy.core.fromnumeric import reshape
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from detectron2.layers import cat, IOULoss, IdLoss
from detectron2.structures import Instances, Boxes
from detectron2.utils.comm import get_world_size
from fvcore.nn import sigmoid_focal_loss_jit
import numpy as np

from ..ppn import PPN
from ..build import PROPOSAL_GENERATOR_REGISTRY


logger = logging.getLogger(__name__)

INF = 100000000

"""
Shape shorthand in this module:

    N: number of images in the minibatch
    L: number of feature maps per image on which RPN is run
    Hi, Wi: height and width of the i-th feature map
    4: size of the box parameterization

Naming convention:

    labels: refers to the ground-truth class of an position.

    reg_targets: refers to the 4-d (left, top, right, bottom) distances that parameterize the ground-truth box.

    logits_pred: predicted classification scores in [-inf, +inf];
    
    reg_pred: the predicted (left, top, right, bottom), corresponding to reg_targets 

    ctrness_pred: predicted centerness scores

"""



def reduce_sum(tensor):
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def compute_ctrness_targets(reg_targets):
    if len(reg_targets) == 0:
        return reg_targets.new_zeros(len(reg_targets))
    left_right = reg_targets[:, [0, 2]] + 1e-5
    top_bottom = reg_targets[:, [1, 3]] + 1e-5
    ctrness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                 (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
    return torch.sqrt(ctrness)

def compute_pi_diag_targets(reg_targets):
    if len(reg_targets) == 0:
        return reg_targets.new_zeros(len(reg_targets))
    diag = (reg_targets[:,[0,2]].sum(axis=1) ** 2 + reg_targets[:,[1,3]].sum(axis=1) **2)/4
    anchor_loc = torch.cat(((reg_targets[:,[0,2]].sum(axis=1)/2 - reg_targets[:,0]).unsqueeze(1), 
                       (reg_targets[:,[1,3]].sum(axis=1)/2 - reg_targets[:,1]).unsqueeze(1)), dim=1)
    #anchor_loc /= anchor_loc.norm(dim=1)
    diag_rate = (anchor_loc[:,0] ** 2 + anchor_loc[:,1] **2) / diag
    #diag_pi = torch.atan2(anchor_loc[:,1], anchor_loc[:,0]) * 2 / np.pi
    return 1 - diag_rate#, diag_pi

@PROPOSAL_GENERATOR_REGISTRY.register()
class FCOS_PPN(PPN):
    def __init__(self, cfg, input_shape):
        super(FCOS_PPN, self).__init__(cfg, input_shape)

        self.focal_loss_alpha = cfg.MODEL.FCOS_PPN.LOSS_ALPHA
        self.focal_loss_gamma = cfg.MODEL.FCOS_PPN.LOSS_GAMMA
        self.center_sample = cfg.MODEL.FCOS_PPN.CENTER_SAMPLE
        self.radius = cfg.MODEL.FCOS_PPN.POS_RADIUS
        self.pre_nms_thresh_train = cfg.MODEL.FCOS_PPN.INFERENCE_TH_TRAIN
        self.pre_nms_topk_train = cfg.MODEL.FCOS_PPN.PRE_NMS_TOPK_TRAIN
        self.loc_loss_func = IOULoss(cfg.MODEL.FCOS_PPN.LOC_LOSS_TYPE)
        self.id_loss_func = IdLoss()


        self.pre_nms_thresh_test = cfg.MODEL.FCOS_PPN.INFERENCE_TH_TEST
        self.pre_nms_topk_test = cfg.MODEL.FCOS_PPN.PRE_NMS_TOPK_TEST
        self.nms_thresh = cfg.MODEL.FCOS_PPN.NMS_TH

        self.num_classes = cfg.MODEL.FCOS_PPN.NUM_CLASSES
        self.strides = cfg.MODEL.FCOS_PPN.FPN_STRIDES

        # generate sizes of interest
        soi = []
        prev_size = -1
        for s in cfg.MODEL.FCOS_PPN.SIZES_OF_INTEREST:
            soi.append([prev_size, s])
            prev_size = s
        soi.append([prev_size, INF])
        self.sizes_of_interest = soi

    def _transpose(self, training_targets, num_loc_list):
        '''
        This function is used to transpose image first training targets to level first ones
        :return: level first training targets
        '''
        for im_i in range(len(training_targets)):
            training_targets[im_i] = torch.split(
                training_targets[im_i], num_loc_list, dim=0
            )

        targets_level_first = []
        for targets_per_level in zip(*training_targets):
            targets_level_first.append(
                torch.cat(targets_per_level, dim=0)
            )
        return targets_level_first

    def _get_ground_truth(self, locations, gt_instances):
        num_loc_list = [len(loc) for loc in locations]

        # compute locations to size ranges
        loc_to_size_range = []
        for l, loc_per_level in enumerate(locations):
            loc_to_size_range_per_level = loc_per_level.new_tensor(self.sizes_of_interest[l])
            loc_to_size_range.append(
                loc_to_size_range_per_level[None].expand(num_loc_list[l], -1)
            )

        loc_to_size_range = torch.cat(loc_to_size_range, dim=0)
        locations = torch.cat(locations, dim=0)

        training_targets = self.compute_targets_for_locations(
            locations, gt_instances, loc_to_size_range, num_loc_list
        )

        training_targets["locations"] = [locations.clone() for _ in range(len(gt_instances))]
        training_targets["im_inds"] = [
            locations.new_ones(locations.size(0), dtype=torch.long) * i for i in range(len(gt_instances))
        ]

        # transpose im first training_targets to level first ones
        training_targets = {
            k: self._transpose(v, num_loc_list) for k, v in training_targets.items()
        }

        training_targets["fpn_levels"] = [
            loc.new_ones(len(loc), dtype=torch.long) * level
            for level, loc in enumerate(training_targets["locations"])
        ]

        # we normalize reg_targets by FPN's strides here
        reg_targets = training_targets["reg_targets"]
        for l in range(len(reg_targets)):
            reg_targets[l] = reg_targets[l] / float(self.strides[l])

        return training_targets

    def get_sample_region(self, boxes, strides, num_loc_list, loc_xs, loc_ys, bitmasks=None, radius=1):
        if bitmasks is not None:
            _, h, w = bitmasks.size()

            ys = torch.arange(0, h, dtype=torch.float32, device=bitmasks.device)
            xs = torch.arange(0, w, dtype=torch.float32, device=bitmasks.device)

            m00 = bitmasks.sum(dim=-1).sum(dim=-1).clamp(min=1e-6)
            m10 = (bitmasks * xs).sum(dim=-1).sum(dim=-1)
            m01 = (bitmasks * ys[:, None]).sum(dim=-1).sum(dim=-1)
            center_x = m10 / m00
            center_y = m01 / m00
        else:
            center_x = boxes[..., [0, 2]].sum(dim=-1) * 0.5
            center_y = boxes[..., [1, 3]].sum(dim=-1) * 0.5

        num_gts = boxes.shape[0]
        K = len(loc_xs)
        boxes = boxes[None].expand(K, num_gts, 4)
        center_x = center_x[None].expand(K, num_gts)
        center_y = center_y[None].expand(K, num_gts)
        center_gt = boxes.new_zeros(boxes.shape)
        # no gt
        if center_x.numel() == 0 or center_x[..., 0].sum() == 0:
            return loc_xs.new_zeros(loc_xs.shape, dtype=torch.uint8)
        beg = 0
        for level, num_loc in enumerate(num_loc_list):
            end = beg + num_loc
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt
            center_gt[beg:end, :, 0] = torch.where(xmin > boxes[beg:end, :, 0], xmin, boxes[beg:end, :, 0])
            center_gt[beg:end, :, 1] = torch.where(ymin > boxes[beg:end, :, 1], ymin, boxes[beg:end, :, 1])
            center_gt[beg:end, :, 2] = torch.where(xmax > boxes[beg:end, :, 2], boxes[beg:end, :, 2], xmax)
            center_gt[beg:end, :, 3] = torch.where(ymax > boxes[beg:end, :, 3], boxes[beg:end, :, 3], ymax)
            beg = end
        left = loc_xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 2] - loc_xs[:, None]
        top = loc_ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - loc_ys[:, None]
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask

    def compute_targets_for_locations(self, locations, targets, size_ranges, num_loc_list):
        labels = []
        reg_targets = []
        target_inds = []
        valid_sample = []
        invalid_sample = []

        xs, ys = locations[:, 0], locations[:, 1]

        num_targets = 0
        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            bboxes = targets_per_im.gt_boxes.tensor
            labels_per_im = targets_per_im.gt_classes

            # no gt
            if bboxes.numel() == 0:
                labels.append(labels_per_im.new_zeros(locations.size(0)) + self.num_classes)
                reg_targets.append(locations.new_zeros((locations.size(0), 4)))
                target_inds.append(labels_per_im.new_zeros(locations.size(0)) - 1)
                continue

            area = targets_per_im.gt_boxes.area()

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

            is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0
            if self.center_sample:
                if targets_per_im.has("gt_bitmasks_full"):
                    bitmasks = targets_per_im.gt_bitmasks_full
                else:
                    bitmasks = None
                is_in_radius = self.get_sample_region(
                    bboxes, self.strides, num_loc_list, xs, ys,
                    bitmasks=bitmasks, radius=self.radius
                )
            else:
                is_in_radius = is_in_boxes

            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= size_ranges[:, [0]]) & \
                (max_reg_targets_per_im <= size_ranges[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_radius == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
            target_inds_per_im = locations_to_gt_inds + num_targets
            num_targets += len(targets_per_im)

            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = self.num_classes

            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)
            target_inds.append(target_inds_per_im)

            is_in_boxes = is_in_boxes[range(len(locations)), locations_to_gt_inds]
            is_cared_in_the_level = is_cared_in_the_level[range(len(locations)), locations_to_gt_inds]

            valid_inds = is_in_boxes * is_cared_in_the_level
            valid_sample.append(valid_inds)
            invalid_sample.append(valid_inds == 0)

        return {
            "labels": labels,
            "reg_targets": reg_targets,
            "target_inds": target_inds,
            "valid_inds": valid_sample,
            "invalid_inds": invalid_sample
        }

    def losses(self, anchors, logits_pred, reg_pred, id_vec_pred, gt_instances):
        """
        Return the losses from a set of FCOS_PPN predictions and their associated ground-truth.

        Returns:
            dict[loss name -> loss value]: A dict mapping from loss name to loss value.
        """
        locations = [
            x.get_centers()
            for x in anchors
        ]

        training_targets = self._get_ground_truth(locations, gt_instances)

        # Collect all logits and regression predictions over feature maps
        # and images to arrive at the same shape as the labels and targets
        # The final ordering is L, N, H, W from slowest to fastest axis.

        instances = Instances((0, 0))
        instances.labels = cat([
            # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
            x.reshape(-1) for x in training_targets["labels"]
        ], dim=0)
        instances.gt_inds = cat([
            # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
            x.reshape(-1) for x in training_targets["target_inds"]
        ], dim=0)
        instances.im_inds = cat([
            x.reshape(-1) for x in training_targets["im_inds"]
        ], dim=0)
        instances.reg_targets = cat([
            # Reshape: (N, Hi, Wi, 4) -> (N*Hi*Wi, 4)
            x.reshape(-1, 4) for x in training_targets["reg_targets"]
        ], dim=0,)
        instances.locations = cat([
            x.reshape(-1, 2) for x in training_targets["locations"]
        ], dim=0)
        instances.fpn_levels = cat([
            x.reshape(-1) for x in training_targets["fpn_levels"]
        ], dim=0)

        instances.valid_inds = cat([
            x.reshape(-1) for x in training_targets["valid_inds"]
        ])
        instances.invalid_inds = cat([
            x.reshape(-1) for x in training_targets["invalid_inds"]
        ])

        instances.logits_pred = cat([
            x.reshape(-1, self.num_classes) for x in logits_pred
        ], dim=0)
        instances.reg_pred = cat([
            x.reshape(-1, 4) for x in reg_pred
        ], dim=0,)
        instances.identity_pred = cat([
            x.reshape(-1, 1) for x in id_vec_pred
        ], dim=0,)

        return self.fcos_losses(instances)

    def fcos_losses(self, instances):
        num_classes = instances.logits_pred.size(1)
        assert num_classes == self.num_classes

        labels = instances.labels.flatten()
        num_gpus = get_world_size()

        pos_inds = torch.nonzero(labels != num_classes).squeeze(1)
        num_pos_local = pos_inds.numel()
        total_num_pos = reduce_sum(pos_inds.new_tensor([num_pos_local])).item()
        num_pos_avg = max(total_num_pos / num_gpus, 1.0)

        """
        neg_inds = torch.nonzero(labels == num_classes).squeeze(1)
        num_neg_local = neg_inds.numel()
        total_num_neg = reduce_sum(pos_inds.new_tensor([num_pos_local])).item()
        num_neg_avg = max(total_num_neg / num_gpus, 1.0)
        """

        # prepare one_hot
        class_target = torch.zeros_like(instances.logits_pred)
        class_target[pos_inds, labels[pos_inds]] = 1

        class_loss = sigmoid_focal_loss_jit(
            instances.logits_pred,
            class_target,
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / num_pos_avg
        
        instances = instances[pos_inds]
        instances.pos_inds = pos_inds

        if pos_inds.numel() > 0:
            reg_loss = self.loc_loss_func(
                instances.reg_pred,
                instances.reg_targets,
            )

            id_vec_positive_sample = instances.identity_pred
            id_vec_gt = instances.gt_inds
            id_vec_grouped_to_gt = []

            for uid in torch.unique(id_vec_gt):
                id_vec_grouped_to_gt.append(id_vec_positive_sample[id_vec_gt == uid])
            
            pull_loss, push_loss = self.id_loss_func(
                id_vec_grouped_to_gt
            ) 

        else:
            reg_loss = instances.reg_pred.sum() * 0
            pull_loss = push_loss = instances.identity_pred.sum() * 0

        losses = {
            "loss_fcos_cls": class_loss,
            "loss_fcos_loc": reg_loss,
            "loss_fcos_pull": pull_loss,
            "loss_fcos_push": push_loss,
        }
        extras = {
            "instances": instances,
        }
        return extras, losses
 
    def predict_part_proposals(
            self, locale_info, box_feature, cls_feature, logits_pred ,id_vec_pred,
            image_sizes
    ):
        if self.training:
            self.pre_nms_thresh = self.pre_nms_thresh_train
            self.pre_nms_topk = self.pre_nms_topk_train
        else:
            self.pre_nms_thresh = self.pre_nms_thresh_test
            self.pre_nms_topk = self.pre_nms_topk_test

        sampled_boxes = []

        bundle = {
            "l": locale_info, "o": logits_pred,
            "b": box_feature, "c": cls_feature,
            "i": id_vec_pred
        }

        for _, per_bundle in enumerate(zip(*bundle.values())):
            # get per-level bundle
            per_bundle = dict(zip(bundle.keys(), per_bundle))
            # recall that during training, we normalize regression targets with FPN's stride.
            # we denormalize them here.
            l = per_bundle["l"]
            o = per_bundle["o"]
            b = per_bundle["b"]
            c = per_bundle["c"]
            i = per_bundle["i"]

            sampled_boxes.append(
                self.forward_for_single_feature_map(
                    l, o, b, c, i, image_sizes
                )
            )

        return sampled_boxes

    def forward_for_single_feature_map(
            self, locale_info, 
            logits_pred, box_feature, cls_feature, id_vec_pred,
            image_sizes,
    ):
        N, _, H, W = box_feature.shape
        # put in the same format as locations
        logits_pred = logits_pred.sigmoid()
        box_feature = box_feature.view(N, -1, H, W).permute(0, 2, 3, 1).reshape(N, H*W, -1)
        cls_feature = cls_feature.view(N, -1, H, W).permute(0, 2, 3, 1).reshape(N, H*W, -1)

        candidate_inds = logits_pred > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.reshape(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_topk)

        results = []
        for i in range(N):
            per_box_cls = logits_pred[i]
            per_candidate_inds = candidate_inds[i]
            per_box_cls = per_box_cls[per_candidate_inds]
            
            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]

            per_box_feature = box_feature[i]
            per_box_feature = per_box_feature[per_box_loc]

            per_cls_feature = cls_feature[i]
            per_cls_feature = per_cls_feature[per_box_loc]

            per_id_vec = id_vec_pred[i]
            per_id_vec = per_id_vec[per_box_loc]

            per_locale_info = locale_info[per_box_loc]

            per_pre_nms_top_n = pre_nms_top_n[i]

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_box_feature = per_box_feature[top_k_indices]
                per_cls_feature = per_cls_feature[top_k_indices]
                per_locale_info = per_locale_info[top_k_indices]
                per_id_vec = per_id_vec[top_k_indices]

            boxlist = Instances(image_sizes[i])
            boxlist.box_feature = per_box_feature
            boxlist.cls_feature = per_cls_feature
            boxlist.locale_info = per_locale_info
            boxlist.id_vec = per_id_vec

            results.append(boxlist)

        return results
