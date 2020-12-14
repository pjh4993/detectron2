# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
import pickle
from collections import OrderedDict, ChainMap
import pycocotools.mask as mask_util
import torch
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate

import detectron2.utils.comm as comm
from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.evaluation.fast_eval_api import COCOeval_opt
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.logger import create_small_table
import matplotlib.pyplot as plt
from tqdm import tqdm

from .evaluator import DatasetEvaluator


class METACOCOEvaluator(DatasetEvaluator):
    """
    Evaluate AR for object proposals, AP for instance detection/segmentation, AP
    for keypoint detection outputs using COCO's metrics.
    See http://cocodataset.org/#detection-eval and
    http://cocodataset.org/#keypoints-eval to understand its metrics.

    In addition to COCO, this evaluator is able to support any bounding box detection,
    instance segmentation, or keypoint detection dataset.
    """

    def __init__(
        self,
        dataset_name,
        tasks,
        distributed,
        output_dir=None,
        *,
        use_fast_impl=True,
        kpt_oks_sigmas=(),
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:

                    "json_file": the path to the COCO format annotation

                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            tasks (tuple[str]): tasks that can be evaluated under the given configuration.
                A task is one of "bbox", "segm", "keypoints".
                DEPRECATED pass cfgNode here to generate tasks from config
            distributed (True): if True, will collect results from all ranks and run evaluation
                in the main process.
                Otherwise, will only evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:

                1. "instance_predictions.pth" a file in torch serialization
                   format that contains all the raw original predictions.
                2. "coco_instances_results.json" a json file in COCO's result
                   format.
            use_fast_impl (bool): use a fast but **unofficial** implementation to compute AP.
                Although the results should be very close to the official implementation in COCO
                API, it is still recommended to compute results with the official API for use in
                papers. The faster implementation also uses more RAM.
            kpt_oks_sigmas (list[float]): The sigmas used to calculate keypoint OKS.
                See http://cocodataset.org/#keypoints-eval
                When empty, it will use the defaults in COCO.
                Otherwise it should be the same length as ROI_KEYPOINT_HEAD.NUM_KEYPOINTS.
        """
        self._logger = logging.getLogger(__name__)
        if isinstance(tasks, CfgNode):
            kpt_oks_sigmas = (
                tasks.TEST.KEYPOINT_OKS_SIGMAS if not kpt_oks_sigmas else kpt_oks_sigmas
            )
            self._tasks = self._tasks_from_config(tasks)
            self._logger.warn(
                "COCO Evaluator instantiated using config, this is deprecated behavior."
                " Please pass tasks in directly"
            )
        else:
            self._tasks = tasks

        self._distributed = distributed
        self._output_dir = output_dir
        self._use_fast_impl = use_fast_impl
        self.rank = comm.get_rank()
        self.image_id = 0

        self._cpu_device = torch.device("cpu")

        self._metadata = MetadataCatalog.get(dataset_name)
        if not hasattr(self._metadata, "json_file"):
            self._logger.info(
                f"'{dataset_name}' is not registered by `register_coco_instances`."
                " Therefore trying to convert it to COCO format ..."
            )

            cache_path = os.path.join(output_dir, f"{dataset_name}_coco_format.json")
            self._metadata.json_file = cache_path
            convert_to_coco_json(dataset_name, cache_path)

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)
        
        self.dataset_length = len(self._coco_api.getImgIds())
        self.img_id_hash = dict()


        self._kpt_oks_sigmas = kpt_oks_sigmas
        # Test set json files do not contain annotations (evaluation must be
        # performed using the COCO evaluation server).
        self._do_evaluation = True

    def reset(self, dataset_len):
        self._predictions = []
        self._gts = []
        self.dataset_length = dataset_len

    @staticmethod
    def _tasks_from_config(cfg):
        """
        Returns:
            tuple[str]: tasks that can be evaluated under the given configuration.
        """
        tasks = ("bbox",)
        if cfg.MODEL.MASK_ON:
            tasks = tasks + ("segm",)
        if cfg.MODEL.KEYPOINT_ON:
            tasks = tasks + ("keypoints",)
        return tasks

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        pred_instance, gt_instance = outputs
        for input, pred, gt  in zip(inputs, pred_instance, gt_instance):
            supp_set_id = [x['image_id'] for x in input['support_set']]
            set_labels = input['labels'].tolist()

            for que_input, que_output, gt_query in zip(input['query_set'], pred, gt['instances'][1]):
                curr_img_id = self.image_id + self.rank * self.dataset_length
                self.img_id_hash[curr_img_id] = que_input['image_id']
                self.image_id+=1

                predictions = {"image_id": curr_img_id,}

                gt_set = copy.deepcopy(que_input)
                gt_set['instances'] = copy.deepcopy(gt_query)
                gt_set['image_id'] = curr_img_id

                instances = que_output["instances"].to(self._cpu_device)
                predictions["instances"] = instances_to_coco_json(instances, curr_img_id, set_labels, supp_set_id)

                self._predictions.append(predictions)
                self._gts.append(gt_set)

    def evaluate(self, img_ids=None):
        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
        """
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            img_id_hash = comm.gather(self.img_id_hash, dst=0)
            img_id_hash = dict(ChainMap(*img_id_hash))

            renew_gts = comm.gather(self._gts, dst=0)
            renew_gts = list(itertools.chain(*renew_gts))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions
            img_id_hash = self.img_id_hash
            renew_gts = self._gts

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save({'predictinos': predictions, 'img_id_hash': img_id_hash}, f)

        self._results = OrderedDict()
        self._eval_predictions(set(self._tasks), renew_gts, predictions, img_ids=img_ids)
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_predictions(self, tasks, gt_results, predictions, img_ids=None):
        """
        Evaluate predictions on the given tasks.
        Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for COCO format ...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))

        # unmap the category ids for COCO
        reverse_id_mapping = {
            k: k for k in range(len(self._metadata.thing_classes))
        }
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            for result in coco_results:
                category_id = result["category_id"]
                assert (
                    category_id in reverse_id_mapping
                ), "A prediction has category_id={}, which is not available in the dataset.".format(
                    category_id
                )
                result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info(
            "Evaluating predictions with {} COCO API...".format(
                "unofficial" if self._use_fast_impl else "official"
            )
        )

        coco_gt = self._change_ann_to_coco_json(gt_results, reverse_id_mapping)

        for task in sorted(tasks):
            coco_eval = (
                _evaluate_predictions_on_coco(
                    coco_gt,
                    coco_results,
                    task,
                    self._metadata,
                    kpt_oks_sigmas=self._kpt_oks_sigmas,
                    use_fast_impl=self._use_fast_impl,
                    img_ids=img_ids,
                )
                if len(coco_results) > 0
                else None  # cocoapi does not handle empty results very well
            )

            res = self._derive_coco_results(
                coco_eval, task, class_names=self._metadata.get("thing_classes")
            )
            #self._draw_iou_score_plot(
            #    coco_eval, class_names=self._metadata.get("thing_classes")
            #)
            self._results[task] = res

    def _change_ann_to_coco_json(self, gt_results, reverse_id_mapping):
        gt_json_dict = {
            "images": [],
            "type": "instances",
            "annotations": [],
            "categories": []
        }

        bnd_id = 1
        for gt_object in tqdm(gt_results, desc='gt object to coco style json:'):
            gt_img_info = get_image_info(gt_object)
            gt_ann_info = get_coco_annotation_from_ann(gt_object, reverse_id_mapping)
            img_id = gt_img_info['id']
            gt_json_dict['images'].append(gt_img_info)
            
            for ann in gt_ann_info:
                ann.update({'image_id': img_id, 'id': bnd_id})
                gt_json_dict['annotations'].append(ann)
                bnd_id = bnd_id + 1
            
        if hasattr(self._metadata, 'thing_dataset_id_to_contiguous_id') is False:
            self._metadata.thing_dataset_id_to_contiguous_id = {
                k : k for k in range(len(self._metadata.thing_classes))
            }
        for label_id, inner_id in self._metadata.thing_dataset_id_to_contiguous_id.items():
            label_name = self._metadata.thing_classes[inner_id]
            category_info = {'supercategory': 'none', 'id': label_id, 'name': label_name}
            gt_json_dict['categories'].append(category_info)
        
        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_gt_renew.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                json.dump(gt_json_dict, f)
        
        return COCO(file_path)

    def _draw_iou_score_plot(self, coco_eval, class_names=None):
        self._output_dir
        ious = coco_eval.ious
        score = coco_eval.score_tag

        best_iou = []
        best_score = []
        for k, iou in ious.items():
            if len(iou):
                best_iou.append(iou.max(axis=1))
                best_score.append(score[k])

        best_iou = np.concatenate(best_iou)
        best_score = np.concatenate(best_score)

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.scatter(best_score, best_iou, s=2)
        ax.plot([0,1],[0,1], 'r')
        ax.set_title('IOU, score scatter graph')

        fig.set_size_inches(10.0, 10.0)
        fig.savefig(os.path.join(self._output_dir, 'fig1.png'))

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """

        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        }[iou_type]

        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = {
            metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        self._logger.info(
            "Evaluation results for {}: \n".format(iou_type) + create_small_table(results)
        )
        if not np.isfinite(sum(results.values())):
            self._logger.info("Some metrics cannot be computed and is shown as NaN.")

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP: \n".format(iou_type) + table)

        results.update({"AP-" + name: ap for name, ap in results_per_category})

        return results


def instances_to_coco_json(instances, img_id, set_labels, supp_set_id):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.

    Args:
        instances (Instances):
        img_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()
    classes = [set_labels[x] for x in classes]

    has_mask = instances.has("pred_masks")
    if has_mask:
        # use RLE to encode the masks, because they are too large and takes memory
        # since this evaluator stores outputs of the entire dataset
        rles = [
            mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            for mask in instances.pred_masks
        ]
        for rle in rles:
            # "counts" is an array encoded by mask_util as a byte-stream. Python3's
            # json writer which always produces strings cannot serialize a bytestream
            # unless you decode it. Thankfully, utf-8 works out (which is also what
            # the pycocotools/_mask.pyx does).
            rle["counts"] = rle["counts"].decode("utf-8")

    has_keypoints = instances.has("pred_keypoints")
    if has_keypoints:
        keypoints = instances.pred_keypoints

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
            "supp_set_id": supp_set_id
        }
        if has_mask:
            result["segmentation"] = rles[k]
        if has_keypoints:
            # In COCO annotations,
            # keypoints coordinates are pixel indices.
            # However our predictions are floating point coordinates.
            # Therefore we subtract 0.5 to be consistent with the annotation format.
            # This is the inverse of data loading logic in `datasets/coco.py`.
            keypoints[k][:, :2] -= 0.5
            result["keypoints"] = keypoints[k].flatten().tolist()
        results.append(result)
    return results

def get_coco_annotation_from_ann(gt_object, reverse_id_mapping):
    instance_info = gt_object['instances']
    ann_list = []
    for box_info, cls_info in zip(instance_info.get_fields()['gt_boxes'].tensor, instance_info.get_fields()['gt_classes']):
        xmin = int(box_info[0].item())
        ymin = int(box_info[1].item())
        xmax = int(box_info[2].item())
        ymax = int(box_info[3].item())

        o_width = xmax - xmin
        o_height = ymax - ymin

        ann = {
            'area': o_width * o_height,
            'iscrowd': 0,
            'bbox': [xmin, ymin, o_width, o_height],
            'category_id': reverse_id_mapping[cls_info.item()],
            #'category_id': cls_info.item(),
            'ignore': 0,
            'segmentation': []  # This script is not for segmentation
        }
        ann_list.append(ann)
    return ann_list

def get_image_info(gt_object):
    image_info = {
        'file_name': os.path.basename(gt_object['file_name']),
        'height': gt_object['instances'].image_size[0],
        'width': gt_object['instances'].image_size[1],
        'id': gt_object['image_id']
    }
    return image_info

def _evaluate_predictions_on_coco(
    coco_gt, coco_results, iou_type, metadata, kpt_oks_sigmas=None, use_fast_impl=True, img_ids=None
):
    """
    Evaluate the coco results using COCOEval API.
    """
    assert len(coco_results) > 0

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = (COCOeval_opt if use_fast_impl else COCOeval)(coco_gt, coco_dt, iou_type)
    if img_ids is not None:
        coco_eval.params.imgIds = img_ids

    if iou_type == "keypoints":
        # Use the COCO default keypoint OKS sigmas unless overrides are specified
        if kpt_oks_sigmas:
            assert hasattr(coco_eval.params, "kpt_oks_sigmas"), "pycocotools is too old!"
            coco_eval.params.kpt_oks_sigmas = np.array(kpt_oks_sigmas)
        # COCOAPI requires every detection and every gt to have keypoints, so
        # we just take the first entry from both
        num_keypoints_dt = len(coco_results[0]["keypoints"]) // 3
        num_keypoints_gt = len(next(iter(coco_gt.anns.values()))["keypoints"]) // 3
        num_keypoints_oks = len(coco_eval.params.kpt_oks_sigmas)
        assert num_keypoints_oks == num_keypoints_dt == num_keypoints_gt, (
            f"[COCOEvaluator] Prediction contain {num_keypoints_dt} keypoints. "
            f"Ground truth contains {num_keypoints_gt} keypoints. "
            f"The length of cfg.TEST.KEYPOINT_OKS_SIGMAS is {num_keypoints_oks}. "
            "They have to agree with each other. For meaning of OKS, please refer to "
            "http://cocodataset.org/#keypoints-eval."
        )

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval