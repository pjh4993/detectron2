# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import numpy as np
import time
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as maskUtils

from detectron2 import _C


class COCOeval_opt(COCOeval):
    """
    This is a slightly modified version of the original COCO API, where the functions evaluateImg()
    and accumulate() are implemented in C++ to speedup evaluation
    """

    def evaluate(self):
        """
        Run per image evaluation on given images and store results in self.evalImgs_cpp, a
        datastructure that isn't readable from Python but is used by a c++ implementation of
        accumulate().  Unlike the original COCO PythonAPI, we don't populate the datastructure
        self.evalImgs because this datastructure is a computational bottleneck.
        :return: None
        """
        tic = time.time()

        print("Running per image evaluation...")
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if p.useSegm is not None:
            p.iouType = "segm" if p.useSegm == 1 else "bbox"
            print("useSegm (deprecated) is not None. Running {} evaluation".format(p.iouType))
        print("Evaluate annotation type *{}*".format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params = p

        self._prepare()

        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        if p.iouType == "segm" or p.iouType == "bbox":
            computeIoU = self.computeIoU
        elif p.iouType == "keypoints":
            computeIoU = self.computeOks
        self.ious = {
            (imgId, catId): computeIoU(imgId, catId) for imgId in p.imgIds for catId in catIds
        }

        self.gtData = {
            (imgId, catId): self.computeGTData(imgId, catId) for imgId in p.imgIds for catId in catIds
        }

        self.dtData = {
            (imgId, catId): self.computeDTData(imgId, catId) for imgId in p.imgIds for catId in catIds
        }

        maxDet = p.maxDets[-1]

        # <<<< Beginning of code differences with original COCO API
        def convert_instances_to_cpp(instances, is_det=False):
            # Convert annotations for a list of instances in an image to a format that's fast
            # to access in C++
            instances_cpp = []
            for instance in instances:
                instance_cpp = _C.InstanceAnnotation(
                    int(instance["id"]),
                    instance["score"] if is_det else instance.get("score", 0.0),
                    instance["area"],
                    bool(instance.get("iscrowd", 0)),
                    bool(instance.get("ignore", 0)),
                )
                instances_cpp.append(instance_cpp)
            return instances_cpp

        # Convert GT annotations, detections, and IOUs to a format that's fast to access in C++
        ground_truth_instances = [
            [convert_instances_to_cpp(self._gts[imgId, catId]) for catId in p.catIds]
            for imgId in p.imgIds
        ]
        detected_instances = [
            [convert_instances_to_cpp(self._dts[imgId, catId], is_det=True) for catId in p.catIds]
            for imgId in p.imgIds
        ]
        ious = [[self.ious[imgId, catId] for catId in catIds] for imgId in p.imgIds]

        if not p.useCats:
            # For each image, flatten per-category lists into a single list
            ground_truth_instances = [[[o for c in i for o in c]] for i in ground_truth_instances]
            detected_instances = [[[o for c in i for o in c]] for i in detected_instances]

        # Call C++ implementation of self.evaluateImgs()
        self._evalImgs_cpp = _C.COCOevalEvaluateImages(
            p.areaRng, maxDet, p.iouThrs, ious, ground_truth_instances, detected_instances
        )
        self._evalImgs = None

        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print("COCOeval_opt.evaluate() finished in {:0.2f} seconds.".format(toc - tic))
        # >>>> End of code differences with original COCO API

    def computeDTData(self, imgId, catId):
        """
        score:      Classification score of Regressed box
        DoR:        DiagRate over Regressed box (from Head)
        DoP / DpN:  DiagRate over Proposal (from RPN) / DiagRate predicted by Network
        """

        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt=dt[0:p.maxDets[-1]]

        """
        bound variable name first
        """
        dscore = dctr = dbox = dloc = None

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            dscore = np.array([d['score'] for d in dt])
            dctr = np.array([d['dctrness'] for d in dt])
            dbox = np.array([d['bbox'] for d in dt])
            dloc = np.array([d['location'] for d in dt])
        else:
            raise Exception('unknown iouType for iou computation')

        if len(dbox) == 0:
            return np.array([])

        score = np.expand_dims(dscore, axis=1)
        DoR = self.computeDiagRate(dloc.copy(), dbox.copy())
        DoR = DoR[np.arange(DoR.shape[0]), np.arange(DoR.shape[0]),:]
        DpN = np.expand_dims(dctr, axis=1)

        return np.concatenate(
            (score, DoR, DpN), 
            axis=1)


    def computeGTData(self, imgId, catId):
        """
        DoG:        DiagRate over Ground truth box
        IoR:        Intersection over Regressed box
        """
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt=dt[0:p.maxDets[-1]]

        """
        bound variable name first
        """
        dbox = gbox = dloc = None

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            gbox = np.array([g['bbox'] for g in gt])
            dbox = np.array([d['bbox'] for d in dt])
            dloc = np.array([d['location'] for d in dt])
        else:
            raise Exception('unknown iouType for iou computation')

        if len(gbox) == 0 or len(dbox) == 0:
            return np.array([])

        DoG = self.computeDiagRate(dloc.copy(), gbox.copy())

        iscrowd = [ 1 for o in gt]
        IoR = maskUtils.iou(dbox,gbox,iscrowd)
        IoR = np.expand_dims(IoR, axis=2)

        return np.concatenate(
            (DoG, IoR),
            axis=2)

    def computeDiagRate(self, loc, bbox):
        """

            $name       $help                                       $dType

        arguments

            loc:        location to compute DiagRate                np.array(N, 2)
            bbox:       Target box to compute DiagRate              np.array(N, 4)

        return value

            diag_rate:  DiagRate computed from location given bbox  np.array(N,1)

        """
        loc = np.expand_dims(np.concatenate((loc, -loc), axis=1), axis=1)
        bbox[:,2:] += bbox[:,0:2]
        bbox[:,0:2] *= -1

        ltrb = loc + bbox
        ltrb[ltrb < 0] = 1e-5
        lr = ltrb[:,:,[0,2]]
        tb = ltrb[:,:,[1,3]]
        diag_len = (lr.sum(axis=2) ** 2 + tb.sum(axis=2) **2)/4

        diff_vec = np.concatenate(
            (np.expand_dims(lr.sum(axis=2)/2 - lr[:,:,0], axis=2), 
            np.expand_dims(tb.sum(axis=2)/2 - tb[:,:,1], axis=2)),
            axis=2)

        diag_rate = 1 - (diff_vec[:,:,0] ** 2 + diff_vec[:,:,1] **2) / diag_len

        return np.expand_dims(diag_rate, axis=2)
    

    def accumulate(self):
        """
        Accumulate per image evaluation results and store the result in self.eval.  Does not
        support changing parameter settings from those used by self.evaluate()
        """
        print("Accumulating evaluation results...")
        tic = time.time()
        if not hasattr(self, "_evalImgs_cpp"):
            print("Please run evaluate() first")

        self.eval = _C.COCOevalAccumulate(self._paramsEval, self._evalImgs_cpp)

        # recall is num_iou_thresholds X num_categories X num_area_ranges X num_max_detections
        self.eval["recall"] = np.array(self.eval["recall"]).reshape(
            self.eval["counts"][:1] + self.eval["counts"][2:]
        )

        # precision and scores are num_iou_thresholds X num_recall_thresholds X num_categories X
        # num_area_ranges X num_max_detections
        self.eval["precision"] = np.array(self.eval["precision"]).reshape(self.eval["counts"])
        self.eval["scores"] = np.array(self.eval["scores"]).reshape(self.eval["counts"])
        toc = time.time()
        print("COCOeval_opt.accumulate() finished in {:0.2f} seconds.".format(toc - tic))
