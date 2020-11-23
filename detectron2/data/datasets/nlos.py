# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import contextlib
import datetime
import io
import json
import logging
import numpy as np
import os
from fvcore.common.file_io import PathManager, file_lock
from fvcore.common.timer import Timer
from PIL import Image

from detectron2.structures import Boxes, BoxMode, PolygonMasks

from .. import DatasetCatalog, MetadataCatalog

"""
This file contains functions to parse NLOS-format annotations into dicts in "Detectron2 format".
"""

 
logger = logging.getLogger(__name__)

__all__ = ["load_nlos_json", "load_sem_seg", "convert_to_nlos_json"]


def load_nlos_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
    """
    Load a json file with NLOS's instances annotation format.
    Currently supports instance detection, instance segmentation,
    and person keypoints annotations.

    Args:
        json_file (str): full path to the json file in NLOS instances annotation format.
        image_root (str or path-like): the directory where the images in this json file exists.
        dataset_name (str): the name of the dataset (e.g., nlos_2017_train).
            If provided, this function will also put "thing_classes" into
            the metadata associated with this dataset.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "iscrowd", "bbox", "keypoints",
            "category_id", "segmentation"). The values for these keys will be returned as-is.
            For example, the densepose annotations are loaded in this way.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard dataset dicts format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    from pynlostools.nlos import NLOS

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        nlos_api = NLOS(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(nlos_api.getCatIds())
        cats = nlos_api.loadCats(cat_ids)
        # The categories in a custom json file may not be sorted.
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        meta.thing_classes = thing_classes

        # In NLOS, certain category ids are artificially removed,
        # and by convention they are always ignored.
        # We deal with NLOS's id issue and translate
        # the category ids to contiguous ids in [0, 80).

        # It works by looking at the "categories" field in the json, therefore
        # if users' own json also have incontiguous ids, we'll
        # apply this mapping as well but print a warning.
        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "nlos" not in dataset_name:
                logger.warning(
                    """
Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
"""
                )
        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.thing_dataset_id_to_contiguous_id = id_map

    # sort indices for reproducible results
    img_groupIds = sorted(nlos_api.img_groups.keys())
    # img_groups is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'NLOS_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    img_groups = nlos_api.loadImgs(img_groupIds)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'iscrowd': 0,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]
    anns = [nlos_api.img_groupToAnns[img_group_id] for img_group_id in img_groupIds]

    if "minival" not in json_file:
        # The popular valminusminival & minival annotations for NLOS2014 contain this bug.
        # However the ratio of buggy annotations there is tiny and does not affect accuracy.
        # Therefore we explicitly white-list them.
        ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
        assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique!".format(
            json_file
        )

    img_groups_anns = list(zip(img_groups, anns))

    logger.info("Loaded {} images in NLOS format from {}".format(len(img_groups_anns), json_file))

    dataset_dicts = []

    ann_keys = ["bbox", "category_id"] + (extra_annotation_keys or [])

    num_instances_without_valid_segmentation = 0

    for (img_group_dict, anno_dict_list) in img_groups_anns:
        record = {}
        record["gt_image"] = {
            "name": os.path.join(image_root, img_group_dict["group_name"],img_group_dict['gt_image']['name']),
            "height" : img_group_dict["gt_image"]["height"],
            "width" : img_group_dict["gt_image"]["width"]
        }
        record["laser_image"] = {
            "name": [os.path.join(image_root, img_group_dict["group_name"], x) for x in img_group_dict["laser_image"]["name"]],
            "height" : img_group_dict["laser_image"]["height"],
            "width" : img_group_dict["laser_image"]["width"]
        }
        record["depth_image"] = {
            "name": os.path.join(image_root, img_group_dict["group_name"],img_group_dict['depth_image']['name']),
            "height" : img_group_dict["depth_image"]["height"],
            "width" : img_group_dict["depth_image"]["width"]
        }

        image_group_id = record["image_id"] = img_group_dict["id"]

        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.

            # The original NLOS valminusminival2014 & minival2014 annotation files
            # actually contains bugs that, together with certain ways of using NLOS API,
            # can trigger this assertion.
            assert anno["image_group_id"] == image_group_id

            #assert anno.get("ignore", 0) == 0, '"ignore" in NLOS json file is not supported.'

            obj = {key: anno[key] for key in ann_keys if key in anno}

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            if id_map:
                obj["category_id"] = id_map[obj["category_id"]]
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process. "
            "A valid polygon should be a list[float] with even length >= 6."
        )
    return dataset_dicts


def convert_to_nlos_dict(dataset_name):
    """
    Convert an instance detection/segmentation or keypoint detection dataset
    in detectron2's standard format into NLOS json format.

    Generic dataset description can be found here:
    https://detectron2.readthedocs.io/tutorials/datasets.html#register-a-dataset

    NLOS data format description can be found here:
    http://nlosdataset.org/#format-data

    Args:
        dataset_name (str):
            name of the source dataset
            Must be registered in DatastCatalog and in detectron2's standard format.
            Must have corresponding metadata "thing_classes"
    Returns:
        nlos_dict: serializable dict in NLOS json format
    """

    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)

    # unmap the category mapping ids for NLOS
    if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):
        reverse_id_mapping = {v: k for k, v in metadata.thing_dataset_id_to_contiguous_id.items()}
        reverse_id_mapper = lambda contiguous_id: reverse_id_mapping[contiguous_id]  # noqa
    else:
        reverse_id_mapper = lambda contiguous_id: contiguous_id  # noqa

    categories = [
        {"id": reverse_id_mapper(id), "name": name}
        for id, name in enumerate(metadata.thing_classes)
    ]

    logger.info("Converting dataset dicts into NLOS format")
    nlos_images = []
    nlos_annotations = []

    for image_id, image_dict in enumerate(dataset_dicts):
        nlos_image = {
            "id": image_dict.get("image_id", image_id),
            "width": image_dict["width"],
            "height": image_dict["height"],
            "file_name": image_dict["file_name"],
        }
        nlos_images.append(nlos_image)

        anns_per_image = image_dict.get("annotations", [])
        for annotation in anns_per_image:
            # create a new dict with only NLOS fields
            nlos_annotation = {}

            # NLOS requirement: XYWH box format
            bbox = annotation["bbox"]
            bbox_mode = annotation["bbox_mode"]
            bbox = BoxMode.convert(bbox, bbox_mode, BoxMode.XYWH_ABS)

            # Computing areas using bounding boxes
            bbox_xy = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
            area = Boxes([bbox_xy]).area()[0].item()


            # NLOS requirement:
            #   linking annotations to images
            #   "id" field must start with 1
            nlos_annotation["id"] = len(nlos_annotations) + 1
            nlos_annotation["image_id"] = nlos_image["id"]
            nlos_annotation["bbox"] = [round(float(x), 3) for x in bbox]
            nlos_annotation["area"] = float(area)
            nlos_annotation["iscrowd"] = annotation.get("iscrowd", 0)
            nlos_annotation["category_id"] = reverse_id_mapper(annotation["category_id"])

            nlos_annotations.append(nlos_annotation)

    logger.info(
        "Conversion finished, "
        f"#images: {len(nlos_images)}, #annotations: {len(nlos_annotations)}"
    )

    info = {
        "date_created": str(datetime.datetime.now()),
        "description": "Automatically generated NLOS json file for Detectron2.",
    }
    nlos_dict = {"info": info, "images": nlos_images, "categories": categories, "licenses": None}
    if len(nlos_annotations) > 0:
        nlos_dict["annotations"] = nlos_annotations
    return nlos_dict


def convert_to_nlos_json(dataset_name, output_file, allow_cached=True):
    """
    Converts dataset into NLOS format and saves it to a json file.
    dataset_name must be registered in DatasetCatalog and in detectron2's standard format.

    Args:
        dataset_name:
            reference from the config file to the catalogs
            must be registered in DatasetCatalog and in detectron2's standard format
        output_file: path of json file that will be saved to
        allow_cached: if json file is already present then skip conversion
    """

    # TODO: The dataset or the conversion script *may* change,
    # a checksum would be useful for validating the cached data

    PathManager.mkdirs(os.path.dirname(output_file))
    with file_lock(output_file):
        if PathManager.exists(output_file) and allow_cached:
            logger.warning(
                f"Using previously cached NLOS format annotations at '{output_file}'. "
                "You need to clear the cache file if your dataset has been modified."
            )
        else:
            logger.info(f"Converting annotations of dataset '{dataset_name}' to NLOS format ...)")
            nlos_dict = convert_to_nlos_dict(dataset_name)

            logger.info(f"Caching NLOS format annotations at '{output_file}' ...")
            with PathManager.open(output_file, "w") as f:
                json.dump(nlos_dict, f)
