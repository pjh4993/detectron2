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
This file contains functions to parse SKU-format annotations into dicts in "Detectron2 format".
"""

 
logger = logging.getLogger(__name__)

__all__ = ["load_sku_json", "convert_to_sku_json"]


def load_sku_json(csv_file, image_root, dataset_name=None, extra_annotation_keys=None):
    """
    Load a json file with SKU's instances annotation format.
    Currently supports instance detection, instance segmentation,
    and person keypoints annotations.

    Args:
        json_file (str): full path to the json file in SKU instances annotation format.
        image_root (str or path-like): the directory where the images in this json file exists.
        dataset_name (str): the name of the dataset (e.g., sku_2017_train).
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
    from .pyskutools import SKU

    timer = Timer()
    csv_file = PathManager.get_local_path(csv_file)
    with contextlib.redirect_stdout(io.StringIO()):
        sku_api = SKU(csv_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(csv_file, timer.seconds()))

    # sort indices for reproducible results
    img_ids = sorted(sku_api.imgs.keys())
    # imgs is a list of dicts, each looks something like:
    # {'file_name': 'test_0.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'image_id' : 0 (same with file name)
    # }
    imgs = sku_api.loadImgs(img_ids)
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
    anns = [sku_api.imgToAnns[img_id] for img_id in img_ids]

    imgs_anns = list(zip(imgs, anns))

    logger.info("Loaded {} images in SKU format from {}".format(len(imgs_anns), csv_file))

    dataset_dicts = []

    ann_keys = sku_api.ann_keys + (extra_annotation_keys or [])

    num_instances_without_valid_segmentation = 0

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.

            # The original SKU valminusminival2014 & minival2014 annotation files
            # actually contains bugs that, together with certain ways of using SKU API,
            # can trigger this assertion.
            assert anno["image_id"] == image_id

            obj = {key: anno[key] for key in ann_keys if key in anno}

            obj["bbox_mode"] = BoxMode.XYXY_ABS
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


def convert_to_sku_dict(dataset_name):
    """
    Convert an instance detection/segmentation or keypoint detection dataset
    in detectron2's standard format into SKU json format.

    Generic dataset description can be found here:
    https://detectron2.readthedocs.io/tutorials/datasets.html#register-a-dataset

    SKU data format description can be found here:
    http://skudataset.org/#format-data

    Args:
        dataset_name (str):
            name of the source dataset
            Must be registered in DatastCatalog and in detectron2's standard format.
            Must have corresponding metadata "thing_classes"
    Returns:
        sku_dict: serializable dict in SKU json format
    """

    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)
    #??? SKU for categories
    categories = [
        {"id": reverse_id_mapper(id), "name": name}
        for id, name in enumerate(metadata.thing_classes)
    ]

    logger.info("Converting dataset dicts into SKU format")
    sku_images = []
    sku_annotations = []

    for image_id, image_dict in enumerate(dataset_dicts):
        sku_image = {
            "id": image_dict.get("image_id", image_id),
            "width": image_dict["width"],
            "height": image_dict["height"],
            "file_name": image_dict["file_name"],
        }
        sku_images.append(sku_image)

        anns_per_image = image_dict.get("annotations", [])
        for annotation in anns_per_image:
            # create a new dict with only SKU fields
            sku_annotation = {}

            # SKU requirement: XYWH box format
            bbox = annotation["bbox"]
            bbox_mode = annotation["bbox_mode"]
            bbox = BoxMode.convert(bbox, bbox_mode, BoxMode.XYWH_ABS)

            # Computing areas using bounding boxes
            bbox_xy = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
            area = Boxes([bbox_xy]).area()[0].item()

            # SKU requirement:
            #   linking annotations to images
            #   "id" field must start with 1
            sku_annotation["id"] = len(sku_annotations) + 1
            sku_annotation["image_id"] = sku_image["id"]
            sku_annotation["bbox"] = [round(float(x), 3) for x in bbox]

            sku_annotations.append(sku_annotation)

    logger.info(
        "Conversion finished, "
        f"#images: {len(sku_images)}, #annotations: {len(sku_annotations)}"
    )

    info = {
        "date_created": str(datetime.datetime.now()),
        "description": "Automatically generated SKU json file for Detectron2.",
    }
    sku_dict = {"info": info, "images": sku_images, "categories": categories, "licenses": None}
    if len(sku_annotations) > 0:
        sku_dict["annotations"] = sku_annotations
    return sku_dict


def convert_to_sku_json(dataset_name, output_file, allow_cached=True):
    """
    Converts dataset into SKU format and saves it to a json file.
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
                f"Using previously cached SKU format annotations at '{output_file}'. "
                "You need to clear the cache file if your dataset has been modified."
            )
        else:
            logger.info(f"Converting annotations of dataset '{dataset_name}' to SKU format ...)")
            sku_dict = convert_to_sku_dict(dataset_name)

            logger.info(f"Caching SKU format annotations at '{output_file}' ...")
            with PathManager.open(output_file, "w") as f:
                json.dump(sku_dict, f)


if __name__ == "__main__":
    """
    Test the SKU json dataset loader.

    Usage:
        python -m detectron2.data.datasets.sku \
            path/to/json path/to/image_root dataset_name

        "dataset_name" can be "sku_2014_minival_100", or other
        pre-registered ones
    """
    from detectron2.utils.logger import setup_logger
    from detectron2.utils.visualizer import Visualizer
    import detectron2.data.datasets  # noqa # add pre-defined metadata
    import sys

    logger = setup_logger(name=__name__)
    assert sys.argv[3] in DatasetCatalog.list()
    meta = MetadataCatalog.get(sys.argv[3])

    dicts = load_sku_json(sys.argv[1], sys.argv[2], sys.argv[3])
    logger.info("Done loading {} samples.".format(len(dicts)))

    dirname = "sku-data-vis"
    os.makedirs(dirname, exist_ok=True)
    for d in dicts:
        img = np.array(Image.open(d["file_name"]))
        visualizer = Visualizer(img, metadata=meta)
        vis = visualizer.draw_dataset_dict(d)
        fpath = os.path.join(dirname, os.path.basename(d["file_name"]))
        vis.save(fpath)