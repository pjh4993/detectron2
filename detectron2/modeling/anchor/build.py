# Copyright (c) Facebook, Inc. and its affiliates.
import torch

from detectron2.utils.registry import Registry

from .matcher import PAA
from .generator import box_anchor, point_anchor

ANCHOR_GENERATOR_REGISTRY = Registry("ANCHOR_GENERATOR")
ANCHOR_GENERATOR_REGISTRY.__doc__ = """
Registry for anchor generator, i.e. Box style anchors in Faster-RCNN, Retinanet and Point style anchors in FCOS, ATSS.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""

def build_anchor_generator(cfg):
    """
    Build the anchor generator, defined by ``cfg.MODEL.ANCHOR_GENERATOR``.
    Note that it does not load any weights from ``cfg``.
    """
    anchor_generator = cfg.MODEL.ANCHOR.GENERATOR.NAME
    anchor_generator = ANCHOR_GENERATOR_REGISTRY.get(anchor_generator)(cfg)
    anchor_generator.to(torch.device(cfg.MODEL.DEVICE))
    return anchor_generator 

ANCHOR_MATCHER_REGISTRY = Registry("ANCHOR_MATCHER")
ANCHOR_MATCHER_REGISTRY.__doc__ = """
Registry for anchor matcher, i.e. Anchor-GT iou based matcher as Retinanet, Anchor-GT iou based top-k matcher as ATSS 
and Probablistic anchor assignment as PAA.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""

def build_anchor_matcher(cfg):
    """
    Build the anchor matcher, defined by ``cfg.MODEL.ANCHOR_MATCHER``.
    Note that it does not load any weights from ``cfg``.
    """
    anchor_matcher = cfg.MODEL.ANCHOR.MATCHER.NAME
    anchor_matcher = ANCHOR_MATCHER_REGISTRY.get(anchor_matcher)(cfg)
    anchor_matcher.to(torch.device(cfg.MODEL.DEVICE))
    return anchor_matcher 