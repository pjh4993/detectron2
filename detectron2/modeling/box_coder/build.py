# Copyright (c) Facebook, Inc. and its affiliates.
import torch

from detectron2.utils.registry import Registry

BOX_CODER_REGISTRY = Registry("BOX_CODER")
BOX_CODER_REGISTRY.__doc__ = """
Registry for Box coder, i.e. Box2BoxTransform in Faster-RCNN.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def build_box_coder(cfg):
    """
    Build the instance generator, defined by ``cfg.MODEL.BOX_CODER``.
    Note that it does not load any weights from ``cfg``.
    """
    box_coder = cfg.MODEL.BOX_CODER.NAME
    box_coder = BOX_CODER_REGISTRY.get(box_coder)(cfg)
    box_coder.to(torch.device(cfg.MODEL.DEVICE))
    return box_coder 