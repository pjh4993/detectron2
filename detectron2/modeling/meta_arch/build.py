# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch

from detectron2.utils.registry import Registry
from detectron2.modeling.PeakResponseMapping import PeakResponseMapping

META_ARCH_REGISTRY = Registry("META_ARCH")  # noqa F401 isort:skip
META_ARCH_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def build_model(cfg):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
    model.to(torch.device(cfg.MODEL.DEVICE))
    if cfg.MODEL.PRM.ATTACH:
        model.prm = PeakResponseMapping({
            'backbone' : model.backbone,
            'cls_subnet' : model.head.cls_subnet,
            'cls_score' : model.head.cls_score,
            'cfg':cfg.MODEL,
        })
    return model
