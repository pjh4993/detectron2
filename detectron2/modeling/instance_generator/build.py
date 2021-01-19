# Copyright (c) Facebook, Inc. and its affiliates.
import torch

from detectron2.utils.registry import Registry

INSTANCE_GENERATOR_REGISTRY = Registry("INSTANCE_GENERATOR")
INSTANCE_GENERATOR_REGISTRY.__doc__ = """
Registry for Instance generator, i.e. RoI-head.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def build_instance_generator(cfg):
    """
    Build the instance generator, defined by ``cfg.MODEL.INSTANCE_GENERATOR``.
    Note that it does not load any weights from ``cfg``.
    """
    meta_arch = cfg.MODEL.INSTANCE_GENERATOR.NAME
    model = INSTANCE_GENERATOR_REGISTRY.get(meta_arch)(cfg)
    model.to(torch.device(cfg.MODEL.DEVICE))
    return model
