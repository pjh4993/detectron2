# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.utils.registry import Registry

PART_CLUSTER_GENERATOR_REGISTRY = Registry("PART_CLUSTER_GENERATOR")
PART_CLUSTER_GENERATOR_REGISTRY.__doc__ = """
Registry for proposal generator, which produces object proposals from feature maps.

The registered object will be called with `obj(cfg, input_shape)`.
The call should return a `nn.Module` object.
"""

def build_part_cluster_generator(cfg, input_shape):
    """
    Build a proposal generator from `cfg.MODEL.PROPOSAL_GENERATOR.NAME`.
    The name can be "PrecomputedProposals" to use no proposal generator.
    """
    name = cfg.MODEL.PART_CLUSTER_GENERATOR.NAME
    if name == "PrecomputedProposals":
        return None

    return PART_CLUSTER_GENERATOR_REGISTRY.get(name)(cfg, input_shape)
