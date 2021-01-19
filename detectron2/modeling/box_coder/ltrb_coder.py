# Copyright (c) Facebook, Inc. and its affiliates.
import collections
import math
from typing import List
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.modeling.box_coder import build_box_coder
from detectron2.layers import ShapeSpec
from detectron2.structures import Boxes, RotatedBoxes
from detectron2.utils.registry import Registry

from .build import BOX_CODER_REGISTRY

BOX_CODER_REGISTRY.register()
class LTRBCoder(nn.Module):
    @configurable
    def __init__(self, *, ):
