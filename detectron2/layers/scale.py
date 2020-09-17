import torch
from . import Conv2d
from torch import nn
import torch.nn.functional as F


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

class Scale_grouping(nn.Module):
    def __init__(self, in_channels, num_convs ,init_value=1.0,):
        super(Scale_grouping, self).__init__()
        scale_subnet = []
        for _ in range(num_convs):
            scale_subnet.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            scale_subnet.append(nn.GroupNorm(32, in_channels))
            scale_subnet.append(nn.ReLU())
        
        self.scale_subnet = nn.Sequential(*scale_subnet)
        self.scale_reg = nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1)
        for modules in [self.scale_subnet, self.scale_reg]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)
        
    def forward(self, input):
        scale_subnet = self.scale_subnet(input)
        scaled_out = (self.scale_reg(scale_subnet) + 1)
        return scaled_out
