import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

from components.stacks.blocks.D3Block import D3Block
from components.stacks.blocks.DBlock import DBlock
from components.stacks.blocks.S2D import S2D
from components.stacks.blocks.utils import *


class TemporalDiscriminator(nn.Module):
    """
    Temporal Discriminator implementation from https://arxiv.org/abs/2104.00954

    Args:
        `crop_size`:`int`
            2D crop to be randomly applied to the input. Default: 128

    Shape:
        - Input: (N, T, C, H, W)
        - Output: (N, (T//2)//2)

    Examples:

    >>> input = torch.zeros((5, 1, 22, 256, 256))
    >>> output = TemporalDiscriminator()(input)
    >>> output.shape
    torch.Size([5, 1])
    """

    def __init__(self, crop_size=128):
        super().__init__()
        self.crop_size = crop_size
        self.s2d = S2D(scale_factor=0.5)
        self.d3blocks = nn.Sequential(
            D3Block(4, 48, relu_first=False),
            D3Block(48, 96),
        )
        self.dblocks = nn.Sequential(
            DBlock(96, 192),
            DBlock(192, 384),
            DBlock(384, 768),
            DBlock(768, 768, downsampling=False),
        )
        self.bn = nn.BatchNorm1d(768)
        self.linear = spectral_norm(nn.Linear(768, 1))

    def forward(self, x):
        x = random_crop(x, self.crop_size)
        x = self.s2d(x)
        x = self.d3blocks(x)
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        x = self.dblocks(x)
        x = torch.sum(x, dim=(-1, -2))
        x = self.bn(x)
        x = x.reshape(B, T, 768)
        x = self.linear(x)
        x = torch.sum(x, dim=1)
        x = F.relu(x)
        return x
