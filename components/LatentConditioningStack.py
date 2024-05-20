import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

from blocks.LBlock import LBlock
from blocks.SpatialAttention import SpatialAttention


class LatentConditioningStack(nn.Module):
    """
    Latent Conditioning Stack implementation from https://arxiv.org/abs/2104.00954

    Args:
        batch_size: `int`

    Shape:
        - Input: N
        - Output: (N, 768, 8, 8)

    Examples:

    >>> output = LatentConditioningStack()()
    >>> output.shape
    torch.Size([5, 768, 8, 8])
    """

    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.conv3 = spectral_norm(nn.Conv2d(8, 8, 3, padding=1, stride=1))
        self.l1 = LBlock(8, 24)
        self.l2 = LBlock(24, 48)
        self.l3 = LBlock(48, 192)
        self.att = SpatialAttention(192, 192)
        self.l4 = LBlock(192, 768)

    def forward(self):
        x = torch.randn(self.batch_size, 8, 8, 8)
        x = self.conv3(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.att(x)
        x = self.l4(x)
        return x
