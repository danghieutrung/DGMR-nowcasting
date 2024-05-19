import random
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

from blocks.DBlock import DBlock
from blocks.S2D import S2D


class SpatialDiscriminator(nn.Module):
    """
    Spatial Discriminator implementation from https://arxiv.org/abs/2104.00954

    Args:
        `n_frames`: `int`
            number of randomly picked frames. Default: 8

    Shape:
        - Input: (N, T, C, H, W)
        - Output: (N, n_frames)

    Examples:

    >>> input = torch.zeros((5, 22, 1, 256, 256))
    >>> output = TemporalDiscriminator()(input)
    >>> output.shape
    torch.Size([5, 8])
    """

    def __init__(self, n_frame=8):
        super().__init__()
        self.n_frame = n_frame
        self.avgPooling = nn.AvgPool2d(2)
        self.s2d = S2D(2)
        self.dblocks = nn.Sequential(
            DBlock(4, 48, relu_first=False),
            DBlock(48, 96),
            DBlock(96, 192),
            DBlock(192, 384),
            DBlock(384, 768),
            DBlock(768, 768, downsampling=False),
        )
        self.bn = nn.BatchNorm1d(768)
        self.linear = spectral_norm(nn.Linear(768, 1))
        self.relu = nn.ReLU()

    def forward(self, batch):
        _, T, _, _, _ = batch.shape
        out = torch.tensor([])

        for x in batch:
            # Randomly picking n_frame frames out of T frames
            indices = random.sample(range(T), self.n_frame)
            x = x[indices, :, :, :]
            x = self.avgPooling(x)
            x = self.s2d(x)
            x = self.dblocks(x)
            x = torch.sum(x, dim=(-1, -2))
            x = self.bn(x)
            x = self.linear(x)
            x = self.relu(x)
            x = x.permute(1, 0)
            out = torch.cat((out, x))
        return out
print(SpatialDiscriminator()(torch.zeros((5, 22, 1, 256, 256))).shape)