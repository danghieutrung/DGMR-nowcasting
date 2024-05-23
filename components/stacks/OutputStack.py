import torch.nn as nn
from torch.nn.modules.pixelshuffle import PixelShuffle
from torch.nn.utils.parametrizations import spectral_norm

from LatentConditioningStack import LatentConditioningStack

from blocks.ConvGRU import ConvGRU
from blocks.GBlock import GBlock


class OutputStack(nn.Module):
    """
    OutputStack for Sampler implementation from https://arxiv.org/abs/2104.00954

    Args:
        `return_hidden`:`bool`
            if`True`, return hidden state. Default:`True`

    Shape:
        - Input: [(N, 384, 8, 8), (N, 192, 16, 16), (N, 96, 32, 32), (N, 48, 64, 64)]
        - Output: ((N, 96, 64, 64), [(N, 768, 8, 8), (N, 384, 16, 16), (N, 192, 32, 32), (N, 96, 64, 64)]) or (N, 96, 64, 64) (if `return_hidden`=`False`)

    Examples:

    >>> h0 = [torch.zeros((5, 768, 8, 8)), torch.zeros((5, 384, 16, 16)), torch.zeros((5, 192, 32, 32)), torch.zeros((5, 96, 64, 64))]
    >>> output, h = OutputStack()(h0)
    >>> output.shape
    torch.Size([5, 1, 256, 256])
    """

    def __init__(self, return_hidden=True):
        super().__init__()
        self.return_hidden = return_hidden
        self.convGRUs = nn.ModuleList(
            [
                ConvGRU(768, 384, 384),
                ConvGRU(384, 192, 192),
                ConvGRU(192, 96, 96),
                ConvGRU(96, 48, 48),
            ]
        )
        self.conv1s = nn.ModuleList(
            [
                spectral_norm(nn.Conv2d(384, 768, 1, groups=384)),
                spectral_norm(nn.Conv2d(192, 384, 1, groups=192)),
                spectral_norm(nn.Conv2d(96, 192, 1, groups=96)),
                spectral_norm(nn.Conv2d(48, 96, 1, groups=48)),
            ]
        )
        self.gblocks = nn.ModuleList(
            [
                GBlock(768, 768, upsampling=False),
                GBlock(384, 384, upsampling=False),
                GBlock(192, 192, upsampling=False),
                GBlock(96, 96, upsampling=False),
            ]
        )
        self.gblocks_up = nn.ModuleList(
            [
                GBlock(768, 384, upsampling=True),
                GBlock(384, 192, upsampling=True),
                GBlock(192, 96, upsampling=True),
                GBlock(96, 48, upsampling=True),
            ]
        )
        self.bn = nn.BatchNorm2d(48)
        self.relu = nn.ReLU()
        self.conv1_5 = spectral_norm(nn.Conv2d(48, 4, 1))
        self.d2s = PixelShuffle(upscale_factor=2)

    def forward(self, h0):
        h = []
        batch_size = h0[0].shape[0]

        x = LatentConditioningStack(batch_size)()
        for state, convGRU, conv1, G, G_up in zip(
            h0,
            self.convGRUs,
            self.conv1s,
            self.gblocks,
            self.gblocks_up,
        ):
            x = convGRU(x, state)
            h.append(x)
            x = conv1(x)
            x = G(x)
            x = G_up(x)

        x = self.bn(x)
        x = self.relu(x)
        x = self.conv1_5(x)
        x = self.d2s(x)

        if not self.return_hidden:
            return x
        return (x, h)
