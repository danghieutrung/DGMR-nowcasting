import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

from LatentConditioningStack import LatentConditioningStack

from blocks.ConvGRU import ConvGRU
from blocks.GBlock import GBlock
from blocks.LBlock import LBlock
from blocks.SpatialAttention import SpatialAttention


class ConvGRUStack(nn.Module):
    """
    ConvGRUStack for Sampler implementation from https://arxiv.org/abs/2104.00954

    Args:
        `return_hidden`: bool
            if True, return hidden state. Default: `True`

    Shape:
        - Input: [(N, 768, 8, 8), (N, 384, 16, 16), (N, 192, 32, 32), (N, 96, 64, 64)]
        - Output: ((N, 96, 64, 64), [(N, 768, 8, 8), (N, 384, 16, 16), (N, 192, 32, 32), (N, 96, 64, 64)]) or (N, 96, 64, 64) (if `return_hidden`=`False`)

    Examples:

    >>> h0 = [torch.zeros((5, 768, 8, 8)), torch.zeros((5, 384, 16, 16)), torch.zeros((5, 192, 32, 32)), torch.zeros((5, 96, 64, 64))]
    >>> output, h0 = ConvGRUStack()(h)
    """

    def __init__(self, return_hidden=True):
        super().__init__()
        self.return_hidden = return_hidden

        self.convGRUs = nn.ModuleList(
            [ConvGRU(384, 768), ConvGRU(192, 384), ConvGRU(96, 192)]
        )
        self.convGRU_4 = ConvGRU(48, 96)

        self.conv1s = nn.ModuleList(
            [
                spectral_norm(nn.Conv2d(768, 768, 1, groups=768)),
                spectral_norm(nn.Conv2d(384, 384, 1, groups=384)),
                spectral_norm(nn.Conv2d(192, 192, 1, groups=192)),
            ]
        )
        self.conv1_4 = spectral_norm(nn.Conv2d(96, 96, 1, groups=96))

        self.gblocks = nn.ModuleList(
            [
                GBlock(768, 768, upsampling=False),
                GBlock(384, 384, upsampling=False),
                GBlock(192, 192, upsampling=False),
            ]
        )
        self.gblocks_up = nn.ModuleList(
            [
                GBlock(768, 384, upsampling=True),
                GBlock(384, 192, upsampling=True),
                GBlock(192, 96, upsampling=True),
            ]
        )

    def forward(self, prev_state):
        new_state = []
        batch_size = prev_state[0].shape[0]

        x = LatentConditioningStack(batch_size)()
        for state, convGRU, conv1, G, G_up in zip(
            prev_state,
            self.convGRUs,
            self.conv1s,
            self.gblocks,
            self.gblocks_up,
        ):
            x = convGRU(state, x)
            new_state.append(x)
            x = conv1(x)
            x = G(x)
            x = G_up(x)

        x = self.convGRU_4(prev_state[-1], x)
        new_state.append(x)
        output_state = self.conv1_4(x)

        if not self.return_hidden:
            return output_state
        return (output_state, new_state)
