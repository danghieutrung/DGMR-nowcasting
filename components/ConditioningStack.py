import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

from blocks.DBlock import DBlock
from blocks.S2D import S2D


class ConditioningStack(nn.Module):
    """
    Conditioning Stack implementation from https://arxiv.org/abs/2104.00954

    Yields a stack of
    conditioning representations of sizes 64 × 64 × 48, 32 × 32 × 96, 16 × 16 × 192, and 8 × 8 × 384

    Shape:
        - Input: (N, 4, 1, 256, 256)
        - Output: [(N, 48, 64, 64), [N, 96, 32, 32], [N, 192, 16, 16], [N, 384, 8, 8]]

    Example:
    >>> input = torch.zeros((5, 4, 1, 256, 256))
    >>> output = ConditioningStack()(input)
    >>> [out.shape for out in output]
    [torch.Size([5, 48, 64, 64]), torch.Size([5, 96, 32, 32]), torch.Size([5, 192, 16, 16]), torch.Size([5, 384, 8, 8])]
    """

    def __init__(self):
        super().__init__()
        self.s2d = S2D(0.5)
        self.d1 = DBlock(4, 24)
        self.d2 = DBlock(24, 48)
        self.d3 = DBlock(48, 96)
        self.d4 = DBlock(96, 192)
        self.dblocks = nn.ModuleList(
            [DBlock(4, 24), DBlock(24, 48), DBlock(48, 96), DBlock(96, 192)]
        )
        self.conv_relu = nn.ModuleList(
            [
                nn.Sequential(
                    spectral_norm(nn.Conv2d(96, 48, 3, stride=1, padding=1)), nn.ReLU()
                ),
                nn.Sequential(
                    spectral_norm(nn.Conv2d(192, 96, 3, stride=1, padding=1)), nn.ReLU()
                ),
                nn.Sequential(
                    spectral_norm(nn.Conv2d(384, 192, 3, stride=1, padding=1)),
                    nn.ReLU(),
                ),
                nn.Sequential(
                    spectral_norm(nn.Conv2d(768, 384, 3, stride=1, padding=1)),
                    nn.ReLU(),
                ),
            ]
        )

    def forward(self, x):
        x = self.s2d(x)  # 5,4,4,128,128
        _, T, _, _, _ = x.shape

        out = [[], [], [], []]
        for i in range(T):
            obs = x[:, i, :, :, :].squeeze()
            for i, dblock in enumerate(self.dblocks):
                obs = dblock(obs)
                out[i].append(obs)

        out = [torch.cat(o, dim=1) for o in out]
        for i, obs, conv_relu in zip(range(4), out, self.conv_relu):
            out[i] = conv_relu(obs)

        return out
