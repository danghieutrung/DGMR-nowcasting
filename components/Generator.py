import sys

sys.path.append("components/stacks")

import torch
import torch.nn as nn

from stacks.ConditioningStack import ConditioningStack
from stacks.OutputStack import OutputStack


class Sampler(nn.Module):
    def __init__(self, forecast_steps=18):
        super().__init__()
        self.output_stacks = nn.ModuleList(
            [OutputStack() for _ in range(forecast_steps)]
        )

    def forward(self, h):
        out = []
        for output_stack in self.output_stacks:
            output, h = output_stack(h)
            out.append(output)

        out = torch.cat(out, dim=1)
        return out


class Generator(nn.Module):
    """
    Generator implementation from https://arxiv.org/abs/2104.00954

    Args:
        `forecast_steps`:`int`
            number of forecast steps. Default:18

    Shape:
        - Input: (N, T, 1, W, H)
        - Output: (N, `forecast_steps`, 1, W, H)

    Examples:

    >>> input = torch.zeros((5, 4, 1, 256, 256))
    >>> output = Generator(18)(input)
    >>> output.shape
    torch.Size([5, 18, 1, 256, 256])
    """

    def __init__(self, forecast_steps=18):
        super().__init__()
        self.cond_stack = ConditioningStack()
        self.sampler = Sampler(forecast_steps)

    def forward(self, x):
        h = self.cond_stack(x)
        out = self.sampler(h)
        out = out[:, :, None, :, :]
        return out
