import torch
import torch.nn as nn


class LBlock(nn.Module):
    """
    LBlock implementation from https://arxiv.org/abs/2104.00954

    Args:
        `in_channels`: `int`
            number of input channels
        `out_channels`: `int`
            number of output channels

    Shape:
        - Input: (N, C_in, W, H)
        - Output: (N, C_out, W, H)

    Example:

    >>> input = torch.zeros((5, 4, 8, 8))
    >>> output = DBlock(4, 48)(input)
    >>> output.shape
    torch.Size([5, 48, 8, 8])
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels - in_channels, 1)
        self.conv3_1 = nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = torch.cat([x, x1], axis=1)

        x2 = self.relu(x)
        x2 = self.conv3_1(x2)
        x2 = self.relu(x)
        x2 = self.conv3_2(x2)

        out = x1 + x2
        return out
