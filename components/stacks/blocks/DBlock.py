import torch.nn as nn
import torch.nn.functional as F


class DBlock(nn.Module):
    """
    DBlock implementation from https://arxiv.org/abs/2104.00954

    Args:
        `in_channels`:`int`
            number of input channels
        `out_channels`:`int`
            number of output channels
        `relu_first`:`bool`
            apply a ReLU activation before the first 3x3 Conv layer. Default:`True`
        `downsampling`:`bool`
            apply a ReLU down-sampling step by a nn.AvgPool2d(2, 2). Default:`True`

    Shape:
        - Input: (N, C_in, W_in, H_in)
        - Output: (N, C_out, W_out, H_out)

    Examples:

    >>> input = torch.zeros((5, 4, 8, 8))
    >>> output = DBlock(4, 48)(input)
    >>> output.shape
    torch.Size([5, 48, 4, 4])

    >>> input = torch.zeros((5, 4, 8, 8))
    >>> output = DBlock(4, 48, downsampling=False)(input)
    >>> output.shape
    torch.Size([5, 48, 8, 8])
    """

    def __init__(self, in_channels, out_channels, relu_first=True, downsampling=True):
        super().__init__()
        self.relu3_1 = F.relu if relu_first else nn.Identity()
        self.relu3_2 = F.relu
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv3_1 = nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.ds1 = nn.AvgPool2d(2, 2) if downsampling else nn.Identity()
        self.ds2 = nn.AvgPool2d(2, 2) if downsampling else nn.Identity()

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.ds1(x1)

        x2 = self.relu3_1(x)
        x2 = self.conv3_1(x2)
        x2 = self.relu3_2(x2)
        x2 = self.conv3_2(x2)
        x2 = self.ds2(x2)

        out = x1 + x2
        return out
