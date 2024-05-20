import torch.nn as nn


class GBlock(nn.Module):
    """
    GBlock implementation from https://arxiv.org/abs/2104.00954

    Args:
        `in_channels`: `int`
            number of input channels
        `out_channels`: `int`
            number of output channels
        `upsampling`: `bool`
            apply a ReLU up-sampling step by a nn.Upsample(2, 2). Default: `True`

    Shape:
        - Input: (N, C_in, W_in, H_in)
        - Output: (N, C_out, W_out, H_out)

    Examples:

    >>> input = torch.zeros((5, 4, 8, 8))
    >>> output = DBlock(4, 48)(input)
    >>> output.shape
    torch.Size([5, 48, 16, 16])

    >>> input = torch.zeros((5, 4, 8, 8))
    >>> output = DBlock(4, 48, upsampling=False)(input)
    >>> output.shape
    torch.Size([5, 48, 8, 8])
    """

    def __init__(self, in_channels, out_channels, upsampling=True):
        super().__init__()
        self.us1 = (
            nn.Upsample(scale_factor=2, mode="nearest") if upsampling else nn.Identity()
        )
        self.us2 = (
            nn.Upsample(scale_factor=2, mode="nearest") if upsampling else nn.Identity()
        )
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, groups=out_channels)
        self.conv3_1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels)
        self.conv3_2 = nn.Conv2d(
            in_channels, out_channels, 3, 1, 1, groups=out_channels
        )

    def forward(self, x):
        x1 = self.us1(x)
        x1 = self.conv1(x1)

        x2 = self.bn(x)
        x2 = self.relu(x2)
        x2 = self.us2(x2)
        x2 = self.conv3_1(x2)
        x2 = self.bn(x2)
        x2 = self.relu(x2)
        x2 = self.conv3_2(x2)

        out = x1 + x2
        return out
