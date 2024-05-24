import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm


class D3Block(nn.Module):
    """
    D3Block implementation from https://arxiv.org/abs/2104.00954

    Args:
        `in_channels`:`int`
            number of input channels
        `out_channels`:`int`
            number of output channels
        `temporal_first`:`bool`
            if`True`, the temporal dimension is before the channel dimension (N, T, C, H, W).
            Default:`True`
        `relu_first`:`bool`
            apply a ReLU activation before the first 3x3 Conv layer. Default:`True`

    Shape:
        - Input: (N, T, C_in, W, H) or (N, C_in, T, W, H)
        - Output: (N, T//2, C_out, W//2, H//2) or (N, C_out, T//2, W//2, H//2)

    Examples:

    >>> input = torch.zeros((5, 22, 4, 64, 64))
    >>> output = D3Block(4, 48)(input)
    >>> output.shape
    torch.Size([5, 11, 48, 32, 32])
    >>> input = torch.zeros((5, 4, 22, 64, 64))
    >>> output = D3Block(4, 48, temporal_first=False)(input)
    >>> output.shape
    torch.Size([5, 48, 11, 32, 32])
    """

    def __init__(self, in_channels, out_channels, temporal_first=True, relu_first=True):
        super().__init__()
        self.temporal_first = temporal_first
        self.relu3_1 = F.relu if relu_first else nn.Identity()
        self.relu3_2 = F.relu
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv3_1 = spectral_norm(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
        )
        self.conv3_2 = spectral_norm(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        )
        self.avg_pooling1 = nn.AvgPool3d(kernel_size=2, stride=2)
        self.avg_pooling2 = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        if self.temporal_first:
            x = x.permute(0, 2, 1, 3, 4)  # Returns (N, C, T, H, W_])

        x1 = self.conv1(x)
        x1 = self.avg_pooling1(x1)

        x2 = self.relu3_1(x)
        x2 = self.conv3_1(x)
        x2 = self.relu3_2(x2)
        x2 = self.conv3_2(x2)
        x2 = self.avg_pooling2(x2)

        out = x1 + x2
        if self.temporal_first:
            out = out.permute(0, 2, 1, 3, 4)  # Returns (N, T, C, H, W)
        return out
