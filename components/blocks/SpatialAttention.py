import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    """
    Spatial Attention block implementation from https://arxiv.org/abs/2104.00954
    Borrowed from hyungting/DGMR-pytorch: https://github.com/hyungting/DGMR-pytorch/blob/master/DGMR/dgmr_layers/LatentStack.py

    Args:
        `in_channels`: `int`
            number of input channels
        `out_channels`: `int`
            number of output channels
        `ratio_kq`: `int`
            Default: 8
        `ratio_v`: `int`
            Default: 8

    Shape:
        - Input: (N, C, H, W)
        - Output: (N, C, H, W)

    Example:

    >>> input = torch.zeros((5, 192, 8, 8))
    >>> output = SpatialAttention(192, 192)(input)
    >>> output.shape
    torch.Size([5, 192, 8, 8])
    """

    def __init__(self, in_channels, out_channels, ratio_kq=8, ratio_v=8):
        super().__init__()
        self.ratio_kq = ratio_kq
        self.ratio_v = ratio_v
        self.conv_q = nn.Conv2d(
            in_channels, out_channels // ratio_kq, 1, 1, 0, bias=False
        )
        self.conv_k = nn.Conv2d(
            in_channels, out_channels // ratio_kq, 1, 1, 0, bias=False
        )
        self.conv_v = nn.Conv2d(
            in_channels, out_channels // ratio_v, 1, 1, 0, bias=False
        )
        self.conv_out = nn.Conv2d(
            out_channels // ratio_v, out_channels, 1, 1, 0, bias=False
        )

    def einsum(self, q, k, v):
        k = k.view(k.shape[0], k.shape[1], -1)  # B, C, H*W
        v = v.view(v.shape[0], v.shape[1], -1)  # B, C, H*W
        beta = torch.einsum("bchw, bcL->bLhw", q, k)
        beta = torch.softmax(beta, dim=1)
        out = torch.einsum("bLhw, bcL->bchw", beta, v)
        return out

    def forward(self, x):
        q = self.conv_q(x)
        k = self.conv_k(x)
        v = self.conv_v(x)

        out = self.einsum(q, k, v)
        out = self.conv_out(out)
        return x + out
