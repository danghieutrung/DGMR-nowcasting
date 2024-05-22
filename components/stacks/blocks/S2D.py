import torch
import torch.nn as nn


def s2d(x, scale_factor=2):
    B, C, H, W = x.size()
    unfolded_x = torch.nn.functional.unfold(x, scale_factor, stride=scale_factor)
    return unfolded_x.view(B, C * scale_factor**2, H // scale_factor, W // scale_factor)


class S2D(nn.Module):
    """
    S2D block implementation from https://arxiv.org/abs/2104.00954

    Args:
        `scale_factor`:`int`
            multiplier for spatial size. Default: 0.5

    Shape:
        - Input: (N, C, W, H) or (N, T, C, W, H)
        - Output: (N, C*4, W//2, H//2) or (N, T, C*4, W//2, H//2)

    Example:

    >>> input = torch.zeros((5, 22, 1, 128, 128))
    >>> output = S2D(0.5)(input)
    >>> output.shape
    torch.Size([5, 4, 22, 64, 64])
    >>> input = torch.zeros((5, 1, 128, 128))
    >>> output = S2D(0.5)(input)
    >>> output.shape
    torch.Size([5, 4, 64, 64])
    """

    def __init__(self, scale_factor=0.5):
        super().__init__()
        self.scale_factor = int(1 / scale_factor)

    def forward(self, x):
        if x.ndim == 5:
            N, T, C, H, W = x.shape
            x = x.reshape(N * T, C, H, W)
            x = s2d(x)
            x = x.reshape(
                N,
                T,
                C * self.scale_factor * self.scale_factor,
                H // self.scale_factor,
                W // self.scale_factor,
            )
        elif x.ndim == 4:
            x = s2d(x)
        else:
            raise RuntimeError(
                f"Expected 4D (non-temporal) or 5D (temporal) input, but got input of size: {list(x.shape)}"
            )
        return x
