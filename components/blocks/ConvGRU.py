import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm


class ConvGRU(nn.Module):
    """
    ConvGRU implementation from https://arxiv.org/abs/2104.00954

    Args:
        `in_channels`: `int`
            number of input channels
        `hidden_channels`: `int`
            number of hidden channels

    Shape:
        - Input: (N, C_in, W, H), (N, C_h, W, H)
        - Output: (N, C_h, W, H)

    Examples:

    >>> input = torch.zeros((5, 768, 8, 8))
    >>> h0 = torch.zeros((5, 384, 8, 8))
    >>> h = ConvGRU(768, 384)(input, h0)
    >>> h.shape
    torch.Size([5, 384, 8, 8])
    """

    def __init__(self, in_channels, hidden_channels, out_channels=None):
        super().__init__()
        input_size = in_channels + hidden_channels
        if not out_channels:
            out_channels = hidden_channels
        self.reset_gate = spectral_norm(
            nn.Conv2d(input_size, out_channels, 3, padding=1)
        )
        self.update_gate = spectral_norm(
            nn.Conv2d(input_size, out_channels, 3, padding=1)
        )
        self.out_gate = spectral_norm(nn.Conv2d(input_size, out_channels, 3, padding=1))

    def forward(self, x, h0):
        stacked = torch.cat([x, h0], dim=1)

        update = torch.sigmoid(self.update_gate(stacked))
        reset = torch.sigmoid(self.reset_gate(stacked))

        out = torch.tanh(self.out_gate(torch.cat([x, h0*reset], dim=1))) 
        h = h0 * (1-update) + out * update 
        return h
