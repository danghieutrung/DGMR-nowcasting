import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm


class ConvGRU(nn.Module):
    """
    convGRU implementation from https://arxiv.org/abs/2104.00954

    Args:
        `in_channels`: `int`
            number of input channels
        `hidden_channels`: `int`
            number of hidden channels

    Shape:
        - Input: (N, C_in, W, H), (N, C_h, W, H)
        - Output: (N, C_h, W, H)

    Examples:

    >>> input = torch.zeros((5, 48, 8, 8))
    >>> hidden = torch.zeros((5, 96, 8, 8))
    >>> output = ConvGRU(48, 96)(input, hidden)
    >>> output.shape
    torch.Size([5, 96, 8, 8])
    """

    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        input_size = in_channels + hidden_channels
        output_size = hidden_channels
        self.reset_gate = spectral_norm(
            nn.Conv2d(input_size, output_size, 3, padding=1)
        )
        self.update_gate = spectral_norm(
            nn.Conv2d(input_size, output_size, 3, padding=1)
        )
        self.out_gate = spectral_norm(nn.Conv2d(input_size, output_size, 3, padding=1))

    def forward(self, x, prev_state):
        stacked = torch.cat((x, prev_state), dim=1)

        update = torch.sigmoid(self.update_gate(stacked))
        reset = torch.sigmoid(self.update_gate(stacked))
        out = torch.tanh(self.out_gate(torch.cat((x, prev_state * reset), dim=1)))

        new_state = prev_state * (1 - update) + out * update
        return new_state
