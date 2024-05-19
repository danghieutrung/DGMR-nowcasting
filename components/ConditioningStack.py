import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

from blocks.DBlock import DBlock
from blocks.S2D import S2D


class ConditioningStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.s2d = S2D(2)
        self.d1 = DBlock(4, 24)
        self.d2 = DBlock(24, 48)
        self.d3 = DBlock(48, 96)
        self.d4 = DBlock(96, 192)
        self.conv3_1 = spectral_norm(nn.Conv2d(96, 48, 3, stride=1, padding=1))
        self.conv3_2 = spectral_norm(nn.Conv2d(192, 96, 3, stride=1, padding=1))
        self.conv3_3 = spectral_norm(nn.Conv2d(384, 192, 3, stride=1, padding=1))
        self.conv3_4 = spectral_norm(nn.Conv2d(768, 384, 3, stride=1, padding=1))
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, batch):
        out = []
        B, T, C, H, W = batch.shape
        for t in range(T):
            x = x[:, t, :, :, :].squeeze()  # B, C, H, W
            x = self.s2d(x)
            x = self.d1(x)
            #TODO