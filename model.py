import torch
import torch.nn as nn


from components.Generator import Generator
from components.TemporalDiscriminator import TemporalDiscriminator
from components.SpatialDiscriminator import SpatialDiscriminator


class DGMRNowcasting(nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = Generator(18)
        self.temporal_discriminator = TemporalDiscriminator()
        self.spatial_discriminator = SpatialDiscriminator()

    def get_generator(self):
        return self.generator

    def get_temporal_discriminator(self):
        return self.temporal_discriminator

    def get_spatial_discriminator(self):
        return self.spatial_discriminator

    def forward(self, x, obs=None):
        outputs = self.generator(x)
        merged_pred = torch.cat((x, outputs), dim=1)
        T_pred = self.temporal_discriminator(merged_pred)
        S_pred = self.spatial_discriminator(merged_pred)

        if obs != None:
            merged_obs = torch.cat((x, obs), dim=1)
            T_obs = self.temporal_discriminator(merged_obs)
            S_obs = self.spatial_discriminator(merged_obs)
            return outputs, T_pred, S_pred, T_obs, S_obs

        return outputs, T_pred, S_pred
