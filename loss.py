import sys

sys.path.append("components")

import torch
import torch.nn as nn

from components.Generator import Generator
from components.TemporalDiscriminator import TemporalDiscriminator
from components.SpatialDiscriminator import SpatialDiscriminator


N, H, W = 18, 256, 256
forecast_steps = 18

gen = Generator(forecast_steps)
temp_dis = TemporalDiscriminator()
spa_dis = SpatialDiscriminator()

X = torch.zeros((2, 4, 1, 256, 256))
obs = torch.zeros((2, 18, 1, 256, 256))

pred = gen(X)[:, :, None, :, :]
temp_pred, temp_obs = temp_dis(pred), temp_dis(obs)
spa_pred, spa_obs = spa_dis(pred), spa_dis(obs)

print(pred.shape, temp_pred, spa_pred)


def regularization(pred, obs, N=N, H=H, W=W):
    if pred.ndim == 4:
        pred = pred[:, :, None, :, :]
    w = lambda y: max(y + 1, 24)
    l = (pred - obs) * (obs.apply_(w))
    lr = l.mean() / (N * H * W)
    return lr


def G_loss(pred, obs, l=20, N=N, H=H, W=W):
    return (
        temp_pred.mean()
        + spa_pred.mean()
        - l * regularization(pred, obs, N=N, H=H, W=W)
    )


def T_loss(temp_pred, temp_obs):
    return (nn.ReLU(1 - temp_obs) + nn.ReLU(1 + temp_pred)).mean()


def S_loss(spa_pred, spa_obs):
    return (nn.ReLU(1 - spa_obs) + nn.ReLU(1 + spa_pred)).mean()
