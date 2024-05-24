import sys

sys.path.append("components")


import torch.nn.functional as F

from components.Generator import Generator
from components.TemporalDiscriminator import TemporalDiscriminator
from components.SpatialDiscriminator import SpatialDiscriminator


N, H, W = 18, 256, 256
forecast_steps = 18

gen = Generator(forecast_steps)
temp_dis = TemporalDiscriminator()
spa_dis = SpatialDiscriminator()


def regularization(pred, obs, N=N, H=H, W=W):
    """
    Grid cell regularization term implementation from https://arxiv.org/abs/2104.00954

    Args:
        `pred`:`torch.Tensor`
            predictions, outputs from the Generator
        `obs`:`torch.Tensor`
            actual observation corresponding to `pred`
        `N`:`int`
            lead time. Default: 18
        `H`:`int`
            image height. Default: 256
        `W`: `int`
            image width. Default: 256

    Shape:
        - pred: (N, T, 1, W, H) or (N, T, W, H)
        - obs: (N, `forecast_steps`, 1, W, H)

    Examples:

    >>> input = torch.zeros((5, 4, 1, 256, 256))
    >>> obs = torch.zeros((5, 4, 1, 256, 256))
    >>> pred = Generator(18)(input)
    >>> regularization(obs, preds)
    """

    if pred.ndim == 4:
        pred = pred[:, :, None, :, :]
    w = lambda y: max(y + 1, 24)
    l = (pred - obs) * (obs.apply_(w))
    lr = l.mean() / (N * H * W)
    return lr


def G_loss(pred, obs, temp_pred, spa_pred, l=20, N=N, H=H, W=W):
    return (
        temp_pred.mean()
        + spa_pred.mean()
        - l * regularization(pred, obs, N=N, H=H, W=W)
    )


def T_loss(temp_pred, temp_obs):
    return (F.relu(1 - temp_obs) + F.relu(1 + temp_pred)).mean()


def S_loss(spa_pred, spa_obs):
    return (F.relu(1 - spa_obs) + F.relu(1 + spa_pred)).mean()
