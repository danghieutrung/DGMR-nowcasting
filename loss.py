import torch
import torch.nn.functional as F

from components.Generator import Generator
from components.TemporalDiscriminator import TemporalDiscriminator
from components.SpatialDiscriminator import SpatialDiscriminator


N, H, W = 18, 256, 256
forecast_steps = 18

gen = Generator(forecast_steps)
temp_dis = TemporalDiscriminator()
spa_dis = SpatialDiscriminator()


def regularization(pred, obs):
    """
    Grid cell regularization term implementation from https://arxiv.org/abs/2104.00954

    Args:
        `pred`:`torch.Tensor`
            predictions, outputs from the Generator
        `obs`:`torch.Tensor`
            actual observation corresponding to `pred`

    Shape:
        - pred: (N, T, 1, W, H) or (N, T, W, H)
        - obs: (N, `forecast_steps`, 1, W, H)

    Examples:

    >>> input = torch.zeros((5, 12, 1, 256, 256))
    >>> obs = torch.zeros((5, 12, 1, 256, 256))
    >>> pred = Generator(18)(input)
    >>> regularization(obs, preds)
    """

    lr = torch.mean(torch.abs(pred - obs) * torch.clamp(pred + 1, 24))
    return lr


def G_loss(pred, obs, T_pred, S_pred, lambda_=20):
    return torch.abs(
        T_pred.mean() + S_pred.mean() - lambda_ * regularization(pred, obs)
    )


def T_loss(temp_pred, temp_obs):
    return (F.relu(1 - temp_obs) + F.relu(1 + temp_pred)).mean()


def S_loss(spa_pred, spa_obs):
    return (F.relu(1 - spa_obs) + F.relu(1 + spa_pred)).mean()
