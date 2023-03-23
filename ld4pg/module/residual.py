import torch
import torch.nn as nn
from einops import rearrange

from ld4pg.utils.utils import init_zero_, exists


class ScaleShift(nn.Module):
    def __init__(self, time_emb_dim, dim_out):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        )
        init_zero_(self.time_mlp[-1])

    def forward(self, x, time_emb):
        scale, shift = self.time_mlp(time_emb).chunk(2, dim=2)
        x = x * (scale + 1) + shift
        return x


class TimeConditionedResidual(nn.Module):
    def __init__(self, time_emb_dim, dim_out):
        super().__init__()
        self.scale_shift = ScaleShift(time_emb_dim, dim_out)

    def forward(self, x, residual, time_emb):
        return self.scale_shift(x, time_emb) + residual


class Residual(nn.Module):
    def __init__(self, dim, scale_residual=False, scale_residual_constant=1.):
        super().__init__()
        self.residual_scale = nn.Parameter(torch.ones(dim)) if scale_residual else None
        self.scale_residual_constant = scale_residual_constant

    def forward(self, x, residual):
        if exists(self.residual_scale):
            residual = residual * self.residual_scale

        if self.scale_residual_constant != 1:
            residual = residual * self.scale_residual_constant

        return x + residual


class GRUGatingResidual(nn.Module):
    def __init__(self, dim, scale_residual=False):
        super().__init__()
        self.gru = nn.GRUCell(dim, dim)
        self.residual_scale = nn.Parameter(torch.ones(dim)) if scale_residual else None

    def forward(self, x, residual):
        if exists(self.residual_scale):
            residual = residual * self.residual_scale

        gated_output = self.gru(
            rearrange(x, 'b n d -> (b n) d'),
            rearrange(residual, 'b n d -> (b n) d')
        )

        return gated_output.reshape_as(x)
