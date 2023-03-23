import torch.nn as nn
import torch.nn.functional as F


class GLU(nn.Module):
    def __init__(self, dim_in, dim_out, activation):
        super().__init__()
        self.act = activation
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * self.act(gate)


class ReluSquared(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2
