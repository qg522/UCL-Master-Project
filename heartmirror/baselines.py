"""Executable ED Repeat and an explicitly specified direct regression baseline."""
import torch
from torch import nn


class DirectRegression(nn.Module):
    # Paper does not specify its direct regressor. This is an engineering baseline.
    def __init__(self, cfg):
        super().__init__()
        self.phases = cfg.phases
        self.net = nn.Sequential(nn.Conv3d(1, 16, 3, padding=1), nn.GELU(),
                                 nn.Conv3d(16, 16, 3, padding=1), nn.GELU(),
                                 nn.Conv3d(16, cfg.phases-1, 1))

    def forward(self, first):
        return torch.cat([first, self.net(first).sigmoid()], dim=1)
