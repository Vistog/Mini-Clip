from __future__ import annotations

import torch
import torch.nn as nn

from mini_clip.registry import register


@register("projection", "linear")
class LinearProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0, **kwargs):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.drop = nn.Dropout(p=float(dropout)) if dropout and float(dropout) > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.drop(x))


@register("projection", "mlp")
class MLPProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 1024, dropout: float = 0.0, **kwargs):
        super().__init__()
        p = float(dropout)
        self.net = nn.Sequential(
            nn.Linear(in_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(p=p) if p > 0 else nn.Identity(),
            nn.Linear(int(hidden_dim), out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
