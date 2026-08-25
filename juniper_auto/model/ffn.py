"""SwiGLU feed-forward block, shared by the dense-anchor FFN and every MoE
expert (routed and shared) -- they differ only in `hidden_dim` and which
layers instantiate them, not in the equation.

    down_proj(SiLU(gate_proj(x)) * up_proj(x))

All three projections are bias-free per the frozen `dense_ffn`/`moe`
config sections.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
