"""RMSNorm, used both as the block pre-norm and as per-head QK-Norm.

Per docs/architecture/precision-policy.md and the frozen
`normalization`/`attention` config sections: the squared-mean reduction
always runs in FP32 regardless of the input activation dtype, and the
normalized value is cast back to the input dtype before the learnable scale
is applied (matching the established RMSNorm reference behavior this
implementation is tested against). QK-Norm reuses this same module: applied
to a tensor shaped `[..., n_heads, head_dim]`, the reduction is over the
last (head_dim) axis only, so each head is normalized independently even
though the scale parameter (shape `[head_dim]`) is shared across all heads
via ordinary broadcasting -- this is what keeps QK-Norm at `head_dim`
parameters per projection instead of `n_heads * head_dim`.
"""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x_fp32 * torch.rsqrt(variance + self.eps)
        # The reduction above runs in FP32 regardless of input dtype; the
        # result is explicitly re-cast to the input activation dtype after
        # scaling, rather than left to widen via FP32-weight x FP16-tensor
        # type promotion (which some reference implementations allow).
        return (self.weight * x_normed.to(input_dtype)).to(input_dtype)
