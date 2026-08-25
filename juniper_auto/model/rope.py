"""Rotary positional encoding.

Frozen v0.1 parameters: theta=100000, rotary_dim=64, rotary_fraction=1.0
(so for the official architectures the rotary dimension equals the full
head dimension). Applied after QK-Norm, per
`attention.qk_norm_placement: before_rope`. No learned parameters.

`inv_freq` is cached as a non-persistent buffer (so it moves with
`.to(device)` / `.to(dtype)` calls on the parent module but is never saved
into a checkpoint, since it is a deterministic function of `dim`/`theta`,
not learned state). cos/sin are recomputed from `position_ids` on every
forward call rather than cached by sequence length, which avoids a stale
cache silently serving the wrong device/dtype/length after a `.to(...)`
call or a resume onto different hardware.
"""

from __future__ import annotations

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 100000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE dim must be even, got {dim}")
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """position_ids: [batch, seq_len] integer positions.

        Returns (cos, sin), each [batch, seq_len, dim], in float32. Callers
        cast to the working activation dtype after the rotation is applied
        (or before, at the caller's discretion) -- this function itself
        never silently downcasts the trig computation.
        """
        inv_freq = self.inv_freq.to(device=position_ids.device)
        freqs = position_ids.to(torch.float32)[..., None] * inv_freq[None, None, :]
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """q, k: [batch, n_heads, seq_len, head_dim]. cos, sin: [batch, seq_len, rotary_dim].

    Only the first `rotary_dim` channels of each head are rotated; any
    remaining channels (when rotary_dim < head_dim) pass through unchanged.
    For the frozen v0.1 architectures rotary_dim == head_dim, so the whole
    head is rotated.
    """
    head_dim = q.shape[-1]
    dtype = q.dtype
    cos = cos.to(dtype).unsqueeze(1)
    sin = sin.to(dtype).unsqueeze(1)

    if rotary_dim == head_dim:
        q_rot = q * cos + rotate_half(q) * sin
        k_rot = k * cos + rotate_half(k) * sin
        return q_rot, k_rot

    q_rot_part, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot_part, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_rot_part = q_rot_part * cos + rotate_half(q_rot_part) * sin
    k_rot_part = k_rot_part * cos + rotate_half(k_rot_part) * sin
    return torch.cat([q_rot_part, q_pass], dim=-1), torch.cat([k_rot_part, k_pass], dim=-1)
