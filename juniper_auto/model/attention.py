"""Causal grouped-query attention with per-head QK-Norm and RoPE.

Frozen v0.1 shape: 8 query heads, 2 KV heads (4 query heads per KV head),
head_dim=64, d_model=512, attention_scale=0.125, no bias, no sliding
window, full causal masking. QK-Norm is applied per head, before RoPE
(`attention.qk_norm_placement: before_rope`).

Uses `torch.nn.functional.scaled_dot_product_attention` with an explicit
boolean mask (never the `is_causal=True` shortcut) so causal masking and
padding masking are unified into one code path that is directly testable,
rather than relying on SDPA's internal causal-mask construction agreeing
with a separately-built padding mask.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from juniper_auto.config.schema import ArchitectureConfig
from juniper_auto.model.norm import RMSNorm
from juniper_auto.model.rope import RotaryEmbedding, apply_rotary_pos_emb


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """x: [batch, n_kv_heads, seq_len, head_dim] -> [batch, n_kv_heads * n_rep, seq_len, head_dim].

    Each KV head's slice is repeated contiguously so KV head `h` serves
    query heads `[h*n_rep, (h+1)*n_rep)` -- an explicit, testable mapping
    rather than an implicit reshape trick.
    """
    if n_rep == 1:
        return x
    batch, n_kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(batch, n_kv_heads, n_rep, seq_len, head_dim)
    return x.reshape(batch, n_kv_heads * n_rep, seq_len, head_dim)


def build_attention_mask(
    seq_len: int,
    key_valid_mask: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor:
    """Returns a boolean attend-mask [batch_or_1, 1, seq_len, seq_len] where
    True means "query i may attend to key j".

    Combines strict causal masking (j <= i) with key-side padding masking
    (key_valid_mask: [batch, seq_len], 1/True = real token). A query row
    that ends up with zero allowed keys (a padded query position whose own
    padded key is also masked out) is given a forced self-attend fallback
    so softmax never divides by zero -- the output at that row is discarded
    by the caller (padding contributes to no loss and no downstream
    computation), only its *finiteness* matters.
    """
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    causal = causal.view(1, 1, seq_len, seq_len)

    if key_valid_mask is None:
        return causal.expand(1, 1, seq_len, seq_len)

    key_valid = key_valid_mask.to(torch.bool).view(-1, 1, 1, seq_len)
    combined = causal & key_valid

    no_valid_key = ~combined.any(dim=-1, keepdim=True)
    eye = torch.eye(seq_len, dtype=torch.bool, device=device).view(1, 1, seq_len, seq_len)
    combined = combined | (no_valid_key & eye)
    return combined


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg: ArchitectureConfig):
        super().__init__()
        a = cfg.attention
        if a.n_query_heads % a.n_kv_heads != 0:
            raise ValueError("n_query_heads must be divisible by n_kv_heads")
        if not a.causal:
            raise ValueError("attention.causal=false is not implemented (this module always builds a causal mask)")
        if a.sliding_window is not None:
            raise ValueError(
                f"attention.sliding_window={a.sliding_window!r} is not implemented (full causal attention only)"
            )
        if cfg.position_encoding.kind != "rope":
            raise ValueError(f"unsupported position_encoding.kind: {cfg.position_encoding.kind!r} (only 'rope' is implemented)")

        self.n_query_heads = a.n_query_heads
        self.n_kv_heads = a.n_kv_heads
        self.head_dim = a.head_dim
        self.n_rep = a.n_query_heads // a.n_kv_heads
        self.scale = a.attention_scale
        self.rotary_dim = cfg.position_encoding.rotary_dim

        d_model = cfg.core.d_model
        q_dim = a.n_query_heads * a.head_dim
        kv_dim = a.n_kv_heads * a.head_dim

        self.q_proj = nn.Linear(d_model, q_dim, bias=a.attention_bias)
        self.k_proj = nn.Linear(d_model, kv_dim, bias=a.attention_bias)
        self.v_proj = nn.Linear(d_model, kv_dim, bias=a.attention_bias)
        self.o_proj = nn.Linear(q_dim, d_model, bias=a.attention_bias)

        self.qk_norm = a.qk_norm
        if a.qk_norm:
            if a.qk_norm_placement != "before_rope" or a.qk_norm_kind != "per_head_rmsnorm":
                raise ValueError(
                    f"unsupported QK-Norm configuration: placement={a.qk_norm_placement!r}, "
                    f"kind={a.qk_norm_kind!r}"
                )
            self.q_norm = RMSNorm(a.head_dim, eps=cfg.normalization.epsilon)
            self.k_norm = RMSNorm(a.head_dim, eps=cfg.normalization.epsilon)
        else:
            self.q_norm = None
            self.k_norm = None

        self.rope = RotaryEmbedding(
            dim=cfg.position_encoding.rotary_dim,
            theta=cfg.position_encoding.theta,
            initial_scaling=cfg.position_encoding.initial_scaling,
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        key_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must have shape [batch, seq_len, d_model], got {tuple(x.shape)}")
        batch, seq_len, _ = x.shape
        if position_ids.shape != (batch, seq_len):
            raise ValueError(
                f"position_ids must have shape {(batch, seq_len)}, got {tuple(position_ids.shape)}"
            )
        if key_valid_mask is not None and key_valid_mask.shape != (batch, seq_len):
            raise ValueError(
                f"key_valid_mask must have shape {(batch, seq_len)}, got {tuple(key_valid_mask.shape)}"
            )

        q = self.q_proj(x).view(batch, seq_len, self.n_query_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.transpose(1, 2)  # [batch, n_query_heads, seq_len, head_dim]
        k = k.transpose(1, 2)  # [batch, n_kv_heads, seq_len, head_dim]
        v = v.transpose(1, 2)

        cos, sin = self.rope(position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, self.rotary_dim)

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        attn_mask = build_attention_mask(seq_len, key_valid_mask, device=x.device)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=False, scale=self.scale
        )
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.n_query_heads * self.head_dim)
        return self.o_proj(out)
