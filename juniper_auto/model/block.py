"""Pre-Norm transformer blocks.

    x = x + Attention(RMSNorm(x))
    x = x + FFN_or_MoE(RMSNorm(x))

Residual scale is 1.0; no ReZero, DeepNorm, or learned residual gates
(all disabled by the frozen `residual` config section). `DenseBlock` is
used for dense-anchor layers in the sparse architecture and for every
layer of the dense control. `MoEBlock` is used only for the sparse
architecture's MoE layers.
"""

from __future__ import annotations

import torch
from torch import nn

from juniper_auto.config.schema import ArchitectureConfig
from juniper_auto.model.attention import GroupedQueryAttention
from juniper_auto.model.ffn import SwiGLU
from juniper_auto.model.moe import MoELayer
from juniper_auto.model.norm import RMSNorm


def _validate_block_assumptions(cfg: ArchitectureConfig) -> None:
    """Both block classes below hard-code Pre-Norm placement, unconditional
    attention/FFN norms, and plain additive residuals with no ReZero/
    DeepNorm/learned gates -- these checks make that hard-coding an
    explicit, fail-loud assumption instead of a silent one, so a future
    config that actually requests a variant this code doesn't implement
    is rejected rather than silently mishandled."""
    r = cfg.residual
    if r.kind != "additive":
        raise ValueError(f"unsupported residual.kind: {r.kind!r} (only 'additive' is implemented)")
    if r.rezero or r.deepnorm or r.learned_gates:
        raise ValueError("ReZero/DeepNorm/learned residual gates are not implemented by DenseBlock/MoEBlock")

    n = cfg.normalization
    if n.kind != "rmsnorm":
        raise ValueError(f"unsupported normalization.kind: {n.kind!r} (only 'rmsnorm' is implemented)")
    if n.placement != "pre_norm":
        raise ValueError(f"unsupported normalization.placement: {n.placement!r} (only 'pre_norm' is implemented)")
    if not n.attention_norm or not n.ffn_or_moe_norm:
        raise ValueError("normalization.attention_norm and ffn_or_moe_norm must both be true (both are applied unconditionally)")


class DenseBlock(nn.Module):
    def __init__(self, cfg: ArchitectureConfig):
        super().__init__()
        _validate_block_assumptions(cfg)
        self.residual_scale = cfg.residual.scale
        self.attention_norm = RMSNorm(cfg.core.d_model, eps=cfg.normalization.epsilon)
        self.attention = GroupedQueryAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.core.d_model, eps=cfg.normalization.epsilon)
        self.ffn = SwiGLU(cfg.core.d_model, cfg.dense_ffn.dim, bias=cfg.dense_ffn.bias)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        key_valid_mask: torch.Tensor | None,
        return_diagnostics: bool = False,
    ):
        x = x + self.residual_scale * self.attention(self.attention_norm(x), position_ids, key_valid_mask)
        x = x + self.residual_scale * self.ffn(self.ffn_norm(x))
        return x, None, None, None


class MoEBlock(nn.Module):
    def __init__(self, cfg: ArchitectureConfig):
        super().__init__()
        _validate_block_assumptions(cfg)
        self.residual_scale = cfg.residual.scale
        self.attention_norm = RMSNorm(cfg.core.d_model, eps=cfg.normalization.epsilon)
        self.attention = GroupedQueryAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.core.d_model, eps=cfg.normalization.epsilon)
        self.moe = MoELayer(cfg)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        key_valid_mask: torch.Tensor | None,
        return_diagnostics: bool = False,
    ):
        x = x + self.residual_scale * self.attention(self.attention_norm(x), position_ids, key_valid_mask)
        moe_out, load_balance_raw, router_z_raw, diagnostics = self.moe(
            self.ffn_norm(x), valid_mask=key_valid_mask, return_diagnostics=return_diagnostics
        )
        x = x + self.residual_scale * moe_out
        return x, load_balance_raw, router_z_raw, diagnostics
