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


class DenseBlock(nn.Module):
    def __init__(self, cfg: ArchitectureConfig):
        super().__init__()
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
        x = x + self.attention(self.attention_norm(x), position_ids, key_valid_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x, None, None, None


class MoEBlock(nn.Module):
    def __init__(self, cfg: ArchitectureConfig):
        super().__init__()
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
        x = x + self.attention(self.attention_norm(x), position_ids, key_valid_mask)
        moe_out, load_balance_raw, router_z_raw, diagnostics = self.moe(
            self.ffn_norm(x), valid_mask=key_valid_mask, return_diagnostics=return_diagnostics
        )
        x = x + moe_out
        return x, load_balance_raw, router_z_raw, diagnostics
