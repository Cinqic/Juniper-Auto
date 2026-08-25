"""Sparse MoE layer: router + top-2 dropless dispatch + always-active
ungated shared expert.

    MoE(x) = SharedExpert(x) + alpha_1 * RoutedExpert_a(x) + alpha_2 * RoutedExpert_b(x)

with alpha_1 + alpha_2 == 1 after renormalizing the top-2 router
probabilities. No capacity factor, no token dropping, no averaging
divisor, no gate on the shared expert, no router/expert bias.

This is a correctness-first *reference* dispatch: it loops over the 8
routed experts and masks/gathers per expert, rather than a batched/kernel
implementation, per the Phase 1 instruction to prioritize an inspectable
implementation over a Phase-2-style optimized kernel. Every token
(including padding) is routed and produces a finite output, so the block
output stays shape-stable; only the *statistics* used for the auxiliary
losses are restricted to valid (non-padding) tokens, via `valid_mask`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from juniper_auto.config.schema import ArchitectureConfig
from juniper_auto.model.ffn import SwiGLU
from juniper_auto.model.losses import compute_load_balance_loss_raw, compute_router_z_loss_raw


@dataclass
class MoEDiagnostics:
    router_logits: torch.Tensor  # [n_tokens, n_experts] fp32
    router_probs: torch.Tensor  # [n_tokens, n_experts] fp32
    topk_idx: torch.Tensor  # [n_tokens, top_k] long
    topk_weights: torch.Tensor  # [n_tokens, top_k] fp32, renormalized, sums to 1 per valid token
    valid_mask: torch.Tensor  # [n_tokens] bool
    assignment_counts_per_expert: torch.Tensor  # [n_experts], valid tokens only


class MoELayer(nn.Module):
    def __init__(self, cfg: ArchitectureConfig):
        super().__init__()
        moe_cfg = cfg.moe
        if moe_cfg is None:
            raise ValueError("MoELayer requires a non-null moe config section")
        if moe_cfg.shared_expert_gated:
            raise ValueError("MoELayer implementation requires an ungated shared expert")
        if moe_cfg.expert_output_combination != "sum":
            raise ValueError("MoELayer implementation requires sum expert-output combination")
        if not moe_cfg.dropless or moe_cfg.token_dropping_allowed:
            raise ValueError("MoELayer implementation requires dropless=True, token_dropping_allowed=False")

        self.n_routed_experts = moe_cfg.n_routed_experts
        self.n_shared_experts = moe_cfg.n_shared_experts
        self.top_k = moe_cfg.top_k
        self.renormalize = moe_cfg.renormalize_top_k_weights
        self.load_balance_coefficient = moe_cfg.load_balance_loss_coefficient
        self.router_z_coefficient = moe_cfg.router_z_loss_coefficient

        self.router = nn.Linear(moe_cfg.router_input_dim, moe_cfg.router_output_dim, bias=moe_cfg.router_bias)

        self.routed_experts = nn.ModuleList(
            [
                SwiGLU(cfg.core.d_model, moe_cfg.expert_ffn_dim, bias=moe_cfg.expert_bias)
                for _ in range(moe_cfg.n_routed_experts)
            ]
        )
        if self.n_shared_experts != 1:
            raise ValueError(
                f"MoELayer implementation assumes exactly one shared expert, got {self.n_shared_experts}"
            )
        self.shared_expert = SwiGLU(cfg.core.d_model, moe_cfg.expert_ffn_dim, bias=moe_cfg.expert_bias)

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, MoEDiagnostics | None]:
        """x: [batch, seq_len, d_model]. valid_mask: [batch, seq_len] bool/int,
        True/1 for real tokens (defaults to all-valid if omitted).

        Returns (output [batch, seq_len, d_model], load_balance_loss_raw,
        router_z_loss_raw, diagnostics_or_None).
        """
        batch, seq_len, d_model = x.shape
        flat_x = x.reshape(-1, d_model)
        n_tokens = flat_x.shape[0]

        if valid_mask is None:
            flat_valid = torch.ones(n_tokens, dtype=torch.bool, device=x.device)
        else:
            flat_valid = valid_mask.reshape(-1).to(torch.bool)

        router_bias = self.router.bias.to(torch.float32) if self.router.bias is not None else None
        with torch.autocast(device_type=x.device.type, enabled=False):
            router_logits = F.linear(flat_x.to(torch.float32), self.router.weight.to(torch.float32), router_bias)
            router_probs = F.softmax(router_logits, dim=-1)

        topk_probs, topk_idx = torch.topk(router_probs, k=self.top_k, dim=-1)
        if self.renormalize:
            denom = topk_probs.sum(dim=-1, keepdim=True).clamp_min(1e-20)
            topk_weights = topk_probs / denom
        else:
            topk_weights = topk_probs

        output = self.shared_expert(flat_x)

        # Weight cast is deferred until each expert_out is actually
        # produced (rather than pre-cast to flat_x.dtype up front): under
        # autocast, expert Linear layers may execute in a lower-precision
        # dtype than flat_x itself, and index_add_ requires the source
        # tensor's dtype to exactly match `output`'s -- casting per-slot
        # against expert_out.dtype keeps this correct regardless of what
        # autocast decided for that op.
        for expert_id, expert in enumerate(self.routed_experts):
            for slot in range(self.top_k):
                slot_mask = topk_idx[:, slot] == expert_id
                if not torch.any(slot_mask):
                    continue
                expert_out = expert(flat_x[slot_mask])
                weight = topk_weights[slot_mask, slot : slot + 1].to(expert_out.dtype)
                output = output.index_add(
                    0, slot_mask.nonzero(as_tuple=True)[0], (weight * expert_out).to(output.dtype)
                )

        load_balance_raw = compute_load_balance_loss_raw(
            router_probs, topk_idx, flat_valid, self.n_routed_experts, self.top_k
        )
        router_z_raw = compute_router_z_loss_raw(router_logits, flat_valid)

        diagnostics = None
        if return_diagnostics:
            valid_topk_idx = topk_idx[flat_valid]
            counts = torch.zeros(self.n_routed_experts, dtype=torch.long, device=x.device)
            if valid_topk_idx.numel() > 0:
                counts.scatter_add_(
                    0, valid_topk_idx.reshape(-1), torch.ones_like(valid_topk_idx.reshape(-1))
                )
            diagnostics = MoEDiagnostics(
                router_logits=router_logits,
                router_probs=router_probs,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                valid_mask=flat_valid,
                assignment_counts_per_expert=counts,
            )

        return output.view(batch, seq_len, d_model), load_balance_raw, router_z_raw, diagnostics
