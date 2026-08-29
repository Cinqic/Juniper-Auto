"""Sparse MoE layer: router + top-2 dropless dispatch + always-active
ungated shared expert.

    MoE(x) = SharedExpert(x) + alpha_1 * RoutedExpert_a(x) + alpha_2 * RoutedExpert_b(x)

with alpha_1 + alpha_2 == 1 after renormalizing the top-2 router
probabilities. No capacity factor, no token dropping, no averaging
divisor, no gate on the shared expert, no router/expert bias. Valid tokens
are dispatched droplessly; padding positions are excluded from both shared
and routed expert execution and receive a zero MoE contribution. Router
probabilities are still available for every flattened position in deep
diagnostics, while all aggregate statistics and losses use only valid
tokens via `valid_mask`.

This module is the MoELayer orchestrator: it owns the router/expert
parameters and config validation, and delegates router math to
`juniper_auto.model.routing`, dispatch to `juniper_auto.model.moe_dispatch`
(a `backend="reference"` correctness-first path, preserved bit-for-bit
identical to the approved Phase 1 implementation for the default no-ablation
call, and an `backend="optimized"` pure-PyTorch alternative -- see
`moe_dispatch.py` and tests/test_model_moe_dispatch.py for the equivalence
proof), instrumentation to `juniper_auto.model.moe_diagnostics`, and
evaluation-only ablation overrides to `juniper_auto.model.moe_ablations`.

`MoEDiagnostics` is re-exported here (rather than only living in
moe_diagnostics.py) because `juniper_auto.model.model` and existing Phase 1
tests import it as `juniper_auto.model.moe.MoEDiagnostics`.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from juniper_auto.config.schema import ArchitectureConfig
from juniper_auto.model.ffn import SwiGLU
from juniper_auto.model.losses import compute_load_balance_loss_raw, compute_router_z_loss_raw
from juniper_auto.model.moe_ablations import (
    MoEAblationConfig,
    is_router_override,
    resolve_dispatch_kwargs,
    resolve_router_override,
    should_disable_shared_expert,
    validate_ablation_for_layer,
)
from juniper_auto.model.moe_diagnostics import MoEDiagnostics, build_moe_diagnostics
from juniper_auto.model.moe_dispatch import DISPATCH_BACKENDS
from juniper_auto.model.routing import compute_router_logits_and_probs, select_topk

DispatchBackend = Literal["reference", "optimized"]

__all__ = ["MoELayer", "MoEDiagnostics", "MoEAblationConfig", "DispatchBackend"]


class MoELayer(nn.Module):
    def __init__(self, cfg: ArchitectureConfig, *, backend: DispatchBackend = "reference"):
        super().__init__()
        moe_cfg = cfg.moe
        if moe_cfg is None:
            raise ValueError("MoELayer requires a non-null moe config section")
        if moe_cfg.shared_expert_gated:
            raise ValueError("MoELayer implementation requires an ungated shared expert")
        if not moe_cfg.shared_expert_always_active:
            raise ValueError("MoELayer implementation requires shared_expert_always_active=True")
        if moe_cfg.expert_output_combination != "sum":
            raise ValueError("MoELayer implementation requires sum expert-output combination")
        if not moe_cfg.dropless or moe_cfg.token_dropping_allowed:
            raise ValueError("MoELayer implementation requires dropless=True, token_dropping_allowed=False")
        if moe_cfg.routing_kind != "token_choice":
            raise ValueError(
                f"unsupported moe.routing_kind: {moe_cfg.routing_kind!r} (only 'token_choice' is implemented)"
            )
        if moe_cfg.router_logits_dtype != "fp32" or moe_cfg.router_softmax_dtype != "fp32":
            raise ValueError(
                "MoELayer implementation always forces FP32 router logits/softmax regardless of ambient "
                f"precision -- router_logits_dtype={moe_cfg.router_logits_dtype!r}, "
                f"router_softmax_dtype={moe_cfg.router_softmax_dtype!r} would silently be overridden, so "
                "this is rejected instead of silently ignored"
            )
        if (
            moe_cfg.training_router_jitter_magnitude is not None
            or moe_cfg.evaluation_router_jitter
            or moe_cfg.inference_router_jitter
        ):
            raise ValueError("router jitter is not implemented by the Phase 1/2 reference MoELayer")
        if backend not in DISPATCH_BACKENDS:
            raise ValueError(f"unsupported backend: {backend!r} (expected one of {sorted(DISPATCH_BACKENDS)})")

        self.n_routed_experts = moe_cfg.n_routed_experts
        self.n_shared_experts = moe_cfg.n_shared_experts
        self.top_k = moe_cfg.top_k
        self.renormalize = moe_cfg.renormalize_top_k_weights
        self.load_balance_coefficient = moe_cfg.load_balance_loss_coefficient
        self.router_z_coefficient = moe_cfg.router_z_loss_coefficient
        self.backend: DispatchBackend = backend

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
        *,
        backend: DispatchBackend | None = None,
        ablation: MoEAblationConfig | None = None,
        return_trace: bool = False,
        max_trace_tokens: int | None = 4096,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, MoEDiagnostics | None]:
        """x: [batch, seq_len, d_model]. valid_mask: [batch, seq_len] bool/int,
        True/1 for real tokens (defaults to all-valid if omitted).

        `backend` overrides `self.backend` for this call only (used by
        reference-vs-optimized comparison tests/experiments so a single
        layer instance with fixed weights can be called both ways).
        `ablation`, if given, applies one evaluation-only override -- see
        `juniper_auto.model.moe_ablations` for exact semantics; `None`
        (the default) is the unmodified production path. `return_trace`
        requires `return_diagnostics=True` and attaches a bounded per-token
        routing trace to the returned diagnostics.

        Returns (output [batch, seq_len, d_model], load_balance_loss_raw,
        router_z_loss_raw, diagnostics_or_None).
        """
        if x.ndim != 3:
            raise ValueError(f"x must have shape [batch, seq_len, d_model], got {tuple(x.shape)}")
        batch, seq_len, d_model = x.shape
        if valid_mask is not None and valid_mask.shape != (batch, seq_len):
            raise ValueError(
                f"valid_mask must have shape {(batch, seq_len)}, got {tuple(valid_mask.shape)}"
            )
        if return_trace and not return_diagnostics:
            raise ValueError("return_trace=True requires return_diagnostics=True")
        if ablation is not None and self.training:
            raise RuntimeError(
                "MoE ablations are evaluation-only; call eval() on the layer/model before passing ablation"
            )
        validate_ablation_for_layer(ablation, self.n_routed_experts)

        flat_x = x.reshape(-1, d_model)
        n_tokens = flat_x.shape[0]

        if valid_mask is None:
            flat_valid = torch.ones(n_tokens, dtype=torch.bool, device=x.device)
        else:
            flat_valid = valid_mask.reshape(-1).to(torch.bool)

        router_logits, router_probs = compute_router_logits_and_probs(flat_x, self.router.weight, self.router.bias)

        if is_router_override(ablation):
            topk_idx, topk_weights = resolve_router_override(
                ablation, n_tokens=n_tokens, n_experts=self.n_routed_experts, top_k=self.top_k, device=x.device
            )
        else:
            topk_idx, topk_weights = select_topk(router_probs, self.top_k, self.renormalize)

        dispatch_fn = DISPATCH_BACKENDS[backend or self.backend]
        dispatch_kwargs = resolve_dispatch_kwargs(ablation)
        disable_shared = should_disable_shared_expert(ablation)
        all_valid = valid_mask is None or bool(flat_valid.all().item())
        if all_valid:
            shared_out = self.shared_expert(flat_x)
            if disable_shared:
                shared_out = torch.zeros_like(shared_out)
            output = dispatch_fn(
                flat_x,
                self.routed_experts,
                topk_idx,
                topk_weights,
                self.n_routed_experts,
                self.top_k,
                shared_out,
                **dispatch_kwargs,
            )
        else:
            # Padding must not consume meaningful expert compute or produce
            # an expert contribution. Dispatch only the compact valid-token
            # view, then scatter back to the original flattened positions.
            # This also handles an all-padding mask without calling experts.
            valid_positions = flat_valid.nonzero(as_tuple=True)[0]
            shared_out = torch.zeros_like(flat_x)
            output = torch.zeros_like(flat_x)
            if valid_positions.numel() > 0:
                valid_x = flat_x[valid_positions]
                valid_shared = self.shared_expert(valid_x)
                if disable_shared:
                    valid_shared = torch.zeros_like(valid_shared)
                valid_output = dispatch_fn(
                    valid_x,
                    self.routed_experts,
                    topk_idx[flat_valid],
                    topk_weights[flat_valid],
                    self.n_routed_experts,
                    self.top_k,
                    valid_shared,
                    **dispatch_kwargs,
                )
                shared_out = shared_out.index_copy(0, valid_positions, valid_shared.to(shared_out.dtype))
                output = output.index_copy(0, valid_positions, valid_output.to(output.dtype))

        load_balance_raw = compute_load_balance_loss_raw(
            router_probs, topk_idx, flat_valid, self.n_routed_experts, self.top_k
        )
        router_z_raw = compute_router_z_loss_raw(router_logits, flat_valid)

        diagnostics = None
        if return_diagnostics:
            # routed_only is reconstructed by subtraction (rather than a
            # second from-scratch dispatch pass) purely for contribution-norm
            # reporting -- it is never used for the loss-critical `output`
            # above, so this introduces no behavioral change to production
            # output, only a small extra subtraction when diagnostics are
            # explicitly requested.
            routed_only_output = output - shared_out
            diagnostics = build_moe_diagnostics(
                router_logits=router_logits,
                router_probs=router_probs,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                valid_mask=flat_valid,
                n_routed_experts=self.n_routed_experts,
                shared_out=shared_out,
                routed_only_output=routed_only_output,
                load_balance_loss_raw=load_balance_raw,
                router_z_loss_raw=router_z_raw,
                load_balance_loss_weighted=self.load_balance_coefficient * load_balance_raw,
                router_z_loss_weighted=self.router_z_coefficient * router_z_raw,
                shared_expert_activated=not disable_shared,
                ablation_mode=ablation.mode if ablation is not None else None,
                **dispatch_kwargs,
                return_trace=return_trace,
                batch=batch,
                seq_len=seq_len,
                max_trace_tokens=max_trace_tokens,
            )

        return output.view(batch, seq_len, d_model), load_balance_raw, router_z_raw, diagnostics
