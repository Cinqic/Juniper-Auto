"""Pure router math shared by every MoE dispatch backend: FP32 logits,
FP32 softmax, top-k selection, and weight renormalization.

Extracted unmodified from the Phase 1 `MoELayer.forward` body so that the
reference and optimized dispatch backends (`juniper_auto.model.moe_dispatch`)
are guaranteed to route identically -- neither backend computes its own
router math, so a routing bug cannot exist in only one backend and a
dispatch bug cannot masquerade as a routing difference.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_router_logits_and_probs(
    flat_x: torch.Tensor,
    router_weight: torch.Tensor,
    router_bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """flat_x: [n_tokens, d_model]. Returns (router_logits, router_probs),
    both [n_tokens, n_experts] FP32, computed with autocast explicitly
    disabled so ambient mixed precision can never downcast this path --
    this is the frozen v0.1 precision policy (router_logits_dtype=fp32,
    router_softmax_dtype=fp32), not something a caller can opt out of."""
    bias_fp32 = router_bias.to(torch.float32) if router_bias is not None else None
    with torch.autocast(device_type=flat_x.device.type, enabled=False):
        router_logits = F.linear(flat_x.to(torch.float32), router_weight.to(torch.float32), bias_fp32)
        router_probs = F.softmax(router_logits, dim=-1)
    return router_logits, router_probs


def select_topk(
    router_probs: torch.Tensor,
    top_k: int,
    renormalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """router_probs: [n_tokens, n_experts] FP32. Selects the top-k experts by
    softmax probability (not by raw logit) and returns (topk_idx, topk_weights),
    both [n_tokens, top_k]. When renormalize is True, weights are rescaled to
    sum to 1 per token (the frozen v0.1 default); otherwise the raw top-k
    probabilities are used unmodified.

    torch.topk on a continuous probability tensor selects k distinct indices
    by construction (no duplicate expert can be selected for one token)."""
    topk_probs, topk_idx = torch.topk(router_probs, k=top_k, dim=-1)
    if renormalize:
        denom = topk_probs.sum(dim=-1, keepdim=True).clamp_min(1e-20)
        topk_weights = topk_probs / denom
    else:
        topk_weights = topk_probs
    return topk_idx, topk_weights
