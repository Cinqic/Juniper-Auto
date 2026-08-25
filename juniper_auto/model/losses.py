"""Loss primitives: shifted causal cross-entropy, MoE load-balancing loss,
and router Z-loss.

The load-balancing and Z-loss formulas are the Juniper Auto v0.1
implementation semantics documented in
docs/adr/0008-moe-auxiliary-loss-semantics.md -- they are not assumed to be
numerically identical to any other model family's routing-loss formula.

For a single MoE layer, over `N` valid tokens, `E` routed experts, and `K`
selected experts per token:

    f_e = (# selected assignments to expert e) / (N * K)
    p_e = mean_over_valid_tokens(full_softmax_probability[e])
    L_balance_layer = E * sum_e(f_e * p_e)

    z_t = logsumexp(router_logits_t)                 (per valid token t)
    L_z_layer = mean_over_valid_tokens(z_t ** 2)

`f_e` is built from integer top-k *indices* (via scatter_add of constant
ones), so it carries no gradient -- only `p_e` (the real softmax
probability) is differentiable, which is what lets the router learn from
this term without a false gradient path through the discrete top-k
selection. Both losses are computed only over valid (non-padding) tokens.
Layer-level values are averaged across MoE layers by the caller (see
`juniper_auto.model.model`); the coefficients (0.01, 0.001) are applied
after that averaging, not per layer.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """logits: [batch, seq_len, vocab]. labels: [batch, seq_len] (token ids,
    or `ignore_index` for positions that must not contribute to the loss --
    e.g. padding, or the position with no valid next-token target).

    Standard next-token shift: position i's logits predict position i+1's
    label. Computed and reduced in FP32 regardless of the input logits
    dtype.
    """
    shift_logits = logits[..., :-1, :].to(torch.float32).contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )


def compute_load_balance_loss_raw(
    router_probs: torch.Tensor,
    topk_idx: torch.Tensor,
    valid_mask: torch.Tensor,
    n_experts: int,
    top_k: int,
) -> torch.Tensor:
    """router_probs: [n_tokens, n_experts] FP32 full softmax.
    topk_idx: [n_tokens, top_k] long, selected expert indices.
    valid_mask: [n_tokens] bool, True for real (non-padding) tokens.

    Returns the raw (unweighted) layer-level load-balance loss, a scalar.
    """
    valid_mask = valid_mask.to(torch.bool)
    n_valid = valid_mask.sum()
    if n_valid.item() == 0:
        return router_probs.new_zeros(())

    valid_topk_idx = topk_idx[valid_mask]
    valid_probs = router_probs[valid_mask]

    assignment_counts = torch.zeros(n_experts, dtype=torch.float32, device=router_probs.device)
    flat_idx = valid_topk_idx.reshape(-1)
    assignment_counts.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32))
    f = assignment_counts / (n_valid.to(torch.float32) * top_k)

    p = valid_probs.mean(dim=0)
    return n_experts * (f * p).sum()


def compute_router_z_loss_raw(
    router_logits: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """router_logits: [n_tokens, n_experts] FP32. valid_mask: [n_tokens] bool.

    Returns the raw (unweighted) layer-level router Z-loss, a scalar.
    """
    valid_mask = valid_mask.to(torch.bool)
    n_valid = valid_mask.sum()
    if n_valid.item() == 0:
        return router_logits.new_zeros(())

    valid_logits = router_logits[valid_mask]
    z = torch.logsumexp(valid_logits, dim=-1)
    return (z**2).mean()
