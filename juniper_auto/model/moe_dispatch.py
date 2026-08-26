"""Two dispatch backends that combine an already-computed top-k routing
decision with the expert modules to produce the routed contribution to a
MoE layer's output. Both backends:

  - are dropless (every selected assignment is executed, nothing dropped
    for capacity reasons),
  - accept the same optional ablation hooks (`disabled_expert_ids`,
    `expert_replacement_map`, `zero_expert_ids` -- see
    `juniper_auto.model.moe_ablations` for exact semantics), which are all
    no-ops by default (empty set / None), and
  - accumulate their routed contribution into a caller-supplied
    `initial_output` tensor via the same `index_add` pattern, rather than
    returning a routed-only tensor that the caller must remember to add.

`reference_dispatch` is the Phase 1 correctness-first path: an explicit
Python loop over (expert, slot) with boolean-mask gather. It loops over
`n_routed_experts * top_k` (expert, slot) pairs and does one boolean `==`
compare over the full token dimension per pair -- deliberately the most
literal, inspectable translation of "each token's top-k assignments are
individually executed and scattered back."

For default arguments (no ablation), `reference_dispatch` performs the
exact same sequence of tensor operations in the exact same order as the
Phase 1 `MoELayer.forward` body it was extracted from -- this is verified
directly in tests/test_model_moe_dispatch.py against a loaded copy of the
approved Phase 1 commit's `moe.py`, not just asserted here.

`optimized_dispatch` is the Phase 2 pure-PyTorch alternative: it sorts the
flattened (token, slot) assignments by expert id once, then does exactly
one gather + one dense matmul + one scatter per *expert* (not per
(expert, slot) pair), replacing up to `top_k` boolean full-tensor compares
per expert with a single global argsort. It is not guaranteed to produce
bit-identical output to the reference path (summation order differs), but
must match it within a tight, explicitly justified floating-point
tolerance -- see tests/test_model_moe_dispatch.py.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn


def reference_dispatch(
    flat_x: torch.Tensor,
    routed_experts: Sequence[nn.Module],
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    n_routed_experts: int,
    top_k: int,
    initial_output: torch.Tensor,
    *,
    disabled_expert_ids: frozenset[int] = frozenset(),
    expert_replacement_map: Mapping[int, int] | None = None,
    zero_expert_ids: frozenset[int] = frozenset(),
) -> torch.Tensor:
    output = initial_output
    for expert_id, expert in enumerate(routed_experts):
        if expert_id in disabled_expert_ids or expert_id in zero_expert_ids:
            # Ablation: this slot's weighted contribution is forced to zero
            # by skipping it entirely -- the original selected weight is
            # never redistributed to other selected experts.
            continue
        exec_id = expert_replacement_map.get(expert_id, expert_id) if expert_replacement_map else expert_id
        exec_expert = routed_experts[exec_id]
        for slot in range(top_k):
            slot_mask = topk_idx[:, slot] == expert_id
            if not torch.any(slot_mask):
                continue
            expert_out = exec_expert(flat_x[slot_mask])
            weight = topk_weights[slot_mask, slot : slot + 1].to(expert_out.dtype)
            output = output.index_add(
                0, slot_mask.nonzero(as_tuple=True)[0], (weight * expert_out).to(output.dtype)
            )
    return output


def optimized_dispatch(
    flat_x: torch.Tensor,
    routed_experts: Sequence[nn.Module],
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    n_routed_experts: int,
    top_k: int,
    initial_output: torch.Tensor,
    *,
    disabled_expert_ids: frozenset[int] = frozenset(),
    expert_replacement_map: Mapping[int, int] | None = None,
    zero_expert_ids: frozenset[int] = frozenset(),
) -> torch.Tensor:
    device = flat_x.device
    n_tokens = topk_idx.shape[0]

    flat_expert_ids = topk_idx.reshape(-1)
    flat_weights = topk_weights.reshape(-1)
    token_idx_repeated = (
        torch.arange(n_tokens, device=device).unsqueeze(1).expand(n_tokens, top_k).reshape(-1)
    )

    # Group every (token, slot) assignment by expert id with a single sort,
    # instead of one boolean full-tensor compare per (expert, slot) pair.
    sort_order = torch.argsort(flat_expert_ids, stable=True)
    sorted_expert_ids = flat_expert_ids[sort_order]
    sorted_token_idx = token_idx_repeated[sort_order]
    sorted_weights = flat_weights[sort_order]
    counts = torch.bincount(sorted_expert_ids, minlength=n_routed_experts).tolist()

    output = initial_output
    start = 0
    for expert_id in range(n_routed_experts):
        count = counts[expert_id]
        if count == 0:
            continue
        idx_slice = sorted_token_idx[start : start + count]
        weight_slice = sorted_weights[start : start + count]
        start += count
        if expert_id in disabled_expert_ids or expert_id in zero_expert_ids:
            continue
        exec_id = expert_replacement_map.get(expert_id, expert_id) if expert_replacement_map else expert_id
        exec_expert = routed_experts[exec_id]
        expert_out = exec_expert(flat_x[idx_slice])
        weight = weight_slice.unsqueeze(-1).to(expert_out.dtype)
        output = output.index_add(0, idx_slice, (weight * expert_out).to(output.dtype))
    return output


DISPATCH_BACKENDS = {
    "reference": reference_dispatch,
    "optimized": optimized_dispatch,
}
