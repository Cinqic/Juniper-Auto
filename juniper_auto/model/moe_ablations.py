"""Evaluation-only MoE ablation controls (Phase 2 instructions section 13).

These are analysis-mode overrides, never a default: `MoELayer.forward`'s
`ablation` parameter defaults to `None`, and the `None` path is byte-for-byte
the pre-Phase-2 dispatch -- no ablation state can leak into a normal forward
call. `MoEAblationConfig` instances are constructed explicitly by a caller
that wants one specific evaluation-only override for one specific call.

Exact, frozen semantics per mode (deliberately decided, not left ambiguous):

  disable_routed_expert(expert_id)
      For every token whose top-k selection includes `expert_id`, that
      slot's weighted contribution is forced to zero by skipping it in
      dispatch. The original selected weight is NOT redistributed to the
      token's other selected expert(s) -- no renormalization occurs. This
      isolates the removed expert's marginal contribution: "what would the
      output have been if this expert contributed nothing," not "what
      would routing have looked like if this expert didn't exist."

  zero_expert_output(expert_ids)
      Identical mechanism to disable_routed_expert, generalized to an
      arbitrary set of expert ids evaluated in one call.

  disable_shared_expert()
      The shared expert's contribution is forced to zero. The routed
      contribution (and its weights) is unaffected.

  replace_routed_expert(expert_id, replacement_expert_id)
      For every token whose top-k selection includes `expert_id`, the
      *replacement* expert's forward is run instead, using the *original*
      selected weight for that slot. `expert_id`'s own parameters are never
      called (and so receive no gradient) for this forward call.

  uniform_router()
      The router's top-k selection is replaced entirely by a deterministic
      round-robin assignment: token i (by its flattened position within
      this forward call, 0-indexed) selects experts
      [i % E, (i+1) % E, ..., (i+top_k-1) % E], each with equal weight
      1/top_k. This does NOT depend on torch.topk's tie-breaking behavior
      (no topk call is involved) and is fully deterministic. Real router
      logits/probs are still computed and available for diagnostics, but do
      not drive selection under this mode.

  random_router(seed)
      The router's top-k selection is replaced by top_k unique experts per
      token, drawn from an explicit seeded `torch.Generator` (CPU, isolated
      from ambient global RNG state and from device/backend), each with
      equal weight 1/top_k. Reproducible: the same seed and the same
      (n_tokens, n_experts, top_k) always reproduce the same selection,
      independent of device.

All modes leave `router_logits` / `router_probs` (the real computed
values) untouched for diagnostics -- only the *selection* (topk_idx /
topk_weights) or the *dispatch* (which experts actually execute and
contribute) is overridden, per mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch

AblationMode = Literal[
    "disable_routed_expert",
    "disable_shared_expert",
    "replace_routed_expert",
    "uniform_router",
    "random_router",
    "zero_expert_output",
]

_ROUTER_OVERRIDE_MODES = frozenset({"uniform_router", "random_router"})


@dataclass(frozen=True)
class MoEAblationConfig:
    mode: AblationMode
    expert_id: int | None = None
    replacement_expert_id: int | None = None
    expert_ids: tuple[int, ...] = field(default_factory=tuple)
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.mode == "disable_routed_expert" and self.expert_id is None:
            raise ValueError("disable_routed_expert requires expert_id")
        if self.mode == "replace_routed_expert" and (
            self.expert_id is None or self.replacement_expert_id is None
        ):
            raise ValueError("replace_routed_expert requires expert_id and replacement_expert_id")
        if self.mode == "zero_expert_output" and not self.expert_ids:
            raise ValueError("zero_expert_output requires a non-empty expert_ids")
        if self.mode == "random_router" and self.seed is None:
            raise ValueError("random_router requires an explicit seed for reproducibility")


def is_router_override(ablation: MoEAblationConfig | None) -> bool:
    return ablation is not None and ablation.mode in _ROUTER_OVERRIDE_MODES


def resolve_router_override(
    ablation: MoEAblationConfig,
    n_tokens: int,
    n_experts: int,
    top_k: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (topk_idx, topk_weights), both [n_tokens, top_k], for the
    uniform_router / random_router ablation modes. Only called when
    `is_router_override(ablation)` is True."""
    if ablation.mode == "uniform_router":
        token_idx = torch.arange(n_tokens, device=device)
        slot_offsets = torch.arange(top_k, device=device)
        topk_idx = (token_idx.unsqueeze(1) + slot_offsets.unsqueeze(0)) % n_experts
        topk_weights = torch.full((n_tokens, top_k), 1.0 / top_k, device=device, dtype=torch.float32)
        return topk_idx, topk_weights

    if ablation.mode == "random_router":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(ablation.seed)
        random_scores = torch.rand(n_tokens, n_experts, generator=generator)
        _, topk_idx = torch.topk(random_scores, k=top_k, dim=-1)
        topk_idx = topk_idx.to(device)
        topk_weights = torch.full((n_tokens, top_k), 1.0 / top_k, device=device, dtype=torch.float32)
        return topk_idx, topk_weights

    raise ValueError(f"{ablation.mode!r} is not a router-override ablation mode")


def resolve_dispatch_kwargs(ablation: MoEAblationConfig | None) -> dict:
    """Returns the dispatch-backend kwargs (disabled_expert_ids,
    expert_replacement_map, zero_expert_ids) implied by `ablation`. All
    empty/None when `ablation` is None or does not target dispatch (e.g. a
    router-override or shared-expert-disable mode) -- those two are handled
    separately by the caller (MoELayer.forward), not here."""
    if ablation is None:
        return {}
    if ablation.mode == "disable_routed_expert":
        return {"disabled_expert_ids": frozenset({ablation.expert_id})}
    if ablation.mode == "zero_expert_output":
        return {"zero_expert_ids": frozenset(ablation.expert_ids)}
    if ablation.mode == "replace_routed_expert":
        return {"expert_replacement_map": {ablation.expert_id: ablation.replacement_expert_id}}
    return {}


def should_disable_shared_expert(ablation: MoEAblationConfig | None) -> bool:
    return ablation is not None and ablation.mode == "disable_shared_expert"
