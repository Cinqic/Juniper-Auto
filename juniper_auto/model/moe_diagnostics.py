"""MoE routing instrumentation: per-call diagnostics (`MoEDiagnostics`,
extended from the Phase 1 dataclass), per-token routing traces, aggregated
multi-batch window statistics, and routing-health detectors.

Everything here is opt-in and computed only when `return_diagnostics=True`
is passed to `MoELayer.forward` -- ordinary training/inference forward
calls never construct any of these objects. Aggregate scalars (means,
norms, entropy) are cheap and safe to collect routinely; the optional
per-token trace (`return_trace=True`) is explicitly bounded (see
`MAX_TRACE_TOKENS_DEFAULT`) because it allocates one Python record per
token and is meant for small, deliberate diagnostic runs, not production
inference.

Batch-level vs. window-level vs. trained-model behavior (Phase 2
instructions section 11): everything computed from a single forward call
(a `MoEDiagnostics` instance) is a *batch-level* statistic. Aggregating
many such instances (`RoutingWindowAccumulator`) produces a *window-level*
statistic -- still not evidence of learned/trained behavior, since Juniper
Auto has no pretrained checkpoint. Neither this module nor its callers may
claim "expert specialization" or "dead expert" from an untrained model on a
tiny sample; the detectors below are measurement infrastructure, to be
applied to real training-time windows once a model is actually being
trained.
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch

MAX_TRACE_TOKENS_DEFAULT = 4096


@dataclass
class MoEDiagnostics:
    # --- Phase 1 fields (unchanged; existing callers rely on these) ---
    router_logits: torch.Tensor  # [n_tokens, n_experts] fp32
    router_probs: torch.Tensor  # [n_tokens, n_experts] fp32
    topk_idx: torch.Tensor  # [n_tokens, top_k] long
    topk_weights: torch.Tensor  # [n_tokens, top_k] fp32, renormalized, sums to 1 per valid token
    valid_mask: torch.Tensor  # [n_tokens] bool
    assignment_counts_per_expert: torch.Tensor  # [n_experts], valid tokens only

    # --- Phase 2 additions: all aggregate/valid-token-only unless noted ---
    entropy: torch.Tensor | None = None  # [n_tokens], full-distribution entropy (nats)
    normalized_entropy: torch.Tensor | None = None  # [n_tokens], entropy / log(n_experts)
    top1_top2_prob_margin: torch.Tensor | None = None  # [n_tokens]
    top1_top2_logit_margin: torch.Tensor | None = None  # [n_tokens]
    router_logit_abs_mean: torch.Tensor | None = None  # scalar, valid tokens only
    router_logit_rms: torch.Tensor | None = None  # scalar, valid tokens only
    router_logit_abs_max: torch.Tensor | None = None  # scalar, valid tokens only
    expert_pair_coactivation: torch.Tensor | None = None  # [n_experts, n_experts], strictly upper-triangular
    shared_contribution_norm_mean: torch.Tensor | None = None  # scalar
    shared_contribution_norm_rms: torch.Tensor | None = None  # scalar
    routed_contribution_norm_mean: torch.Tensor | None = None  # scalar
    routed_contribution_norm_rms: torch.Tensor | None = None  # scalar
    routed_shared_norm_ratio: torch.Tensor | None = None  # scalar
    load_balance_loss_raw: torch.Tensor | None = None  # scalar, carried through for convenience
    router_z_loss_raw: torch.Tensor | None = None  # scalar
    token_trace: list["TokenRoutingTraceRecord"] | None = None  # only if return_trace=True


def compute_entropy(router_probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """router_probs: [n_tokens, n_experts] fp32 full softmax distribution.
    Returns (entropy, normalized_entropy), both [n_tokens]. normalized_entropy
    = entropy / log(n_experts), in [0, 1] (1.0 = uniform distribution)."""
    n_experts = router_probs.shape[-1]
    probs = router_probs.clamp_min(1e-12)
    entropy = -(probs * probs.log()).sum(dim=-1)
    normalized_entropy = entropy / math.log(n_experts)
    return entropy, normalized_entropy


def compute_topk_prob_margin(router_probs: torch.Tensor) -> torch.Tensor:
    top2, _ = torch.topk(router_probs, k=2, dim=-1)
    return top2[:, 0] - top2[:, 1]


def compute_topk_logit_margin(router_logits: torch.Tensor) -> torch.Tensor:
    top2, _ = torch.topk(router_logits, k=2, dim=-1)
    return top2[:, 0] - top2[:, 1]


def compute_router_logit_magnitude_stats(
    router_logits: torch.Tensor, valid_mask: torch.Tensor
) -> dict[str, torch.Tensor]:
    n_valid = valid_mask.sum()
    if n_valid.item() == 0:
        z = router_logits.new_zeros(())
        return {"abs_mean": z, "rms": z, "abs_max": z}
    valid_logits = router_logits[valid_mask]
    abs_logits = valid_logits.abs()
    return {
        "abs_mean": abs_logits.mean(),
        "rms": valid_logits.pow(2).mean().sqrt(),
        "abs_max": abs_logits.max(),
    }


def compute_expert_pair_coactivation(
    topk_idx: torch.Tensor, valid_mask: torch.Tensor, n_experts: int
) -> torch.Tensor:
    """Returns a [n_experts, n_experts] long tensor. matrix[a, b] for a < b
    is the number of valid tokens whose top-k selection included both
    expert a and expert b; matrix[a, b] for a >= b is always 0. Strictly
    upper-triangular by construction so no pair is ever counted twice, and
    top-k routing guarantees a != b within one token's selection (torch.topk
    over distinct probabilities never repeats an index)."""
    device = topk_idx.device
    matrix = torch.zeros(n_experts, n_experts, dtype=torch.long, device=device)
    valid_topk = topk_idx[valid_mask]
    if valid_topk.numel() == 0:
        return matrix
    k = valid_topk.shape[1]
    for i in range(k):
        for j in range(i + 1, k):
            a = valid_topk[:, i]
            b = valid_topk[:, j]
            lo = torch.minimum(a, b)
            hi = torch.maximum(a, b)
            flat_idx = lo * n_experts + hi
            counts = torch.bincount(flat_idx, minlength=n_experts * n_experts)
            matrix += counts.view(n_experts, n_experts)
    return matrix


def compute_contribution_norms(
    shared_out: torch.Tensor, routed_only_output: torch.Tensor, valid_mask: torch.Tensor
) -> dict[str, torch.Tensor]:
    n_valid = valid_mask.sum()
    if n_valid.item() == 0:
        z = shared_out.new_zeros(())
        return {
            "shared_norm_mean": z, "shared_norm_rms": z,
            "routed_norm_mean": z, "routed_norm_rms": z,
            "routed_shared_ratio": z,
        }
    shared_norms = shared_out[valid_mask].float().norm(dim=-1)
    routed_norms = routed_only_output[valid_mask].float().norm(dim=-1)
    eps = 1e-12
    shared_mean = shared_norms.mean()
    return {
        "shared_norm_mean": shared_mean,
        "shared_norm_rms": shared_norms.pow(2).mean().sqrt(),
        "routed_norm_mean": routed_norms.mean(),
        "routed_norm_rms": routed_norms.pow(2).mean().sqrt(),
        "routed_shared_ratio": routed_norms.mean() / (shared_mean + eps),
    }


def build_moe_diagnostics(
    *,
    router_logits: torch.Tensor,
    router_probs: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    valid_mask: torch.Tensor,
    n_routed_experts: int,
    shared_out: torch.Tensor,
    routed_only_output: torch.Tensor,
    load_balance_loss_raw: torch.Tensor,
    router_z_loss_raw: torch.Tensor,
    return_trace: bool = False,
    batch: int,
    seq_len: int,
    max_trace_tokens: int | None = MAX_TRACE_TOKENS_DEFAULT,
) -> MoEDiagnostics:
    valid_topk_idx = topk_idx[valid_mask]
    counts = torch.zeros(n_routed_experts, dtype=torch.long, device=topk_idx.device)
    if valid_topk_idx.numel() > 0:
        counts.scatter_add_(0, valid_topk_idx.reshape(-1), torch.ones_like(valid_topk_idx.reshape(-1)))

    entropy, normalized_entropy = compute_entropy(router_probs)
    logit_stats = compute_router_logit_magnitude_stats(router_logits, valid_mask)
    contribution = compute_contribution_norms(shared_out, routed_only_output, valid_mask)

    token_trace = None
    if return_trace:
        n_tokens = topk_idx.shape[0]
        if max_trace_tokens is not None and n_tokens > max_trace_tokens:
            raise ValueError(
                f"return_trace=True requested for {n_tokens} tokens, exceeding max_trace_tokens="
                f"{max_trace_tokens}. Per-token traces are meant for small, bounded diagnostic runs "
                "-- pass a larger max_trace_tokens explicitly (or None to disable the guard) only if "
                "you have deliberately sized the diagnostic run's memory footprint."
            )
        token_trace = build_token_trace(topk_idx, topk_weights, valid_mask, batch, seq_len)

    return MoEDiagnostics(
        router_logits=router_logits,
        router_probs=router_probs,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        valid_mask=valid_mask,
        assignment_counts_per_expert=counts,
        entropy=entropy,
        normalized_entropy=normalized_entropy,
        top1_top2_prob_margin=compute_topk_prob_margin(router_probs),
        top1_top2_logit_margin=compute_topk_logit_margin(router_logits),
        router_logit_abs_mean=logit_stats["abs_mean"],
        router_logit_rms=logit_stats["rms"],
        router_logit_abs_max=logit_stats["abs_max"],
        expert_pair_coactivation=compute_expert_pair_coactivation(topk_idx, valid_mask, n_routed_experts),
        shared_contribution_norm_mean=contribution["shared_norm_mean"],
        shared_contribution_norm_rms=contribution["shared_norm_rms"],
        routed_contribution_norm_mean=contribution["routed_norm_mean"],
        routed_contribution_norm_rms=contribution["routed_norm_rms"],
        routed_shared_norm_ratio=contribution["routed_shared_ratio"],
        load_balance_loss_raw=load_balance_loss_raw.detach(),
        router_z_loss_raw=router_z_loss_raw.detach(),
        token_trace=token_trace,
    )


# --------------------------------------------------------------------------
# Per-token routing trace
# --------------------------------------------------------------------------


@dataclass
class TokenRoutingTraceRecord:
    layer_index: int  # -1 until assigned by assemble_full_trace / a model-level caller
    batch_index: int
    seq_position: int
    flat_token_index: int
    is_valid: bool
    expert_1: int
    expert_2: int  # -1 if top_k < 2
    weight_1: float
    weight_2: float  # 0.0 if top_k < 2


def build_token_trace(
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    valid_mask: torch.Tensor,
    batch: int,
    seq_len: int,
) -> list[TokenRoutingTraceRecord]:
    top_k = topk_idx.shape[1]
    idx_list = topk_idx.tolist()
    weight_list = topk_weights.tolist()
    valid_list = valid_mask.tolist()
    records = []
    for flat_idx, (experts, weights, is_valid) in enumerate(zip(idx_list, weight_list, valid_list)):
        b, s = divmod(flat_idx, seq_len)
        records.append(
            TokenRoutingTraceRecord(
                layer_index=-1,
                batch_index=b,
                seq_position=s,
                flat_token_index=flat_idx,
                is_valid=bool(is_valid),
                expert_1=experts[0],
                expert_2=experts[1] if top_k >= 2 else -1,
                weight_1=weights[0],
                weight_2=weights[1] if top_k >= 2 else 0.0,
            )
        )
    return records


def assemble_full_trace(layer_kinds: list[str], diagnostics: list[MoEDiagnostics | None]) -> list[TokenRoutingTraceRecord]:
    """Reassigns each MoEDiagnostics' token_trace records' layer_index to the
    model's 1-indexed layer position (matching core.dense_layers/moe_layers
    numbering) and concatenates them. `diagnostics` is a per-model-layer list
    (as returned in ModelOutput.diagnostics -- one entry per layer, None for
    dense layers); `layer_kinds` is `model.layer_kinds`."""
    if len(layer_kinds) != len(diagnostics):
        raise ValueError("layer_kinds and diagnostics must have the same length (one entry per model layer)")
    full_trace: list[TokenRoutingTraceRecord] = []
    for position, (kind, diag) in enumerate(zip(layer_kinds, diagnostics), start=1):
        if kind != "moe" or diag is None or diag.token_trace is None:
            continue
        for record in diag.token_trace:
            full_trace.append(dataclasses.replace(record, layer_index=position))
    return full_trace


def export_trace_json(records: list[TokenRoutingTraceRecord], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump([dataclasses.asdict(r) for r in records], f, indent=2)


# --------------------------------------------------------------------------
# Window-level aggregation and routing-health detectors
# --------------------------------------------------------------------------


class RoutingWindowAccumulator:
    """Aggregates multiple batches' MoEDiagnostics into window-level
    statistics. A single batch's load imbalance proves nothing (an
    untrained router's per-batch counts are noisy); this is the unit that
    detectors below are meant to be applied to."""

    def __init__(self, n_experts: int, top_k: int):
        self.n_experts = n_experts
        self.top_k = top_k
        self.n_batches = 0
        self.total_valid_tokens = 0
        self.assignment_counts = torch.zeros(n_experts, dtype=torch.long)
        self._entropy_sum = 0.0
        self._n_entropy_samples = 0
        self._logit_abs_mean_sum = 0.0
        self._prob_margin_sum = 0.0

    def add(self, diagnostics: MoEDiagnostics) -> None:
        self.n_batches += 1
        self.assignment_counts += diagnostics.assignment_counts_per_expert.detach().cpu()
        n_valid = int(diagnostics.valid_mask.sum().item())
        self.total_valid_tokens += n_valid
        if n_valid == 0:
            return
        valid_entropy = diagnostics.entropy[diagnostics.valid_mask]
        self._entropy_sum += valid_entropy.sum().item()
        self._n_entropy_samples += n_valid
        self._logit_abs_mean_sum += float(diagnostics.router_logit_abs_mean.item()) * n_valid
        valid_margin = diagnostics.top1_top2_prob_margin[diagnostics.valid_mask]
        self._prob_margin_sum += valid_margin.sum().item()

    def load_shares(self) -> torch.Tensor:
        denom = max(self.total_valid_tokens * self.top_k, 1)
        return self.assignment_counts.float() / denom

    def mean_normalized_entropy(self) -> float:
        if self._n_entropy_samples == 0:
            return float("nan")
        return (self._entropy_sum / self._n_entropy_samples) / math.log(self.n_experts)

    def mean_logit_abs(self) -> float:
        return self._logit_abs_mean_sum / self.total_valid_tokens if self.total_valid_tokens else float("nan")

    def mean_top1_margin(self) -> float:
        return self._prob_margin_sum / self.total_valid_tokens if self.total_valid_tokens else float("nan")


# Thresholds are analysis/evaluation configuration, not architecture -- they
# do not appear in configs/architecture/*.yaml. Documented here as the single
# source of truth; each has a synthetic-case test in
# tests/test_model_moe_diagnostics.py proving it fires and doesn't misfire.
DEAD_EXPERT_LOAD_SHARE_THRESHOLD = 0.0
STARVED_EXPERT_LOAD_SHARE_RATIO = 0.1  # < 10% of the uniform 1/n_experts share
DOMINANT_EXPERT_LOAD_SHARE_RATIO = 3.0  # > 3x the uniform 1/n_experts share
COLLAPSE_NORMALIZED_ENTROPY_THRESHOLD = 0.3
COLLAPSE_TOP_EXPERT_LOAD_SHARE_THRESHOLD = 0.5
SATURATION_LOGIT_ABS_MEAN_THRESHOLD = 20.0
SATURATION_TOP1_MARGIN_THRESHOLD = 0.99
OSCILLATION_TOP1_CHANGE_RATE_THRESHOLD = 0.5


def detect_dead_experts(load_shares: torch.Tensor) -> list[int]:
    return [i for i, share in enumerate(load_shares.tolist()) if share <= DEAD_EXPERT_LOAD_SHARE_THRESHOLD]


def detect_starved_experts(load_shares: torch.Tensor) -> list[int]:
    n_experts = load_shares.shape[0]
    uniform = 1.0 / n_experts
    return [
        i
        for i, share in enumerate(load_shares.tolist())
        if DEAD_EXPERT_LOAD_SHARE_THRESHOLD < share < STARVED_EXPERT_LOAD_SHARE_RATIO * uniform
    ]


def detect_dominant_experts(load_shares: torch.Tensor) -> list[int]:
    n_experts = load_shares.shape[0]
    uniform = 1.0 / n_experts
    return [i for i, share in enumerate(load_shares.tolist()) if share > DOMINANT_EXPERT_LOAD_SHARE_RATIO * uniform]


def detect_routing_collapse(mean_normalized_entropy: float, load_shares: torch.Tensor) -> bool:
    top_share = load_shares.max().item() if load_shares.numel() else 0.0
    return (
        mean_normalized_entropy < COLLAPSE_NORMALIZED_ENTROPY_THRESHOLD
        and top_share > COLLAPSE_TOP_EXPERT_LOAD_SHARE_THRESHOLD
    )


def detect_router_saturation(mean_logit_abs: float, mean_top1_margin: float) -> bool:
    return mean_logit_abs > SATURATION_LOGIT_ABS_MEAN_THRESHOLD or mean_top1_margin > SATURATION_TOP1_MARGIN_THRESHOLD


def detect_routing_oscillation(topk_idx_a: torch.Tensor, topk_idx_b: torch.Tensor) -> float:
    """Given two probes of the SAME input at two different points (e.g. two
    training steps), returns the fraction of tokens whose top-1 (slot 0)
    expert differs. Caller compares against OSCILLATION_TOP1_CHANGE_RATE_THRESHOLD."""
    if topk_idx_a.shape != topk_idx_b.shape:
        raise ValueError("topk_idx_a and topk_idx_b must have the same shape to compare an identical probe")
    changed = (topk_idx_a[:, 0] != topk_idx_b[:, 0]).float().mean()
    return float(changed.item())
