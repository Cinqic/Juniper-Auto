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
from collections.abc import Mapping

import torch
from torch import nn

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
    load_balance_loss_weighted: torch.Tensor | None = None  # scalar, coefficient applied exactly once
    router_z_loss_weighted: torch.Tensor | None = None  # scalar, coefficient applied exactly once
    ablation_mode: str | None = None  # explicit audit label; None is the production path
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
    load_balance_loss_weighted: torch.Tensor,
    router_z_loss_weighted: torch.Tensor,
    shared_expert_activated: bool,
    ablation_mode: str | None,
    disabled_expert_ids: frozenset[int] = frozenset(),
    expert_replacement_map: Mapping[int, int] | None = None,
    zero_expert_ids: frozenset[int] = frozenset(),
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
        token_trace = build_token_trace(
            topk_idx,
            topk_weights,
            valid_mask,
            batch,
            seq_len,
            shared_expert_activated=shared_expert_activated,
            disabled_expert_ids=disabled_expert_ids,
            expert_replacement_map=expert_replacement_map,
            zero_expert_ids=zero_expert_ids,
        )

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
        load_balance_loss_weighted=load_balance_loss_weighted.detach(),
        router_z_loss_weighted=router_z_loss_weighted.detach(),
        ablation_mode=ablation_mode,
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
    executed_expert_1: int  # -1 when padding/disabled/zeroed
    executed_expert_2: int  # -1 when padding/disabled/zeroed/top_k < 2
    weight_1: float
    weight_2: float  # 0.0 if top_k < 2
    routed_assignment_count: int
    weights_sum: float
    weights_normalized: bool
    shared_expert_activated: bool
    reconstruction_position: int
    token_id: int | None = None
    token_text: str | None = None


def build_token_trace(
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    valid_mask: torch.Tensor,
    batch: int,
    seq_len: int,
    *,
    shared_expert_activated: bool = True,
    disabled_expert_ids: frozenset[int] = frozenset(),
    expert_replacement_map: Mapping[int, int] | None = None,
    zero_expert_ids: frozenset[int] = frozenset(),
) -> list[TokenRoutingTraceRecord]:
    top_k = topk_idx.shape[1]
    idx_list = topk_idx.tolist()
    weight_list = topk_weights.tolist()
    valid_list = valid_mask.tolist()
    records = []
    for flat_idx, (experts, weights, is_valid) in enumerate(zip(idx_list, weight_list, valid_list)):
        b, s = divmod(flat_idx, seq_len)
        weights_sum = float(sum(weights))
        executed = []
        for expert_id in experts:
            if not is_valid or expert_id in disabled_expert_ids or expert_id in zero_expert_ids:
                executed.append(-1)
            else:
                executed.append(
                    expert_replacement_map.get(expert_id, expert_id) if expert_replacement_map else expert_id
                )
        records.append(
            TokenRoutingTraceRecord(
                layer_index=-1,
                batch_index=b,
                seq_position=s,
                flat_token_index=flat_idx,
                is_valid=bool(is_valid),
                expert_1=experts[0],
                expert_2=experts[1] if top_k >= 2 else -1,
                executed_expert_1=executed[0],
                executed_expert_2=executed[1] if top_k >= 2 else -1,
                weight_1=weights[0],
                weight_2=weights[1] if top_k >= 2 else 0.0,
                routed_assignment_count=sum(expert_id >= 0 for expert_id in executed),
                weights_sum=weights_sum,
                weights_normalized=math.isclose(weights_sum, 1.0, rel_tol=1e-5, abs_tol=1e-5),
                shared_expert_activated=bool(shared_expert_activated and is_valid),
                reconstruction_position=flat_idx,
            )
        )
    return records


def assemble_full_trace(
    layer_kinds: list[str],
    diagnostics: list[MoEDiagnostics | None],
    *,
    token_ids: torch.Tensor | None = None,
    token_text_by_id: dict[int, str] | None = None,
) -> list[TokenRoutingTraceRecord]:
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
        if token_ids is not None:
            if token_ids.ndim != 2:
                raise ValueError("token_ids must have shape [batch, seq_len]")
            expected_shape = (diag.token_trace[-1].batch_index + 1, diag.token_trace[-1].seq_position + 1)
            if tuple(token_ids.shape) != expected_shape:
                raise ValueError(
                    f"token_ids must match traced [batch, seq_len] shape {expected_shape}, got {tuple(token_ids.shape)}"
                )
        for record in diag.token_trace:
            token_id = None
            token_text = None
            if token_ids is not None:
                token_id = int(token_ids[record.batch_index, record.seq_position].item())
                if token_text_by_id is not None:
                    token_text = token_text_by_id.get(token_id)
            full_trace.append(
                dataclasses.replace(
                    record,
                    layer_index=position,
                    token_id=token_id,
                    token_text=token_text,
                )
            )
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


# --------------------------------------------------------------------------
# Post-backward expert gradient telemetry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpertGradientNorms:
    """Per-MoE-layer gradient L2 norms collected after ``backward()``.

    ``None`` means the module received no gradient at all in the inspected
    backward pass; ``0.0`` means it participated but the resulting gradient
    happened to be exactly zero. That distinction is important when
    diagnosing starvation or a detached expert path.
    """

    layer_index: int
    router: float | None
    shared_expert: float | None
    routed_experts: tuple[float | None, ...]


def _module_gradient_l2_norm(module: nn.Module) -> float | None:
    squared_sum: torch.Tensor | None = None
    saw_gradient = False
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        saw_gradient = True
        term = parameter.grad.detach().float().pow(2).sum()
        squared_sum = term if squared_sum is None else squared_sum + term
    if not saw_gradient:
        return None
    return float(squared_sum.sqrt().item())


def collect_layer_expert_gradient_norms(moe_layer: nn.Module, *, layer_index: int = -1) -> ExpertGradientNorms:
    """Collect router/shared/per-routed-expert norms from one MoE layer.

    This is deliberately a post-backward function rather than a forward
    diagnostic field: gradients do not exist when ``MoEDiagnostics`` is
    constructed. The collector never mutates or detaches the training
    graph and is safe to call between ``backward()`` and ``zero_grad()``.
    """
    required = ("router", "shared_expert", "routed_experts")
    missing = [name for name in required if not hasattr(moe_layer, name)]
    if missing:
        raise TypeError(f"expected a MoE layer with {required}, missing {missing}")
    return ExpertGradientNorms(
        layer_index=layer_index,
        router=_module_gradient_l2_norm(moe_layer.router),
        shared_expert=_module_gradient_l2_norm(moe_layer.shared_expert),
        routed_experts=tuple(_module_gradient_l2_norm(expert) for expert in moe_layer.routed_experts),
    )


def collect_model_expert_gradient_norms(model: nn.Module) -> list[ExpertGradientNorms]:
    """Collect gradient telemetry for every MoE layer in model order."""
    if not hasattr(model, "layers") or not hasattr(model, "layer_kinds"):
        raise TypeError("expected a JuniperAutoModel-like object with layers and layer_kinds")
    records = []
    for layer_index, (kind, block) in enumerate(zip(model.layer_kinds, model.layers), start=1):
        if kind == "moe":
            records.append(collect_layer_expert_gradient_norms(block.moe, layer_index=layer_index))
    return records


# Thresholds are analysis/evaluation configuration, not architecture. They
# therefore live in a caller-overridable dataclass rather than the frozen
# architecture YAML. Defaults are synthetic engineering baselines, not
# research truths or trained-model calibration.
@dataclass(frozen=True)
class RoutingHealthThresholds:
    dead_expert_load_share: float = 0.0
    starved_expert_uniform_share_ratio: float = 0.1
    dominant_expert_uniform_share_ratio: float = 3.0
    collapse_normalized_entropy: float = 0.3
    collapse_top_expert_load_share: float = 0.5
    saturation_logit_abs_mean: float = 20.0
    saturation_top1_margin: float = 0.99
    oscillation_top1_change_rate: float = 0.5

    def __post_init__(self) -> None:
        unit_interval_fields = (
            "dead_expert_load_share",
            "collapse_normalized_entropy",
            "collapse_top_expert_load_share",
            "saturation_top1_margin",
            "oscillation_top1_change_rate",
        )
        for name in unit_interval_fields:
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        for name in (
            "starved_expert_uniform_share_ratio",
            "dominant_expert_uniform_share_ratio",
            "saturation_logit_abs_mean",
        ):
            value = getattr(self, name)
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0, got {value}")


DEFAULT_ROUTING_HEALTH_THRESHOLDS = RoutingHealthThresholds()

# Backward-compatible named defaults for existing callers and documentation.
DEAD_EXPERT_LOAD_SHARE_THRESHOLD = DEFAULT_ROUTING_HEALTH_THRESHOLDS.dead_expert_load_share
STARVED_EXPERT_LOAD_SHARE_RATIO = DEFAULT_ROUTING_HEALTH_THRESHOLDS.starved_expert_uniform_share_ratio
DOMINANT_EXPERT_LOAD_SHARE_RATIO = DEFAULT_ROUTING_HEALTH_THRESHOLDS.dominant_expert_uniform_share_ratio
COLLAPSE_NORMALIZED_ENTROPY_THRESHOLD = DEFAULT_ROUTING_HEALTH_THRESHOLDS.collapse_normalized_entropy
COLLAPSE_TOP_EXPERT_LOAD_SHARE_THRESHOLD = DEFAULT_ROUTING_HEALTH_THRESHOLDS.collapse_top_expert_load_share
SATURATION_LOGIT_ABS_MEAN_THRESHOLD = DEFAULT_ROUTING_HEALTH_THRESHOLDS.saturation_logit_abs_mean
SATURATION_TOP1_MARGIN_THRESHOLD = DEFAULT_ROUTING_HEALTH_THRESHOLDS.saturation_top1_margin
OSCILLATION_TOP1_CHANGE_RATE_THRESHOLD = DEFAULT_ROUTING_HEALTH_THRESHOLDS.oscillation_top1_change_rate


def detect_dead_experts(
    load_shares: torch.Tensor,
    thresholds: RoutingHealthThresholds = DEFAULT_ROUTING_HEALTH_THRESHOLDS,
) -> list[int]:
    return [i for i, share in enumerate(load_shares.tolist()) if share <= thresholds.dead_expert_load_share]


def detect_starved_experts(
    load_shares: torch.Tensor,
    thresholds: RoutingHealthThresholds = DEFAULT_ROUTING_HEALTH_THRESHOLDS,
) -> list[int]:
    n_experts = load_shares.shape[0]
    uniform = 1.0 / n_experts
    return [
        i
        for i, share in enumerate(load_shares.tolist())
        if thresholds.dead_expert_load_share < share < thresholds.starved_expert_uniform_share_ratio * uniform
    ]


def detect_dominant_experts(
    load_shares: torch.Tensor,
    thresholds: RoutingHealthThresholds = DEFAULT_ROUTING_HEALTH_THRESHOLDS,
) -> list[int]:
    n_experts = load_shares.shape[0]
    uniform = 1.0 / n_experts
    return [
        i for i, share in enumerate(load_shares.tolist())
        if share > thresholds.dominant_expert_uniform_share_ratio * uniform
    ]


def detect_routing_collapse(
    mean_normalized_entropy: float,
    load_shares: torch.Tensor,
    thresholds: RoutingHealthThresholds = DEFAULT_ROUTING_HEALTH_THRESHOLDS,
) -> bool:
    top_share = load_shares.max().item() if load_shares.numel() else 0.0
    return (
        mean_normalized_entropy < thresholds.collapse_normalized_entropy
        and top_share > thresholds.collapse_top_expert_load_share
    )


def detect_router_saturation(
    mean_logit_abs: float,
    mean_top1_margin: float,
    thresholds: RoutingHealthThresholds = DEFAULT_ROUTING_HEALTH_THRESHOLDS,
) -> bool:
    return (
        mean_logit_abs > thresholds.saturation_logit_abs_mean
        or mean_top1_margin > thresholds.saturation_top1_margin
    )


def detect_routing_oscillation(topk_idx_a: torch.Tensor, topk_idx_b: torch.Tensor) -> float:
    """Given two probes of the SAME input at two different points (e.g. two
    training steps), returns the fraction of tokens whose top-1 (slot 0)
    expert differs. Caller compares against OSCILLATION_TOP1_CHANGE_RATE_THRESHOLD."""
    if topk_idx_a.shape != topk_idx_b.shape:
        raise ValueError("topk_idx_a and topk_idx_b must have the same shape to compare an identical probe")
    changed = (topk_idx_a[:, 0] != topk_idx_b[:, 0]).float().mean()
    return float(changed.item())


def is_pathological_routing_oscillation(
    change_rate: float,
    thresholds: RoutingHealthThresholds = DEFAULT_ROUTING_HEALTH_THRESHOLDS,
) -> bool:
    """Classify a measured identical-probe top-1 change rate."""
    if not 0.0 <= change_rate <= 1.0:
        raise ValueError(f"change_rate must be in [0, 1], got {change_rate}")
    return change_rate > thresholds.oscillation_top1_change_rate
