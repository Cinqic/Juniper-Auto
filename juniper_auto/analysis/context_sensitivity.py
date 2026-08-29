"""Context-sensitivity measurement infrastructure (Phase 2 instructions
section 12): the OpenMoE-style concern that a router may end up driven
mostly by token identity rather than the surrounding hidden-state context.

Juniper Auto currently has no frozen tokenizer, no real corpus, and no
trained router. This module can therefore measure *whether a routing
decision changed between two contexts*, at initialization, on synthetic or
untrained-model inputs -- it cannot and does not claim anything about
learned semantic specialization. Every function that touches the official
untrained model is loudly labeled as an engineering/proxy test in its
docstring and in the value it returns; do not strip that labeling when
consuming these results.

Two entry points:

  - `compare_routing_across_variants` / `compute_router_decisions`: a
    low-level pair of functions operating directly on router weights and a
    batch of hidden-state vectors. Used for controlled synthetic tests
    (see tests/test_context_sensitivity.py) that prove the metrics
    themselves can distinguish context-independent, partially
    context-dependent, and strongly context-dependent routing -- with
    hand-constructed hidden states where the "right answer" is known.

  - `ProbeCase` / `run_probe_case`: a model-level harness that runs a full
    `JuniperAutoModel` forward pass over several full-sequence "context
    variants" of the same probe token identity and compares routing at
    that token's position, per MoE layer. This is the interface later
    phases (once a real tokenizer and trained checkpoint exist) can feed
    real semantic probe sets into, unchanged.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import torch

from juniper_auto.model.moe_diagnostics import compute_entropy
from juniper_auto.model.routing import compute_router_logits_and_probs, select_topk


@dataclass(frozen=True)
class TextContextVariant:
    label: str
    text: str


@dataclass(frozen=True)
class TextContextProbeTemplate:
    """Tokenizer-independent methodology for a later semantic probe.

    Phase 2 has no tokenizer, so these text templates are not executed as
    language evidence. They freeze the controlled categories and lexical
    identity that Phase 3+ can tokenize into ``ProbeCase`` instances without
    redesigning the measurement after seeing results.
    """

    category: str
    probe_text: str
    variants: tuple[TextContextVariant, ...]


CANONICAL_CONTEXT_PROBE_TEMPLATES: tuple[TextContextProbeTemplate, ...] = (
    TextContextProbeTemplate(
        category="semantic_ambiguity",
        probe_text="bank",
        variants=(
            TextContextVariant("financial-institution", "The bank approved the business loan."),
            TextContextVariant("river-edge", "The canoe rested beside the river bank."),
        ),
    ),
    TextContextProbeTemplate(
        category="same_syntax_different_domains",
        probe_text="class",
        variants=(
            TextContextVariant("ordinary-prose", "The class discussed history after lunch."),
            TextContextVariant("python-code", "class Vehicle: pass"),
        ),
    ),
    TextContextProbeTemplate(
        category="code_prose_lexical_overlap",
        probe_text="return",
        variants=(
            TextContextVariant("ordinary-prose", "Please return the borrowed book tomorrow."),
            TextContextVariant("source-code", "def identity(x): return x"),
        ),
    ),
    TextContextProbeTemplate(
        category="mathematical_symbol_reuse",
        probe_text="*",
        variants=(
            TextContextVariant("multiplication", "The area is width * height."),
            TextContextVariant("wildcard", "Match every file with *.json."),
            TextContextVariant("pointer-like-syntax", "Read the value through *pointer."),
        ),
    ),
    TextContextProbeTemplate(
        category="syntax_reuse_across_formats",
        probe_text=":",
        variants=(
            TextContextVariant("prose", "Note: the result is provisional."),
            TextContextVariant("python", "if ready: launch()"),
            TextContextVariant("json", '{"ready": true}'),
            TextContextVariant("mathematics", "f: A to B"),
        ),
    ),
    TextContextProbeTemplate(
        category="positional_control",
        probe_text="anchor",
        variants=(
            TextContextVariant("early-position", "anchor alpha beta gamma delta"),
            TextContextVariant("late-position", "alpha beta gamma delta anchor"),
        ),
    ),
)


def validate_context_probe_templates(
    templates: tuple[TextContextProbeTemplate, ...] = CANONICAL_CONTEXT_PROBE_TEMPLATES,
) -> None:
    """Fail if the frozen methodology loses identity or category controls."""
    required_categories = {
        "semantic_ambiguity",
        "same_syntax_different_domains",
        "code_prose_lexical_overlap",
        "mathematical_symbol_reuse",
        "syntax_reuse_across_formats",
        "positional_control",
    }
    categories = {template.category for template in templates}
    missing = required_categories - categories
    if missing:
        raise ValueError(f"context probe methodology is missing categories: {sorted(missing)}")
    if len(categories) != len(templates):
        raise ValueError("context probe template categories must be unique")
    for template in templates:
        if len(template.variants) < 2:
            raise ValueError(f"{template.category} needs at least two context variants")
        for variant in template.variants:
            if template.probe_text not in variant.text:
                raise ValueError(
                    f"{template.category}/{variant.label} does not contain exact probe text {template.probe_text!r}"
                )


def _js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> float:
    """Jensen-Shannon divergence (nats) between two categorical
    distributions over experts. Symmetric, bounded in [0, log(2)]."""
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (p / m).log()).sum()
    kl_qm = (q * (q / m).log()).sum()
    return float((0.5 * kl_pm + 0.5 * kl_qm).item())


def compute_router_decisions(
    router_weight: torch.Tensor,
    router_bias: torch.Tensor | None,
    hidden_states: torch.Tensor,
    top_k: int,
    renormalize: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """hidden_states: [n_variants, d_model]. Returns
    (router_logits, router_probs, topk_idx, topk_weights), each with a
    leading n_variants dimension. Reuses the exact same router math as the
    model (juniper_auto.model.routing) -- no separate/duplicated formula."""
    logits, probs = compute_router_logits_and_probs(hidden_states, router_weight, router_bias)
    topk_idx, topk_weights = select_topk(probs, top_k, renormalize)
    return logits, probs, topk_idx, topk_weights


def compare_routing_across_variants(
    router_probs: torch.Tensor,
    topk_idx: torch.Tensor,
) -> dict[str, float]:
    """router_probs: [n_variants, n_experts]. topk_idx: [n_variants, top_k].
    Computes routing-difference metrics over every pair of variants and
    averages them. "pair" here means a pair of context *variants* being
    compared against each other -- not a pair of co-activated experts (see
    `juniper_auto.model.moe_diagnostics.compute_expert_pair_coactivation`
    for that unrelated meaning of "pair").

    Returns a dict with:
      top1_change_rate          -- fraction of variant-pairs whose slot-0
                                    (highest-weight) selected expert differs.
      exact_topk_change_rate    -- fraction of variant-pairs whose full
                                    top-k SET of selected experts differs
                                    (order-independent).
      pair_change_rate          -- alias of exact_topk_change_rate, named to
                                    match the Phase 2 instructions' wording.
      mean_entropy_difference   -- mean |entropy_i - entropy_j| over pairs.
      mean_js_divergence        -- mean Jensen-Shannon divergence (nats)
                                    between full router probability vectors.
    """
    n_variants = router_probs.shape[0]
    if n_variants < 2:
        raise ValueError("compare_routing_across_variants requires at least 2 variants")
    entropy, _ = compute_entropy(router_probs)
    pairs = list(itertools.combinations(range(n_variants), 2))

    top1_changed = 0
    set_changed = 0
    entropy_diffs = []
    js_divs = []
    for i, j in pairs:
        top1_changed += int(topk_idx[i, 0].item() != topk_idx[j, 0].item())
        set_changed += int(set(topk_idx[i].tolist()) != set(topk_idx[j].tolist()))
        entropy_diffs.append(abs(float(entropy[i].item() - entropy[j].item())))
        js_divs.append(_js_divergence(router_probs[i], router_probs[j]))

    n_pairs = len(pairs)
    exact_topk_change_rate = set_changed / n_pairs
    return {
        "top1_change_rate": top1_changed / n_pairs,
        "exact_topk_change_rate": exact_topk_change_rate,
        "pair_change_rate": exact_topk_change_rate,
        "mean_entropy_difference": sum(entropy_diffs) / n_pairs,
        "mean_js_divergence": sum(js_divs) / n_pairs,
    }


# --------------------------------------------------------------------------
# Model-level probe harness
# --------------------------------------------------------------------------


@dataclass
class ProbeVariant:
    label: str
    token_ids: list[int]
    position: int  # index within token_ids where the probe identity sits


@dataclass
class ProbeCase:
    identity_label: str
    variants: list[ProbeVariant]

    def __post_init__(self) -> None:
        if len(self.variants) < 2:
            raise ValueError("a ProbeCase needs at least 2 variants to compare")
        lengths = {len(v.token_ids) for v in self.variants}
        if len(lengths) != 1:
            raise ValueError(
                f"all ProbeVariant.token_ids must have the same length for batching, got lengths {sorted(lengths)}"
            )
        for v in self.variants:
            if not (0 <= v.position < len(v.token_ids)):
                raise ValueError(f"variant {v.label!r} position {v.position} out of range for its sequence")


def run_probe_case(model, probe_case: ProbeCase, device: str | torch.device | None = None) -> dict[int, dict[str, float]]:
    """Runs one forward pass over all of `probe_case`'s variants (batched)
    and compares routing at each variant's marked position, per MoE layer.
    Returns {layer_position (1-indexed): metrics_dict} for every MoE layer
    in `model.layer_kinds`. Dense layers are absent from the returned dict
    (they have no router)."""
    seq_len = len(probe_case.variants[0].token_ids)
    input_ids = torch.tensor([v.token_ids for v in probe_case.variants], dtype=torch.long)
    positions = [v.position for v in probe_case.variants]
    if device is not None:
        input_ids = input_ids.to(device)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            out = model(input_ids, return_diagnostics=True)
    finally:
        model.train(was_training)

    results: dict[int, dict[str, float]] = {}
    n_variants = len(probe_case.variants)
    flat_positions = [variant_idx * seq_len + positions[variant_idx] for variant_idx in range(n_variants)]
    for layer_position, (kind, diag) in enumerate(zip(model.layer_kinds, out.diagnostics), start=1):
        if kind != "moe" or diag is None:
            continue
        probs_at_position = diag.router_probs[flat_positions]
        topk_at_position = diag.topk_idx[flat_positions]
        results[layer_position] = compare_routing_across_variants(probs_at_position, topk_at_position)
    return results


def run_untrained_official_model_probe(
    model,
    probe_token_id: int,
    context_token_ids: list[list[int]],
    seed: int,
) -> dict[int, dict[str, float]]:
    """ENGINEERING/PROXY TEST -- NOT SEMANTIC SPECIALIZATION EVIDENCE.

    `model` is an untrained (randomly initialized) JuniperAutoModel. Any
    routing differences this reports reflect an untrained router's response
    to different hidden-state inputs at initialization -- they are not, and
    cannot be, evidence of learned context-aware / semantic routing, since
    no training has occurred and no real tokenizer or corpus exists yet.
    This function exists to sanity-check the probe harness end-to-end
    against the official architecture, and to give later phases (once a
    real tokenizer and trained checkpoint exist) a template call.

    `context_token_ids`: a list of equal-length token-id sequences, each of
    which must contain `probe_token_id` at least once; the FIRST occurrence
    in each sequence is used as the compared position. `seed` seeds the
    variant construction (currently only used to fix which occurrence is
    picked when ambiguous, present for reproducibility parity with other
    Phase 2 entry points).
    """
    del seed  # currently deterministic given context_token_ids' shape; kept for interface parity
    variants = []
    for i, sequence in enumerate(context_token_ids):
        if probe_token_id not in sequence:
            raise ValueError(f"context sequence {i} does not contain probe_token_id {probe_token_id}")
        variants.append(
            ProbeVariant(label=f"context-{i}", token_ids=list(sequence), position=sequence.index(probe_token_id))
        )
    probe_case = ProbeCase(identity_label=f"token-{probe_token_id}", variants=variants)
    return run_probe_case(model, probe_case)
