"""Context-sensitivity probe harness (Phase 2 instructions section 12):
prove the metric distinguishes context-independent, partially
context-dependent, and strongly context-dependent routing using controlled
synthetic hidden states with a known "right answer" -- then a labeled,
untrained-official-model sanity smoke test that explicitly does NOT claim
semantic evidence."""

from __future__ import annotations

import torch

from juniper_auto.analysis.context_sensitivity import (
    ProbeCase,
    ProbeVariant,
    compare_routing_across_variants,
    compute_router_decisions,
    run_probe_case,
    run_untrained_official_model_probe,
)
from juniper_auto.model import build_model
from tests.model_fixtures import make_tiny_sparse_config


def _router(d_model=16, n_experts=8, seed=0):
    torch.manual_seed(seed)
    router = torch.nn.Linear(d_model, n_experts, bias=False)
    return router


def test_identical_hidden_states_are_context_independent():
    router = _router()
    base = torch.randn(1, router.in_features).expand(6, -1)
    _, probs, topk_idx, _ = compute_router_decisions(router.weight, None, base, top_k=2, renormalize=True)
    metrics = compare_routing_across_variants(probs, topk_idx)
    assert metrics["top1_change_rate"] == 0.0
    assert metrics["exact_topk_change_rate"] == 0.0
    assert metrics["mean_js_divergence"] == 0.0
    assert metrics["mean_entropy_difference"] == 0.0


def test_large_orthogonal_perturbations_are_strongly_context_dependent():
    d_model, n_experts = 16, 8
    router = _router(d_model, n_experts, seed=1)
    torch.manual_seed(2)
    base = torch.randn(d_model)
    # One large, distinct perturbation per variant, scaled well beyond the
    # router's typical logit range so each variant's top-1 expert is driven
    # by a different perturbation direction.
    variants = []
    for i in range(n_experts):
        direction = torch.zeros(d_model)
        direction[i % d_model] = 50.0
        variants.append(base + direction)
    hidden_states = torch.stack(variants)
    _, probs, topk_idx, _ = compute_router_decisions(router.weight, None, hidden_states, top_k=2, renormalize=True)
    metrics = compare_routing_across_variants(probs, topk_idx)
    assert metrics["top1_change_rate"] > 0.5
    assert metrics["mean_js_divergence"] > 0.1


def test_small_perturbations_are_partially_context_dependent_between_the_two_extremes():
    d_model, n_experts = 16, 8
    router = _router(d_model, n_experts, seed=3)
    torch.manual_seed(4)
    base = torch.randn(d_model)

    identical = base.expand(6, -1)
    _, probs_id, topk_id, _ = compute_router_decisions(router.weight, None, identical, top_k=2, renormalize=True)
    identical_metrics = compare_routing_across_variants(probs_id, topk_id)

    strong = torch.stack([base + torch.nn.functional.one_hot(torch.tensor(i % d_model), d_model).float() * 50.0 for i in range(6)])
    _, probs_strong, topk_strong, _ = compute_router_decisions(router.weight, None, strong, top_k=2, renormalize=True)
    strong_metrics = compare_routing_across_variants(probs_strong, topk_strong)

    torch.manual_seed(5)
    partial = base.unsqueeze(0) + torch.randn(6, d_model) * 0.05
    _, probs_partial, topk_partial, _ = compute_router_decisions(router.weight, None, partial, top_k=2, renormalize=True)
    partial_metrics = compare_routing_across_variants(probs_partial, topk_partial)

    # Partial perturbation's JS divergence should sit strictly between the
    # context-independent floor and the strongly-perturbed ceiling.
    assert identical_metrics["mean_js_divergence"] == 0.0
    assert identical_metrics["mean_js_divergence"] < partial_metrics["mean_js_divergence"]
    assert partial_metrics["mean_js_divergence"] < strong_metrics["mean_js_divergence"]


def test_pair_change_rate_is_an_alias_of_exact_topk_change_rate():
    router = _router()
    torch.manual_seed(6)
    hidden = torch.randn(5, router.in_features)
    _, probs, topk_idx, _ = compute_router_decisions(router.weight, None, hidden, top_k=2, renormalize=True)
    metrics = compare_routing_across_variants(probs, topk_idx)
    assert metrics["pair_change_rate"] == metrics["exact_topk_change_rate"]


def test_compare_routing_requires_at_least_two_variants():
    router = _router()
    hidden = torch.randn(1, router.in_features)
    _, probs, topk_idx, _ = compute_router_decisions(router.weight, None, hidden, top_k=2, renormalize=True)
    import pytest

    with pytest.raises(ValueError, match="at least 2 variants"):
        compare_routing_across_variants(probs, topk_idx)


# --------------------------------------------------------------------------
# Model-level probe harness
# --------------------------------------------------------------------------


def test_probe_case_rejects_variants_of_different_lengths():
    import pytest

    with pytest.raises(ValueError, match="same length"):
        ProbeCase(
            identity_label="x",
            variants=[
                ProbeVariant(label="a", token_ids=[1, 2, 3], position=1),
                ProbeVariant(label="b", token_ids=[1, 2], position=1),
            ],
        )


def test_run_probe_case_returns_metrics_only_for_moe_layers():
    cfg = make_tiny_sparse_config(
        n_layers=6, dense_layers=[1, 6], moe_layers=[2, 3, 4, 5],
        d_model=8, n_routed_experts=4, top_k=2, expert_ffn_dim=8,
        n_query_heads=2, n_kv_heads=1, head_dim=4, vocab_size=50,
    )
    model = build_model(cfg, seed=0)
    torch.manual_seed(9)
    probe_token = 7
    variants = []
    for i in range(4):
        seq = torch.randint(0, cfg.embeddings.vocab_size, (10,)).tolist()
        seq[3] = probe_token
        variants.append(ProbeVariant(label=f"v{i}", token_ids=seq, position=3))
    probe_case = ProbeCase(identity_label="probe-7", variants=variants)

    metrics_by_layer = run_probe_case(model, probe_case)
    assert set(metrics_by_layer.keys()) == {2, 3, 4, 5}
    for metrics in metrics_by_layer.values():
        assert 0.0 <= metrics["top1_change_rate"] <= 1.0
        assert 0.0 <= metrics["exact_topk_change_rate"] <= 1.0
        assert metrics["mean_js_divergence"] >= 0.0


def test_untrained_official_model_probe_runs_and_is_explicitly_labeled():
    cfg = make_tiny_sparse_config(
        n_layers=4, dense_layers=[1, 4], moe_layers=[2, 3],
        d_model=8, n_routed_experts=4, top_k=2, expert_ffn_dim=8,
        n_query_heads=2, n_kv_heads=1, head_dim=4, vocab_size=50,
    )
    model = build_model(cfg, seed=0)
    torch.manual_seed(10)
    probe_token = 5
    contexts = []
    for _ in range(4):
        seq = torch.randint(0, cfg.embeddings.vocab_size, (8,)).tolist()
        seq[2] = probe_token
        contexts.append(seq)

    metrics_by_layer = run_untrained_official_model_probe(model, probe_token, contexts, seed=0)
    assert set(metrics_by_layer.keys()) == {2, 3}

    assert "ENGINEERING/PROXY TEST" in run_untrained_official_model_probe.__doc__
    assert "NOT SEMANTIC SPECIALIZATION EVIDENCE" in run_untrained_official_model_probe.__doc__
