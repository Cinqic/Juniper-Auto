"""MoE routing instrumentation: entropy, margins, pair co-activation,
contribution norms, router-logit magnitude, window aggregation, and
routing-health detectors -- each checked against a hand-computed expected
value or a synthetic case deliberately constructed to trigger it."""

from __future__ import annotations

import math

import torch

from juniper_auto.model.moe import MoELayer
from juniper_auto.model.moe_diagnostics import (
    RoutingWindowAccumulator,
    compute_contribution_norms,
    compute_entropy,
    compute_expert_pair_coactivation,
    compute_router_logit_magnitude_stats,
    compute_topk_logit_margin,
    compute_topk_prob_margin,
    detect_dead_experts,
    detect_dominant_experts,
    detect_router_saturation,
    detect_routing_collapse,
    detect_routing_oscillation,
    detect_starved_experts,
)
from tests.model_fixtures import make_tiny_sparse_config


def test_entropy_matches_hand_computed_value_for_uniform_and_peaked_distributions():
    probs = torch.tensor([[0.25, 0.25, 0.25, 0.25], [1.0 - 3e-6, 1e-6, 1e-6, 1e-6]])
    entropy, normalized = compute_entropy(probs)
    torch.testing.assert_close(entropy[0], torch.tensor(math.log(4.0)), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(normalized[0], torch.tensor(1.0), atol=1e-5, rtol=1e-5)
    assert entropy[1].item() < 1e-3
    assert normalized[1].item() < 1e-3


def test_topk_margins_match_hand_computed_values():
    probs = torch.tensor([[0.7, 0.2, 0.1]])
    logits = torch.tensor([[5.0, 2.0, -1.0]])
    prob_margin = compute_topk_prob_margin(probs)
    logit_margin = compute_topk_logit_margin(logits)
    torch.testing.assert_close(prob_margin, torch.tensor([0.5]), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(logit_margin, torch.tensor([3.0]), atol=1e-6, rtol=1e-6)


def test_router_logit_magnitude_stats_match_hand_computed_values():
    logits = torch.tensor([[4.0, -2.0], [1.0, -1.0], [10.0, 0.0]])
    valid = torch.tensor([True, True, False])
    stats = compute_router_logit_magnitude_stats(logits, valid)
    # valid rows only: [4,-2,1,-1]
    expected_abs_mean = (4 + 2 + 1 + 1) / 4
    expected_rms = math.sqrt((16 + 4 + 1 + 1) / 4)
    torch.testing.assert_close(stats["abs_mean"], torch.tensor(expected_abs_mean), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(stats["rms"], torch.tensor(expected_rms), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(stats["abs_max"], torch.tensor(4.0), atol=1e-5, rtol=1e-5)


def test_router_logit_magnitude_stats_handle_zero_valid_tokens():
    logits = torch.randn(5, 3)
    valid = torch.zeros(5, dtype=torch.bool)
    stats = compute_router_logit_magnitude_stats(logits, valid)
    assert stats["abs_mean"].item() == 0.0
    assert stats["rms"].item() == 0.0
    assert stats["abs_max"].item() == 0.0


def test_expert_pair_coactivation_matches_hand_computed_matrix():
    # 3 tokens, top_k=2, 4 experts. Pairs: (0,1), (0,1), (2,3). Token 2 (idx 2)
    # is padding and must not contribute.
    topk_idx = torch.tensor([[0, 1], [1, 0], [2, 3]])
    valid = torch.tensor([True, True, False])
    matrix = compute_expert_pair_coactivation(topk_idx, valid, n_experts=4)
    expected = torch.zeros(4, 4, dtype=torch.long)
    expected[0, 1] = 2  # (0,1) appears twice among valid tokens, order-independent
    assert torch.equal(matrix, expected)
    # strictly upper-triangular: nothing on/under the diagonal
    assert torch.equal(torch.tril(matrix), torch.zeros(4, 4, dtype=torch.long))


def test_expert_pair_coactivation_counts_each_pair_once_per_token_not_twice():
    # top_k=3: token contributes exactly C(3,2)=3 pairs, not 6.
    topk_idx = torch.tensor([[0, 1, 2]])
    valid = torch.tensor([True])
    matrix = compute_expert_pair_coactivation(topk_idx, valid, n_experts=3)
    assert matrix.sum().item() == 3
    assert matrix[0, 1].item() == 1 and matrix[0, 2].item() == 1 and matrix[1, 2].item() == 1


def test_contribution_norms_match_hand_computed_values_with_stub_experts():
    layer_cfg = make_tiny_sparse_config(
        n_routed_experts=2, top_k=1, d_model=2, expert_ffn_dim=2, n_query_heads=1, n_kv_heads=1, head_dim=2
    )
    layer = MoELayer(layer_cfg)
    layer.shared_expert.forward = lambda x: torch.full((x.shape[0], 2), 3.0)  # norm = sqrt(9+9) = sqrt(18)
    layer.routed_experts[0].forward = lambda x: torch.full((x.shape[0], 2), 4.0)  # norm = sqrt(16+16)=sqrt(32)
    layer.routed_experts[1].forward = lambda x: torch.full((x.shape[0], 2), 4.0)

    x = torch.randn(3, layer_cfg.core.d_model)
    _, _, _, diag = layer(x.unsqueeze(0), valid_mask=torch.ones(1, 3, dtype=torch.bool), return_diagnostics=True)

    torch.testing.assert_close(diag.shared_contribution_norm_mean, torch.tensor(math.sqrt(18.0)), atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(diag.routed_contribution_norm_mean, torch.tensor(math.sqrt(32.0)), atol=1e-3, rtol=1e-3)


def test_contribution_norms_handle_zero_valid_tokens():
    shared = torch.randn(4, 3)
    routed = torch.randn(4, 3)
    valid = torch.zeros(4, dtype=torch.bool)
    stats = compute_contribution_norms(shared, routed, valid)
    for v in stats.values():
        assert v.item() == 0.0


# --------------------------------------------------------------------------
# Window aggregation
# --------------------------------------------------------------------------


def test_routing_window_accumulator_aggregates_counts_and_load_shares_across_batches():
    torch.manual_seed(0)
    cfg = make_tiny_sparse_config(
        n_routed_experts=4, top_k=2, d_model=8, expert_ffn_dim=8, n_query_heads=1, n_kv_heads=1, head_dim=8
    )
    layer = MoELayer(cfg)
    accumulator = RoutingWindowAccumulator(n_experts=4, top_k=2)
    total_valid = 0
    for seed in range(5):
        torch.manual_seed(seed)
        x = torch.randn(1, 7, cfg.core.d_model)
        valid = torch.ones(1, 7, dtype=torch.bool)
        valid[0, -1] = False
        _, _, _, diag = layer(x, valid_mask=valid, return_diagnostics=True)
        accumulator.add(diag)
        total_valid += 6

    assert accumulator.total_valid_tokens == total_valid
    assert accumulator.assignment_counts.sum().item() == total_valid * 2
    shares = accumulator.load_shares()
    torch.testing.assert_close(shares.sum(), torch.tensor(1.0), atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------------
# Detectors -- each must fire on a constructed pathological case and NOT
# fire on a constructed healthy case.
# --------------------------------------------------------------------------


def test_dead_expert_detector_fires_only_on_zero_share():
    healthy = torch.tensor([0.25, 0.25, 0.25, 0.25])
    assert detect_dead_experts(healthy) == []
    dead = torch.tensor([0.0, 0.4, 0.3, 0.3])
    assert detect_dead_experts(dead) == [0]


def test_starved_expert_detector_fires_on_near_zero_but_nonzero_share():
    n = 4
    uniform = 1.0 / n
    healthy = torch.full((n,), uniform)
    assert detect_starved_experts(healthy) == []
    starved = torch.tensor([0.01 * uniform, 0.33, 0.33, 0.33])
    assert detect_starved_experts(starved) == [0]
    # exactly-zero is "dead", not "starved" -- the two are disjoint sets
    fully_dead = torch.tensor([0.0, 0.33, 0.33, 0.34])
    assert detect_starved_experts(fully_dead) == []


def test_dominant_expert_detector_fires_on_excess_share():
    n = 4
    healthy = torch.full((n,), 1.0 / n)
    assert detect_dominant_experts(healthy) == []
    dominant = torch.tensor([0.85, 0.05, 0.05, 0.05])
    assert detect_dominant_experts(dominant) == [0]


def test_routing_collapse_detector_requires_both_low_entropy_and_concentration():
    healthy_shares = torch.full((4,), 0.25)
    assert not detect_routing_collapse(mean_normalized_entropy=0.95, load_shares=healthy_shares)
    collapsed_shares = torch.tensor([0.9, 0.03, 0.03, 0.04])
    assert detect_routing_collapse(mean_normalized_entropy=0.1, load_shares=collapsed_shares)
    # low entropy alone (without concentration) must not fire
    assert not detect_routing_collapse(mean_normalized_entropy=0.1, load_shares=healthy_shares)


def test_router_saturation_detector_fires_on_large_logits_or_large_margin():
    assert not detect_router_saturation(mean_logit_abs=1.0, mean_top1_margin=0.3)
    assert detect_router_saturation(mean_logit_abs=50.0, mean_top1_margin=0.3)
    assert detect_router_saturation(mean_logit_abs=1.0, mean_top1_margin=0.999)


def test_diagnostics_remain_finite_and_stable_under_large_magnitude_router_inputs():
    # Phase 2 instructions section 17: deliberately test large-magnitude
    # router inputs/logits and confirm saturation-related diagnostics stay
    # numerically stable (no inf/nan from softmax overflow or entropy log(0)).
    cfg = make_tiny_sparse_config(
        n_routed_experts=6, top_k=2, d_model=8, expert_ffn_dim=8, n_query_heads=1, n_kv_heads=1, head_dim=8
    )
    layer = MoELayer(cfg)
    torch.nn.init.normal_(layer.router.weight, std=50.0)  # deliberately large router weights
    x = torch.randn(1, 12, cfg.core.d_model) * 1000.0  # deliberately large-magnitude hidden states

    out, lb, z, diag = layer(x, valid_mask=torch.ones(1, 12, dtype=torch.bool), return_diagnostics=True)

    assert torch.isfinite(out).all()
    assert torch.isfinite(lb).all() and torch.isfinite(z).all()
    assert torch.isfinite(diag.router_logits).all()
    assert torch.isfinite(diag.router_probs).all()
    torch.testing.assert_close(diag.router_probs.sum(dim=-1), torch.ones(12), atol=1e-4, rtol=1e-4)
    assert torch.isfinite(diag.entropy).all() and (diag.entropy >= 0).all()
    assert torch.isfinite(diag.normalized_entropy).all()
    assert torch.isfinite(diag.top1_top2_prob_margin).all()
    assert torch.isfinite(diag.top1_top2_logit_margin).all()
    assert torch.isfinite(diag.router_logit_abs_mean).item()
    assert torch.isfinite(diag.router_logit_rms).item()
    assert torch.isfinite(diag.router_logit_abs_max).item()
    # This case is expected to genuinely trip the saturation detector --
    # confirms the detector is reachable from a real forward pass, not just
    # from hand-fed synthetic scalars.
    from juniper_auto.model.moe_diagnostics import detect_router_saturation

    assert detect_router_saturation(
        mean_logit_abs=diag.router_logit_abs_mean.item(),
        mean_top1_margin=diag.top1_top2_prob_margin.mean().item(),
    )


def test_diagnostics_remain_finite_under_large_magnitude_inputs_with_padding():
    cfg = make_tiny_sparse_config(
        n_routed_experts=6, top_k=2, d_model=8, expert_ffn_dim=8, n_query_heads=1, n_kv_heads=1, head_dim=8
    )
    layer = MoELayer(cfg)
    x = torch.randn(1, 10, cfg.core.d_model) * 5000.0
    valid = torch.ones(1, 10, dtype=torch.bool)
    valid[0, -4:] = False
    out, lb, z, diag = layer(x, valid_mask=valid, return_diagnostics=True)
    assert torch.isfinite(out).all()
    assert torch.isfinite(lb).all() and torch.isfinite(z).all()
    assert torch.isfinite(diag.shared_contribution_norm_mean).item()
    assert torch.isfinite(diag.routed_contribution_norm_mean).item()


def test_routing_oscillation_detector_measures_top1_change_rate():
    a = torch.tensor([[0, 1], [2, 3], [0, 2]])
    b_identical = a.clone()
    assert detect_routing_oscillation(a, b_identical) == 0.0
    b_all_changed = torch.tensor([[1, 0], [3, 2], [2, 0]])
    assert detect_routing_oscillation(a, b_all_changed) == 1.0
    b_one_changed = torch.tensor([[9, 1], [2, 3], [0, 2]])
    assert abs(detect_routing_oscillation(a, b_one_changed) - (1 / 3)) < 1e-6
