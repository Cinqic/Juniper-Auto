"""Evaluation-only MoE ablations (Phase 2 instructions section 13-14): each
mode's exact mathematical semantics, checked with deterministic constant-
output expert stubs so the expected output tensor can be computed exactly
rather than merely asserting "something changed". Also proves seeded
random-routing reproducibility and that no ablation leaks into a normal
(ablation=None) forward call."""

from __future__ import annotations

import torch

from juniper_auto.model.moe import MoELayer
from juniper_auto.model.moe_ablations import MoEAblationConfig
from tests.model_fixtures import make_tiny_sparse_config


def _stub_layer(n_routed=4, top_k=2, d_model=4):
    cfg = make_tiny_sparse_config(
        n_routed_experts=n_routed, top_k=top_k, d_model=d_model, expert_ffn_dim=d_model,
        n_query_heads=1, n_kv_heads=1, head_dim=d_model,
    )
    layer = MoELayer(cfg)
    layer.shared_expert.forward = lambda x: torch.full((x.shape[0], d_model), 100.0)
    for i, expert in enumerate(layer.routed_experts):
        expert.forward = (lambda v: lambda x: torch.full((x.shape[0], d_model), v))(float(i + 1))
    return layer, cfg


def _call(layer, x_2d, ablation=None, **kwargs):
    out, lb, z, diag = layer(x_2d.unsqueeze(0), valid_mask=torch.ones(1, x_2d.shape[0], dtype=torch.bool), ablation=ablation, return_diagnostics=True, **kwargs)
    return out.squeeze(0), lb, z, diag


def test_normal_forward_matches_shared_plus_weighted_routed_equation():
    layer, cfg = _stub_layer(n_routed=4, top_k=2, d_model=4)
    torch.manual_seed(0)
    x = torch.randn(6, cfg.core.d_model)
    out, _, _, diag = _call(layer, x)
    for t in range(6):
        e0, e1 = diag.topk_idx[t].tolist()
        w0, w1 = diag.topk_weights[t].tolist()
        expected = 100.0 + w0 * (e0 + 1) + w1 * (e1 + 1)
        torch.testing.assert_close(out[t], torch.full((cfg.core.d_model,), expected), atol=1e-4, rtol=1e-4)


def test_disable_routed_expert_zeros_its_term_without_renormalizing():
    layer, cfg = _stub_layer(n_routed=4, top_k=2, d_model=4)
    torch.manual_seed(1)
    x = torch.randn(8, cfg.core.d_model)
    ablation = MoEAblationConfig(mode="disable_routed_expert", expert_id=2)
    out, _, _, diag = _call(layer, x, ablation=ablation)
    for t in range(8):
        e0, e1 = diag.topk_idx[t].tolist()
        w0, w1 = diag.topk_weights[t].tolist()
        contrib0 = 0.0 if e0 == 2 else w0 * (e0 + 1)
        contrib1 = 0.0 if e1 == 2 else w1 * (e1 + 1)
        expected = 100.0 + contrib0 + contrib1
        torch.testing.assert_close(out[t], torch.full((cfg.core.d_model,), expected), atol=1e-4, rtol=1e-4)


def test_zero_expert_output_generalizes_disable_to_a_set():
    layer, cfg = _stub_layer(n_routed=4, top_k=2, d_model=4)
    torch.manual_seed(2)
    x = torch.randn(8, cfg.core.d_model)
    ablation = MoEAblationConfig(mode="zero_expert_output", expert_ids=(1, 2))
    out, _, _, diag = _call(layer, x, ablation=ablation)
    for t in range(8):
        e0, e1 = diag.topk_idx[t].tolist()
        w0, w1 = diag.topk_weights[t].tolist()
        contrib0 = 0.0 if e0 in (1, 2) else w0 * (e0 + 1)
        contrib1 = 0.0 if e1 in (1, 2) else w1 * (e1 + 1)
        expected = 100.0 + contrib0 + contrib1
        torch.testing.assert_close(out[t], torch.full((cfg.core.d_model,), expected), atol=1e-4, rtol=1e-4)


def test_disable_shared_expert_zeros_only_the_shared_term():
    layer, cfg = _stub_layer(n_routed=4, top_k=2, d_model=4)
    torch.manual_seed(3)
    x = torch.randn(6, cfg.core.d_model)
    ablation = MoEAblationConfig(mode="disable_shared_expert")
    out, _, _, diag = _call(layer, x, ablation=ablation)
    for t in range(6):
        e0, e1 = diag.topk_idx[t].tolist()
        w0, w1 = diag.topk_weights[t].tolist()
        expected = w0 * (e0 + 1) + w1 * (e1 + 1)  # no +100.0
        torch.testing.assert_close(out[t], torch.full((cfg.core.d_model,), expected), atol=1e-4, rtol=1e-4)


def test_replace_routed_expert_uses_replacement_output_with_original_weight():
    layer, cfg = _stub_layer(n_routed=4, top_k=2, d_model=4)
    torch.manual_seed(4)
    x = torch.randn(8, cfg.core.d_model)
    ablation = MoEAblationConfig(mode="replace_routed_expert", expert_id=0, replacement_expert_id=3)
    out, _, _, diag = _call(layer, x, ablation=ablation)
    for t in range(8):
        e0, e1 = diag.topk_idx[t].tolist()
        w0, w1 = diag.topk_weights[t].tolist()
        val0 = 4.0 if e0 == 0 else float(e0 + 1)  # expert 0's slot runs expert 3 (value 4.0) instead
        val1 = 4.0 if e1 == 0 else float(e1 + 1)
        expected = 100.0 + w0 * val0 + w1 * val1
        torch.testing.assert_close(out[t], torch.full((cfg.core.d_model,), expected), atol=1e-4, rtol=1e-4)


def test_replace_routed_expert_never_calls_the_replaced_experts_own_parameters():
    layer, cfg = _stub_layer(n_routed=4, top_k=2, d_model=4)
    # Restore real (parametrized) experts so gradients are meaningful.
    del layer.shared_expert.forward
    for expert in layer.routed_experts:
        del expert.forward
    torch.manual_seed(5)
    x = torch.randn(1, 10, cfg.core.d_model, requires_grad=True)
    valid = torch.ones(1, 10, dtype=torch.bool)
    ablation = MoEAblationConfig(mode="replace_routed_expert", expert_id=0, replacement_expert_id=1)
    out, lb, z, diag = layer(x, valid_mask=valid, ablation=ablation, return_diagnostics=True)
    (out.sum() + lb + z).backward()
    assert all(p.grad is None for p in layer.routed_experts[0].parameters())


def test_uniform_router_assigns_deterministic_round_robin_experts_with_equal_weight():
    layer, cfg = _stub_layer(n_routed=4, top_k=2, d_model=4)
    x = torch.randn(9, cfg.core.d_model)
    ablation = MoEAblationConfig(mode="uniform_router")
    out, _, _, diag = _call(layer, x, ablation=ablation)
    for t in range(9):
        assert diag.topk_idx[t].tolist() == [t % 4, (t + 1) % 4]
    torch.testing.assert_close(diag.topk_weights, torch.full((9, 2), 0.5), atol=1e-6, rtol=1e-6)


def test_uniform_router_selection_is_independent_of_actual_router_weights():
    layer_a, cfg = _stub_layer(n_routed=4, top_k=2, d_model=4)
    layer_b, _ = _stub_layer(n_routed=4, top_k=2, d_model=4)
    torch.nn.init.normal_(layer_b.router.weight, std=5.0)  # very different router
    x = torch.randn(1, 6, cfg.core.d_model)
    valid = torch.ones(1, 6, dtype=torch.bool)
    ablation = MoEAblationConfig(mode="uniform_router")
    _, _, _, diag_a = layer_a(x, valid_mask=valid, ablation=ablation, return_diagnostics=True)
    _, _, _, diag_b = layer_b(x, valid_mask=valid, ablation=ablation, return_diagnostics=True)
    assert torch.equal(diag_a.topk_idx, diag_b.topk_idx)
    assert torch.equal(diag_a.topk_weights, diag_b.topk_weights)


def test_random_router_selects_unique_experts_and_equal_weight():
    layer, cfg = _stub_layer(n_routed=6, top_k=3, d_model=4)
    x = torch.randn(12, cfg.core.d_model)
    ablation = MoEAblationConfig(mode="random_router", seed=42)
    out, _, _, diag = _call(layer, x, ablation=ablation)
    for row in diag.topk_idx.tolist():
        assert len(set(row)) == 3
    torch.testing.assert_close(diag.topk_weights, torch.full((12, 3), 1 / 3), atol=1e-6, rtol=1e-6)


def test_random_router_is_reproducible_given_the_same_seed():
    layer, cfg = _stub_layer(n_routed=6, top_k=2, d_model=4)
    x = torch.randn(1, 10, cfg.core.d_model)
    valid = torch.ones(1, 10, dtype=torch.bool)
    ablation = MoEAblationConfig(mode="random_router", seed=7)
    _, _, _, diag_1 = layer(x, valid_mask=valid, ablation=ablation, return_diagnostics=True)
    _, _, _, diag_2 = layer(x, valid_mask=valid, ablation=ablation, return_diagnostics=True)
    assert torch.equal(diag_1.topk_idx, diag_2.topk_idx)


def test_random_router_is_independent_of_ambient_global_rng_state():
    layer, cfg = _stub_layer(n_routed=6, top_k=2, d_model=4)
    x = torch.randn(1, 10, cfg.core.d_model)
    valid = torch.ones(1, 10, dtype=torch.bool)
    ablation = MoEAblationConfig(mode="random_router", seed=7)

    torch.manual_seed(0)
    _, _, _, diag_1 = layer(x, valid_mask=valid, ablation=ablation, return_diagnostics=True)
    torch.manual_seed(999)  # different ambient global RNG state
    _, _, _, diag_2 = layer(x, valid_mask=valid, ablation=ablation, return_diagnostics=True)
    assert torch.equal(diag_1.topk_idx, diag_2.topk_idx)


def test_different_seeds_produce_different_random_routing_with_high_probability():
    layer, cfg = _stub_layer(n_routed=8, top_k=2, d_model=4)
    x = torch.randn(1, 20, cfg.core.d_model)
    valid = torch.ones(1, 20, dtype=torch.bool)
    _, _, _, diag_a = layer(x, valid_mask=valid, ablation=MoEAblationConfig(mode="random_router", seed=1), return_diagnostics=True)
    _, _, _, diag_b = layer(x, valid_mask=valid, ablation=MoEAblationConfig(mode="random_router", seed=2), return_diagnostics=True)
    assert not torch.equal(diag_a.topk_idx, diag_b.topk_idx)


def test_ablation_none_is_byte_identical_to_pre_ablation_forward():
    layer, cfg = _stub_layer(n_routed=4, top_k=2, d_model=4)
    torch.manual_seed(6)
    x = torch.randn(1, 7, cfg.core.d_model)
    valid = torch.ones(1, 7, dtype=torch.bool)

    out_a, lb_a, z_a, diag_a = layer(x, valid_mask=valid, ablation=None, return_diagnostics=True)
    out_b, lb_b, z_b, diag_b = layer(x, valid_mask=valid, return_diagnostics=True)  # ablation omitted entirely

    assert torch.equal(out_a, out_b)
    assert torch.equal(lb_a, lb_b)
    assert torch.equal(z_a, z_b)
    assert torch.equal(diag_a.topk_idx, diag_b.topk_idx)


def test_ablation_state_does_not_persist_across_calls():
    # A call using an ablation must not leave any mutated module state behind
    # that changes the *next*, unrelated, ablation=None call's output.
    layer, cfg = _stub_layer(n_routed=4, top_k=2, d_model=4)
    torch.manual_seed(8)
    x = torch.randn(1, 7, cfg.core.d_model)
    valid = torch.ones(1, 7, dtype=torch.bool)

    out_before, _, _, _ = layer(x, valid_mask=valid)
    for ablation in [
        MoEAblationConfig(mode="disable_shared_expert"),
        MoEAblationConfig(mode="disable_routed_expert", expert_id=0),
        MoEAblationConfig(mode="replace_routed_expert", expert_id=0, replacement_expert_id=1),
        MoEAblationConfig(mode="uniform_router"),
        MoEAblationConfig(mode="random_router", seed=1),
        MoEAblationConfig(mode="zero_expert_output", expert_ids=(0, 1)),
    ]:
        layer(x, valid_mask=valid, ablation=ablation)
    out_after, _, _, _ = layer(x, valid_mask=valid)
    assert torch.equal(out_before, out_after)
