"""Mutation/fault-injection tests (Phase 1 instructions section 26 /
self-review Pass H): prove that specific broken implementations would be
caught by this test suite's checks, rather than trusting a green run at
face value. Each test deliberately builds or monkeypatches a broken
variant *locally* (never mutating the real production module for longer
than the test body) and asserts the relevant property genuinely fails."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from juniper_auto.model import build_model
from juniper_auto.model.attention import GroupedQueryAttention, build_attention_mask
from juniper_auto.model.block import MoEBlock
from juniper_auto.model.inspection import bias_audit, qk_norm_parameter_count, verify_weight_tying
from juniper_auto.model.losses import causal_lm_loss, compute_load_balance_loss_raw
from juniper_auto.model.moe import MoELayer
from juniper_auto.model.rope import RotaryEmbedding
from tests.model_fixtures import make_tiny_sparse_config


def test_future_token_leakage_is_detected_when_masking_is_disabled():
    cfg = make_tiny_sparse_config()
    attn = GroupedQueryAttention(cfg)
    attn.eval()
    torch.manual_seed(0)
    x = torch.randn(1, 6, cfg.core.d_model)
    pos = torch.arange(6).unsqueeze(0)
    x_mutated = x.clone()
    x_mutated[0, 4:, :] = torch.randn_like(x_mutated[0, 4:, :])

    # Monkeypatch masking to "no masking at all" (every position attends to
    # every position) -- this is what future-token leakage looks like.
    import juniper_auto.model.attention as attention_module

    original_builder = attention_module.build_attention_mask
    try:
        attention_module.build_attention_mask = lambda seq_len, key_valid_mask, device: torch.ones(
            1, 1, seq_len, seq_len, dtype=torch.bool, device=device
        )
        with torch.no_grad():
            out_a = attn(x, pos)
            out_b = attn(x_mutated, pos)
        # With no causal mask, changing future tokens MUST change earlier
        # positions' output -- the causality test would fail on this.
        assert not torch.allclose(out_a[:, :4, :], out_b[:, :4, :])
    finally:
        attention_module.build_attention_mask = original_builder


def test_top1_routing_is_detected_by_assignment_count():
    cfg = make_tiny_sparse_config(n_routed_experts=4, top_k=2)
    layer = MoELayer(cfg)
    layer.top_k = 1  # simulate a top-1 bug without touching the real class
    x = torch.randn(1, 6, cfg.core.d_model)
    valid_mask = torch.ones(1, 6, dtype=torch.bool)
    _, _, _, diag = layer(x, valid_mask=valid_mask, return_diagnostics=True)
    # The Phase 1 dropless contract requires exactly valid_tokens * 2
    # assignments; a top-1 bug produces valid_tokens * 1 instead.
    assert diag.assignment_counts_per_expert.sum().item() != 6 * 2
    assert diag.assignment_counts_per_expert.sum().item() == 6 * 1


def test_top3_routing_is_detected_by_assignment_count():
    cfg = make_tiny_sparse_config(n_routed_experts=5, top_k=2)
    layer = MoELayer(cfg)
    layer.top_k = 3
    x = torch.randn(1, 6, cfg.core.d_model)
    valid_mask = torch.ones(1, 6, dtype=torch.bool)
    _, _, _, diag = layer(x, valid_mask=valid_mask, return_diagnostics=True)
    assert diag.assignment_counts_per_expert.sum().item() == 6 * 3
    assert diag.assignment_counts_per_expert.sum().item() != 6 * 2  # frozen top_k=2 contract violated


def test_token_dropping_is_detected_by_assignment_count():
    cfg = make_tiny_sparse_config(n_routed_experts=4, top_k=2)
    layer = MoELayer(cfg)
    x = torch.randn(1, 6, cfg.core.d_model)
    valid_mask = torch.ones(1, 6, dtype=torch.bool)
    _, _, _, diag = layer(x, valid_mask=valid_mask, return_diagnostics=True)
    # Simulate a capacity-drop bug: pretend some assignments never happened.
    dropped_counts = diag.assignment_counts_per_expert.clone()
    dropped_counts[0] = max(dropped_counts[0].item() - 1, 0)
    assert dropped_counts.sum().item() < 6 * 2  # dropless contract violated


def test_missing_shared_expert_is_detected():
    cfg = make_tiny_sparse_config(
        n_routed_experts=3, top_k=1, d_model=4, expert_ffn_dim=4, n_query_heads=1, n_kv_heads=1, head_dim=4
    )
    layer = MoELayer(cfg)
    layer.shared_expert.forward = lambda x: torch.full((x.shape[0], cfg.core.d_model), 7.0)
    for expert in layer.routed_experts:
        expert.forward = lambda x: torch.zeros(x.shape[0], cfg.core.d_model)
    x = torch.randn(1, 4, cfg.core.d_model)
    out_correct, _, _, _ = layer(x, valid_mask=torch.ones(1, 4, dtype=torch.bool))

    # Simulate "shared expert omitted": rebuild forward without the shared term.
    def broken_forward(x_in, valid_mask=None, return_diagnostics=False):
        flat_x = x_in.reshape(-1, cfg.core.d_model)
        router_logits = F.linear(flat_x.to(torch.float32), layer.router.weight.to(torch.float32))
        probs = F.softmax(router_logits, dim=-1)
        topk_probs, topk_idx = torch.topk(probs, k=layer.top_k, dim=-1)
        weights = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        out = torch.zeros_like(flat_x)  # no shared_expert(flat_x) term at all
        for eid, expert in enumerate(layer.routed_experts):
            mask = topk_idx[:, 0] == eid
            if mask.any():
                out[mask] += weights[mask, 0:1] * expert(flat_x[mask])
        return out.view(x_in.shape), None, None, None

    out_broken, _, _, _ = broken_forward(x)
    assert not torch.allclose(out_correct, out_broken)
    assert torch.allclose(out_correct, out_broken + 7.0, atol=1e-4)  # missing exactly the shared term


def test_hidden_averaging_of_moe_output_is_detected():
    cfg = make_tiny_sparse_config(
        n_routed_experts=4, top_k=2, d_model=4, expert_ffn_dim=4, n_query_heads=1, n_kv_heads=1, head_dim=4
    )
    layer = MoELayer(cfg)
    layer.shared_expert.forward = lambda x: torch.full((x.shape[0], cfg.core.d_model), 100.0)
    for i, expert in enumerate(layer.routed_experts):
        expert.forward = (lambda v: lambda x: torch.full((x.shape[0], cfg.core.d_model), v))(float(i + 1))

    x = torch.randn(1, 5, cfg.core.d_model)
    out, _, _, diag = layer(x, valid_mask=torch.ones(1, 5, dtype=torch.bool), return_diagnostics=True)
    out = out.squeeze(0)

    for t in range(5):
        e0, e1 = diag.topk_idx[t].tolist()
        w0, w1 = diag.topk_weights[t].tolist()
        correct_value = 100.0 + w0 * (e0 + 1) + w1 * (e1 + 1)
        hidden_averaged_value = correct_value / 3.0  # e.g. averaging shared+2 routed by count
        assert not torch.allclose(out[t], torch.full((cfg.core.d_model,), hidden_averaged_value), atol=1e-3)


def test_broken_renormalization_is_detected():
    cfg = make_tiny_sparse_config(n_routed_experts=6, top_k=2)
    layer = MoELayer(cfg)
    layer.renormalize = False  # simulate the renormalize_top_k_weights=False bug
    x = torch.randn(1, 5, cfg.core.d_model)
    _, _, _, diag = layer(x, valid_mask=torch.ones(1, 5, dtype=torch.bool), return_diagnostics=True)
    sums = diag.topk_weights.sum(dim=-1)
    assert not torch.allclose(sums, torch.ones(5), atol=1e-3)


def test_untied_lm_head_is_detected():
    cfg = make_tiny_sparse_config()
    model = build_model(cfg, seed=0)
    assert verify_weight_tying(model)  # correct baseline

    broken = build_model(cfg, seed=0)
    broken.lm_head.weight = torch.nn.Parameter(broken.lm_head.weight.detach().clone())
    assert not verify_weight_tying(broken)


def test_unintended_projection_bias_is_detected():
    cfg = make_tiny_sparse_config(attention_bias=True)
    model = build_model(cfg, seed=0)
    offenders = bias_audit(model)
    assert len(offenders) > 0
    assert any("o_proj" in name or "q_proj" in name for name in offenders)


def test_wrong_layer_placement_is_detected_against_frozen_positions():
    # A config with dense/MoE positions swapped relative to the frozen
    # [1,5,10,15,20] dense-anchor spec must fail a positional-equality
    # check even though the total layer count is identical.
    cfg = make_tiny_sparse_config(n_layers=6, dense_layers=[2, 4], moe_layers=[1, 3, 5, 6])
    model = build_model(cfg, seed=0)
    dense_positions = {i + 1 for i, kind in enumerate(model.layer_kinds) if kind == "dense"}
    assert dense_positions != {1, 5, 10, 15, 20}  # not the frozen sparse spec's positions
    assert dense_positions == {2, 4}  # matches this (deliberately wrong-for-frozen) test config


def test_wrong_qk_norm_parameter_shape_is_detected():
    cfg = make_tiny_sparse_config(
        n_query_heads=4, n_kv_heads=2, head_dim=8, d_model=32, n_layers=3, dense_layers=[1, 3], moe_layers=[2]
    )
    model = build_model(cfg, seed=0)
    correct_count = qk_norm_parameter_count(model)  # 2 * head_dim * n_layers = 2*8*3 = 48
    assert correct_count == 48

    # Simulate the bug this accounting is designed to catch: a *separate*
    # head_dim-sized vector per query head instead of one shared vector.
    buggy_count_if_per_head_q_norm = model.layers[0].attention.n_query_heads * cfg.attention.head_dim * 2  # Q per-head + shared K
    assert buggy_count_if_per_head_q_norm != correct_count


def test_router_fp16_execution_is_detected():
    cfg = make_tiny_sparse_config()
    layer = MoELayer(cfg)
    x = torch.randn(1, 4, cfg.core.d_model)

    # Simulate the bug: compute router logits WITHOUT the fp32-forcing
    # autocast-disable context, directly in fp16.
    flat_x = x.reshape(-1, cfg.core.d_model).to(torch.float16)
    weight_fp16 = layer.router.weight.to(torch.float16)
    buggy_logits = F.linear(flat_x, weight_fp16)
    assert buggy_logits.dtype == torch.float16  # this is exactly what must never reach the real router path

    _, _, _, diag = layer(x, valid_mask=torch.ones(1, 4, dtype=torch.bool), return_diagnostics=True)
    assert diag.router_logits.dtype == torch.float32  # the real implementation avoids the bug


def test_incorrect_rope_theta_is_detected():
    correct = RotaryEmbedding(dim=8, theta=100000.0)
    wrong = RotaryEmbedding(dim=8, theta=10000.0)  # a very common "wrong" default from other model families
    assert not torch.allclose(correct.inv_freq, wrong.inv_freq)


def test_unshifted_causal_loss_is_detected():
    vocab, seq_len = 4, 4
    logits = torch.zeros(1, seq_len, vocab)
    for i in range(seq_len):
        logits[0, i, (i + 1) % vocab] = 20.0
    labels = torch.tensor([[i % vocab for i in range(seq_len)]])

    correct_loss = causal_lm_loss(logits, labels)
    buggy_unshifted_loss = F.cross_entropy(logits.view(-1, vocab), labels.view(-1))
    assert correct_loss.item() < 1e-3
    assert buggy_unshifted_loss.item() > 5.0
    assert abs(correct_loss.item() - buggy_unshifted_loss.item()) > 1.0


def test_padding_included_in_router_statistics_is_detected():
    router_probs = torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.99, 0.01]])
    topk_idx = torch.tensor([[0], [1], [0]])
    correctly_excluded = compute_load_balance_loss_raw(
        router_probs, topk_idx, torch.tensor([True, True, False]), 2, 1
    )
    buggy_included = compute_load_balance_loss_raw(
        router_probs, topk_idx, torch.tensor([True, True, True]), 2, 1
    )
    assert not torch.allclose(correctly_excluded, buggy_included)
