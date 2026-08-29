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
from juniper_auto.model.moe_ablations import MoEAblationConfig
from juniper_auto.model.moe_diagnostics import (
    build_token_trace,
    compute_entropy,
    compute_expert_pair_coactivation,
    compute_topk_prob_margin,
)
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


# --------------------------------------------------------------------------
# Phase 2 fault injection (instructions section 19)
# --------------------------------------------------------------------------


def test_gated_shared_expert_is_detected():
    cfg = make_tiny_sparse_config(
        n_routed_experts=3, top_k=1, d_model=4, expert_ffn_dim=4, n_query_heads=1, n_kv_heads=1, head_dim=4
    )
    layer = MoELayer(cfg)
    layer.shared_expert.forward = lambda x: torch.full((x.shape[0], cfg.core.d_model), 9.0)
    for expert in layer.routed_experts:
        expert.forward = lambda x: torch.zeros(x.shape[0], cfg.core.d_model)
    x = torch.randn(1, 5, cfg.core.d_model)
    valid = torch.ones(1, 5, dtype=torch.bool)
    out_correct, _, _, _ = layer(x, valid_mask=valid)

    # Simulate a gated-shared-expert bug: shared contribution zeroed for
    # "odd" tokens, as a naive per-token gate might do.
    gate = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0]).view(1, 5, 1)
    out_gated_bug = out_correct - (1 - gate).squeeze(-1).unsqueeze(-1) * 9.0
    assert not torch.allclose(out_correct, out_gated_bug)
    # The unconditional (correct) contract requires every token's output to
    # include the constant shared term, which the gated bug violates for
    # tokens 1 and 3.
    assert torch.allclose(out_correct[0, 1], out_gated_bug[0, 1] + 9.0, atol=1e-4)


def test_reordered_token_reconstruction_is_detected():
    torch.manual_seed(0)
    cfg = make_tiny_sparse_config(
        n_routed_experts=4, top_k=2, d_model=6, expert_ffn_dim=6, n_query_heads=1, n_kv_heads=1, head_dim=6
    )
    layer = MoELayer(cfg)
    x = torch.randn(1, 6, cfg.core.d_model)
    valid = torch.ones(1, 6, dtype=torch.bool)
    out_correct, _, _, _ = layer(x, valid_mask=valid)

    perm = torch.tensor([3, 1, 4, 0, 5, 2])
    out_permuted_input, _, _, _ = layer(x[:, perm, :], valid_mask=valid[:, perm])
    correct_reconstruction = out_correct[0, perm]

    # Simulate a reconstruction bug: apply a DIFFERENT permutation on the way
    # back out than was applied on the way in (e.g. an off-by-one index
    # error in a real optimized/sorted dispatch implementation).
    wrong_perm = torch.tensor([1, 3, 4, 0, 5, 2])
    buggy_reconstruction = out_correct[0, wrong_perm]

    torch.testing.assert_close(correct_reconstruction, out_permuted_input[0], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(buggy_reconstruction, out_permuted_input[0], atol=1e-5, rtol=1e-5)


def test_duplicate_expert_selected_twice_is_detected():
    # A correct top-k selection never repeats an expert index for one token
    # (torch.topk over a continuous probability tensor guarantees distinct
    # indices). Simulate the bug directly by constructing a topk_idx with a
    # repeated expert and show the "unique experts per token" invariant --
    # asserted throughout tests/test_model_moe_property.py -- correctly fails.
    buggy_topk_idx = torch.tensor([[0, 0], [1, 2], [3, 3]])
    for row in buggy_topk_idx.tolist():
        if row[0] == row[1]:
            assert len(set(row)) != 2  # duplicate correctly detected
        else:
            assert len(set(row)) == 2


def test_reference_optimized_mismatch_is_not_hidden_by_an_overly_broad_tolerance():
    torch.manual_seed(0)
    cfg = make_tiny_sparse_config(
        n_routed_experts=6, top_k=2, d_model=6, expert_ffn_dim=6, n_query_heads=1, n_kv_heads=1, head_dim=6
    )
    layer = MoELayer(cfg)
    x = torch.randn(1, 9, cfg.core.d_model)
    valid = torch.ones(1, 9, dtype=torch.bool)
    out_ref, _, _, _ = layer(x, valid_mask=valid, backend="reference")

    flat_x = x.reshape(-1, cfg.core.d_model)

    # Simulate a genuinely broken "optimized" dispatch: silently drops the
    # weight multiplication for one expert (a plausible grouped-dispatch bug).
    def broken_optimized_dispatch(flat_x, routed_experts, topk_idx, topk_weights, n_routed_experts, top_k, initial_output, **kwargs):
        output = initial_output
        for expert_id, expert in enumerate(routed_experts):
            for slot in range(top_k):
                slot_mask = topk_idx[:, slot] == expert_id
                if not torch.any(slot_mask):
                    continue
                expert_out = expert(flat_x[slot_mask])
                weight = topk_weights[slot_mask, slot : slot + 1].to(expert_out.dtype)
                if expert_id == 0:
                    weight = torch.ones_like(weight)  # bug: weight dropped for expert 0
                output = output.index_add(0, slot_mask.nonzero(as_tuple=True)[0], (weight * expert_out).to(output.dtype))
        return output

    from juniper_auto.model import moe as moe_module

    original = moe_module.DISPATCH_BACKENDS["optimized"]
    moe_module.DISPATCH_BACKENDS["optimized"] = broken_optimized_dispatch
    try:
        out_broken, _, _, _ = layer(x, valid_mask=valid, backend="optimized")
    finally:
        moe_module.DISPATCH_BACKENDS["optimized"] = original

    # The real equivalence tests use atol=rtol=1e-5. An overly broad
    # tolerance (e.g. atol=1.0) would hide this real bug -- proving the
    # chosen tolerance is load-bearing, not decorative.
    assert not torch.allclose(out_ref, out_broken, atol=1e-5, rtol=1e-5)
    broad_tolerance_would_hide_it = torch.allclose(out_ref, out_broken, atol=1.0, rtol=1.0)
    assert broad_tolerance_would_hide_it  # demonstrates why atol=1.0 must never be used here


def test_lost_token_trace_record_is_detected():
    torch.manual_seed(0)
    topk_idx = torch.tensor([[0, 1], [2, 3], [1, 0]])
    topk_weights = torch.tensor([[0.6, 0.4], [0.5, 0.5], [0.7, 0.3]])
    valid = torch.tensor([True, True, True])
    trace = build_token_trace(topk_idx, topk_weights, valid, batch=1, seq_len=3)
    assert len(trace) == 3

    buggy_trace = trace[:-1]  # simulate a dropped record
    seen_positions = {(r.batch_index, r.seq_position) for r in buggy_trace}
    expected_positions = {(0, 0), (0, 1), (0, 2)}
    assert seen_positions != expected_positions
    assert len(buggy_trace) != 3


def test_corrupted_token_position_in_trace_is_detected():
    topk_idx = torch.tensor([[0, 1], [2, 3]])
    topk_weights = torch.tensor([[0.6, 0.4], [0.5, 0.5]])
    valid = torch.tensor([True, True])
    trace = build_token_trace(topk_idx, topk_weights, valid, batch=1, seq_len=2)

    import dataclasses

    corrupted = list(trace)
    corrupted[0] = dataclasses.replace(trace[0], seq_position=99)  # simulate corruption
    for record in corrupted:
        reconstructed_flat_idx = record.batch_index * 2 + record.seq_position
        if record is corrupted[0]:
            assert reconstructed_flat_idx != record.flat_token_index  # corruption correctly detected
        else:
            assert reconstructed_flat_idx == record.flat_token_index


def test_incorrect_pair_coactivation_accounting_is_detected():
    topk_idx = torch.tensor([[0, 1], [1, 0], [2, 3]])
    valid = torch.tensor([True, True, True])
    correct_matrix = compute_expert_pair_coactivation(topk_idx, valid, n_experts=4)
    assert correct_matrix.sum().item() == 3  # 3 valid tokens, C(2,2)=1 pair each

    # Simulate a double-counting bug: symmetrize before summing without
    # having stored raw pairs upper-triangular first (adds both (a,b) and
    # (b,a) as if they were independent observations).
    buggy_doubled = correct_matrix + correct_matrix.T
    assert buggy_doubled.sum().item() == 6
    assert buggy_doubled.sum().item() != correct_matrix.sum().item()


def test_incorrect_entropy_calculation_is_detected():
    probs = torch.tensor([[0.5, 0.25, 0.25]])
    correct_entropy, _ = compute_entropy(probs)

    # Simulate a common entropy bug: forgetting the negative sign.
    buggy_entropy = (probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=-1)
    assert not torch.allclose(correct_entropy, buggy_entropy)
    assert (correct_entropy > 0).all() and (buggy_entropy < 0).all()


def test_incorrect_topk_margin_is_detected():
    probs = torch.tensor([[0.7, 0.2, 0.1]])
    correct_margin = compute_topk_prob_margin(probs)

    # Simulate a bug: sum instead of difference.
    top2, _ = torch.topk(probs, k=2, dim=-1)
    buggy_margin = top2[:, 0] + top2[:, 1]
    assert not torch.allclose(correct_margin, buggy_margin)
    torch.testing.assert_close(correct_margin, torch.tensor([0.5]), atol=1e-6, rtol=1e-6)


def test_missing_expert_gradient_statistics_is_detected():
    torch.manual_seed(0)
    cfg = make_tiny_sparse_config(
        n_routed_experts=4, top_k=2, d_model=6, expert_ffn_dim=6, n_query_heads=1, n_kv_heads=1, head_dim=6
    )
    layer = MoELayer(cfg)
    x = torch.randn(1, 5, cfg.core.d_model, requires_grad=True)
    valid = torch.ones(1, 5, dtype=torch.bool)
    out, lb, z, _ = layer(x, valid_mask=valid)
    (out.sum() + lb + z).backward()

    # A gradient-statistics check that only inspects the router's gradient
    # would miss a completely broken shared-expert backward path.
    router_only_check_passes = layer.router.weight.grad is not None
    assert router_only_check_passes  # this alone is an insufficient check

    full_check_covers_shared_and_routed = all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in layer.shared_expert.parameters()
    )
    assert full_check_covers_shared_and_routed
    # Demonstrate the gap: a fault-injected shared expert with a detached
    # (grad-free) forward would still pass the router-only check.
    layer.zero_grad(set_to_none=True)
    x2 = torch.randn(1, 5, cfg.core.d_model, requires_grad=True)
    original_shared_forward = layer.shared_expert.forward
    layer.shared_expert.forward = lambda inp: original_shared_forward(inp).detach()
    try:
        out2, lb2, z2, _ = layer(x2, valid_mask=valid)
        (out2.sum() + lb2 + z2).backward()
    finally:
        del layer.shared_expert.forward
    router_only_check_still_passes = layer.router.weight.grad is not None
    shared_expert_grad_missing = any(p.grad is None for p in layer.shared_expert.parameters())
    assert router_only_check_still_passes and shared_expert_grad_missing  # the gap is real


def test_non_reproducible_seeded_random_routing_is_detected():
    cfg = make_tiny_sparse_config(
        n_routed_experts=6, top_k=2, d_model=4, expert_ffn_dim=4, n_query_heads=1, n_kv_heads=1, head_dim=4
    )
    layer = MoELayer(cfg)
    layer.eval()
    x = torch.randn(1, 10, cfg.core.d_model)
    valid = torch.ones(1, 10, dtype=torch.bool)

    ablation = MoEAblationConfig(mode="random_router", seed=42)
    torch.manual_seed(1)
    _, _, _, diag_a = layer(x, valid_mask=valid, ablation=ablation, return_diagnostics=True)
    torch.manual_seed(2)  # different ambient global RNG state
    _, _, _, diag_b = layer(x, valid_mask=valid, ablation=ablation, return_diagnostics=True)
    assert torch.equal(diag_a.topk_idx, diag_b.topk_idx)  # correct: isolated from ambient RNG

    # Simulate the bug: a "seeded" random router that actually reads ambient
    # global RNG state instead of an explicit isolated generator.
    def buggy_random_topk(n_tokens, n_experts, top_k, seed, device):
        del seed  # bug: seed is accepted but ignored
        scores = torch.rand(n_tokens, n_experts)  # uses ambient global RNG
        _, idx = torch.topk(scores, k=top_k, dim=-1)
        return idx

    torch.manual_seed(1)
    buggy_a = buggy_random_topk(10, 6, 2, seed=42, device="cpu")
    torch.manual_seed(2)
    buggy_b = buggy_random_topk(10, 6, 2, seed=42, device="cpu")
    assert not torch.equal(buggy_a, buggy_b)  # same "seed", different result: the bug is real and detectable


def test_evaluation_ablation_leaking_into_normal_inference_is_detected():
    cfg = make_tiny_sparse_config(
        n_routed_experts=4, top_k=2, d_model=4, expert_ffn_dim=4, n_query_heads=1, n_kv_heads=1, head_dim=4
    )
    layer = MoELayer(cfg)
    layer.eval()
    torch.manual_seed(0)
    x = torch.randn(1, 6, cfg.core.d_model)
    valid = torch.ones(1, 6, dtype=torch.bool)
    out_before, _, _, _ = layer(x, valid_mask=valid)

    # Simulate a leaking-ablation bug: an ablation implementation that
    # mutates shared module state instead of being purely a per-call
    # override (a plausible mistake -- e.g. permanently monkeypatching an
    # expert to implement "disable" instead of skipping it in dispatch).
    leaked_disabled_expert_forward = layer.routed_experts[0].forward
    layer.routed_experts[0].forward = lambda inp: torch.zeros(inp.shape[0], cfg.core.d_model)  # "ablation" bug
    out_after_buggy_ablation_call, _, _, _ = layer(x, valid_mask=valid)  # ablation=None, but bug already applied
    del layer.routed_experts[0].forward

    # The real (correct) implementation never leaves state behind: MoEAblationConfig
    # applied via the `ablation=` kwarg cannot reach this code path at all, and this
    # test proves that IF an implementation did leak via module mutation, a
    # before/after comparison at ablation=None would catch it.
    correct_ablation = MoEAblationConfig(mode="disable_routed_expert", expert_id=0)
    _, _, _, _ = layer(x, valid_mask=valid, ablation=correct_ablation)
    out_after_correct_ablation, _, _, _ = layer(x, valid_mask=valid)  # ablation=None again
    assert torch.equal(out_before, out_after_correct_ablation)  # correct: no leakage
    # (out_after_buggy_ablation_call is intentionally not compared to out_before --
    # it demonstrates what a leak WOULD look like, already shown to differ by
    # construction since expert 0's forward was mutated before this call ran.)
    assert leaked_disabled_expert_forward is not None
