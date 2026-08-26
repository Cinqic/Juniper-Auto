"""MoE router + dropless top-2 dispatch: expert counts, unique selection,
renormalized combination weights, no hidden averaging, dropless assignment
counting, router FP32 precision, padding exclusion, and gradient flow."""

from __future__ import annotations

import torch
import pytest

from juniper_auto.model.moe import MoELayer
from tests.model_fixtures import make_tiny_sparse_config


def _layer(n_routed=4, top_k=2, d_model=8, expert_ffn_dim=8):
    # MoELayer itself never touches attention, but ArchitectureConfig
    # validation still requires n_query_heads * head_dim == d_model, so a
    # single-head config is used to keep d_model freely overridable here.
    cfg = make_tiny_sparse_config(
        n_routed_experts=n_routed,
        top_k=top_k,
        d_model=d_model,
        expert_ffn_dim=expert_ffn_dim,
        n_query_heads=1,
        n_kv_heads=1,
        head_dim=d_model,
    )
    return MoELayer(cfg), cfg


def _call(layer, x_2d, valid_mask_1d=None, **kwargs):
    """MoELayer's real forward contract is [batch, seq_len, d_model]
    (matching how MoEBlock always calls it); tests work with flat
    [n_tokens, d_model] tensors for readability, so this wraps a batch=1
    dimension on the way in and strips it on the way out. `diag` fields are
    already flattened across batch*seq inside MoELayer, so they need no
    reshaping here."""
    x_3d = x_2d.unsqueeze(0)
    valid_3d = valid_mask_1d.unsqueeze(0) if valid_mask_1d is not None else None
    out_3d, lb, z, diag = layer(x_3d, valid_mask=valid_3d, **kwargs)
    return out_3d.squeeze(0), lb, z, diag


def test_expert_counts_match_config():
    layer, cfg = _layer(n_routed=4)
    assert len(layer.routed_experts) == 4
    assert layer.n_shared_experts == 1
    assert isinstance(layer.shared_expert, torch.nn.Module)


def test_moe_rejects_invalid_valid_mask_shape():
    layer, cfg = _layer()
    x = torch.randn(2, 5, cfg.core.d_model)
    with pytest.raises(ValueError, match="valid_mask"):
        layer(x, valid_mask=torch.ones(1, 5, dtype=torch.bool))


def test_every_token_gets_exactly_top_k_unique_routed_experts():
    layer, cfg = _layer(n_routed=6, top_k=2)
    torch.manual_seed(0)
    x = torch.randn(9, cfg.core.d_model)
    _, _, _, diag = _call(layer, x, torch.ones(9, dtype=torch.bool), return_diagnostics=True)
    for t in range(9):
        e0, e1 = diag.topk_idx[t].tolist()
        assert e0 != e1


def test_topk_weights_renormalize_to_one():
    layer, cfg = _layer(n_routed=8, top_k=2)
    torch.manual_seed(1)
    x = torch.randn(6, cfg.core.d_model)
    _, _, _, diag = _call(layer, x, torch.ones(6, dtype=torch.bool), return_diagnostics=True)
    sums = diag.topk_weights.sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones(6), atol=1e-5, rtol=1e-5)


def test_dropless_assignment_count_equals_valid_tokens_times_top_k():
    layer, cfg = _layer(n_routed=5, top_k=2)
    torch.manual_seed(2)
    x = torch.randn(11, cfg.core.d_model)
    valid_mask = torch.tensor([True] * 7 + [False] * 4)
    _, _, _, diag = _call(layer, x, valid_mask, return_diagnostics=True)
    assert diag.assignment_counts_per_expert.sum().item() == 7 * 2


def test_padding_excluded_from_dispatch_statistics_but_still_finite_output():
    layer, cfg = _layer(n_routed=4, top_k=2)
    torch.manual_seed(3)
    x = torch.randn(6, cfg.core.d_model)
    valid_mask = torch.tensor([True, True, True, False, False, False])
    out_a, lb_a, z_a, diag_a = _call(layer, x, valid_mask, return_diagnostics=True)

    x_mutated = x.clone()
    x_mutated[3:] = torch.randn_like(x_mutated[3:]) * 50.0
    out_b, lb_b, z_b, diag_b = _call(layer, x_mutated, valid_mask, return_diagnostics=True)

    # Padding tokens are still routed (MoE has no cross-token mixing, so
    # this cannot corrupt valid tokens' outputs) but must not move the
    # aux-loss statistics computed only over valid tokens.
    torch.testing.assert_close(lb_a, lb_b)
    torch.testing.assert_close(z_a, z_b)
    assert torch.equal(diag_a.assignment_counts_per_expert, diag_b.assignment_counts_per_expert)
    assert torch.isfinite(out_a).all() and torch.isfinite(out_b).all()
    torch.testing.assert_close(out_a[:3], out_b[:3])


def test_shared_plus_topk_routed_combination_with_no_hidden_averaging():
    # Replace every expert with a deterministic constant-output stub so the
    # *combination formula* can be checked exactly, independent of routing
    # selection (covered separately). A hidden division by (top_k + 1) or
    # by 2/3, or a missing shared-expert term, would fail this exactly.
    layer, cfg = _layer(n_routed=4, top_k=2, d_model=4, expert_ffn_dim=4)

    def make_stub(value):
        def stub(x):
            return torch.full((x.shape[0], cfg.core.d_model), value)

        return stub

    layer.shared_expert.forward = make_stub(100.0)
    for i, expert in enumerate(layer.routed_experts):
        expert.forward = make_stub(float(i + 1))

    torch.manual_seed(4)
    x = torch.randn(5, cfg.core.d_model)
    out, _, _, diag = _call(layer, x, torch.ones(5, dtype=torch.bool), return_diagnostics=True)

    for t in range(5):
        e0, e1 = diag.topk_idx[t].tolist()
        w0, w1 = diag.topk_weights[t].tolist()
        expected_value = 100.0 + w0 * (e0 + 1) + w1 * (e1 + 1)
        torch.testing.assert_close(
            out[t], torch.full((cfg.core.d_model,), expected_value), atol=1e-4, rtol=1e-4
        )


def test_shared_expert_is_never_gated():
    # With a shared-expert stub returning a nonzero constant regardless of
    # input, every token's output must include that constant term
    # unconditionally -- a gated implementation could zero it for some tokens.
    layer, cfg = _layer(n_routed=3, top_k=1, d_model=4, expert_ffn_dim=4)
    layer.shared_expert.forward = lambda x: torch.full((x.shape[0], cfg.core.d_model), 7.0)
    for expert in layer.routed_experts:
        expert.forward = lambda x: torch.zeros(x.shape[0], cfg.core.d_model)

    x = torch.randn(4, cfg.core.d_model)
    out, _, _, _ = _call(layer, x, torch.ones(4, dtype=torch.bool))
    torch.testing.assert_close(out, torch.full((4, cfg.core.d_model), 7.0), atol=1e-5, rtol=1e-5)


def test_router_logits_and_softmax_are_fp32_under_fp16_input():
    layer, cfg = _layer()
    x_fp16 = torch.randn(4, cfg.core.d_model).to(torch.float16)
    layer_fp16 = layer.to(torch.float16)
    _, _, _, diag = _call(layer_fp16, x_fp16, torch.ones(4, dtype=torch.bool), return_diagnostics=True)
    assert diag.router_logits.dtype == torch.float32
    assert diag.router_probs.dtype == torch.float32


def test_router_logits_and_softmax_are_fp32_under_cpu_autocast_bf16():
    layer, cfg = _layer()
    x = torch.randn(4, cfg.core.d_model)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
        _, _, _, diag = _call(layer, x, torch.ones(4, dtype=torch.bool), return_diagnostics=True)
    assert diag.router_logits.dtype == torch.float32
    assert diag.router_probs.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA mixed-precision hardware")
def test_router_fp32_path_and_gradients_under_real_cuda_fp16_autocast():
    layer, cfg = _layer()
    layer = layer.to("cuda")
    x = torch.randn(4, cfg.core.d_model, device="cuda", requires_grad=True)
    valid = torch.ones(4, dtype=torch.bool, device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        out, lb, z, diag = _call(layer, x, valid, return_diagnostics=True)
        loss = out.float().square().mean() + lb + z
    assert diag.router_logits.dtype == torch.float32
    assert diag.router_probs.dtype == torch.float32
    loss.backward()
    assert layer.router.weight.grad is not None
    assert torch.isfinite(layer.router.weight.grad).all()


def test_router_softmax_probabilities_sum_to_one():
    layer, cfg = _layer(n_routed=6)
    x = torch.randn(5, cfg.core.d_model)
    _, _, _, diag = _call(layer, x, torch.ones(5, dtype=torch.bool), return_diagnostics=True)
    torch.testing.assert_close(diag.router_probs.sum(dim=-1), torch.ones(5), atol=1e-5, rtol=1e-5)


def test_gradients_reach_router_shared_expert_and_selected_experts_only():
    layer, cfg = _layer(n_routed=8, top_k=2, d_model=6, expert_ffn_dim=6)
    x = torch.randn(4, cfg.core.d_model, requires_grad=True)
    out, lb, z, diag = _call(layer, x, torch.ones(4, dtype=torch.bool), return_diagnostics=True)
    loss = out.sum() + lb + z
    loss.backward()

    assert layer.router.weight.grad is not None
    assert torch.isfinite(layer.router.weight.grad).all()
    for p in layer.shared_expert.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()

    selected = set(diag.topk_idx.reshape(-1).tolist())
    for expert_id, expert in enumerate(layer.routed_experts):
        grads = [p.grad for p in expert.parameters()]
        if expert_id in selected:
            assert all(g is not None and torch.isfinite(g).all() for g in grads)
        # Experts never selected in this batch are not required to have a
        # gradient at all -- their forward was never called.


def test_token_order_is_preserved_in_reconstruction():
    layer, cfg = _layer(n_routed=4, top_k=2, d_model=6, expert_ffn_dim=6)
    torch.manual_seed(5)
    x = torch.randn(6, cfg.core.d_model)
    valid_mask = torch.ones(6, dtype=torch.bool)
    out_full, _, _, _ = _call(layer, x, valid_mask)

    perm = torch.tensor([3, 1, 4, 0, 5, 2])
    out_permuted_input, _, _, _ = _call(layer, x[perm], valid_mask[perm])
    torch.testing.assert_close(out_full[perm], out_permuted_input, atol=1e-5, rtol=1e-5)
