"""Reference-vs-optimized MoE dispatch equivalence (Phase 2 instructions
section 5), and a golden comparison against the approved Phase 1 commit's
`moe.py` (section 4) proving the Phase 2 module split (routing.py /
moe_dispatch.py / moe_ablations.py / moe_diagnostics.py) did not silently
change the reference dispatch's semantics."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from juniper_auto.model.moe import MoELayer
from tests.model_fixtures import make_tiny_sparse_config

REPO_ROOT = Path(__file__).resolve().parent.parent
PHASE_1_TAG = "phase-1-architecture"
PHASE_1_APPROVED_COMMIT = "073acf46e04241ed35d00bc4b4c29ac463ee744d"


def _load_phase1_golden_moe_module():
    resolved = subprocess.run(
        ["git", "rev-list", "-n", "1", PHASE_1_TAG],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if resolved.returncode != 0 or resolved.stdout.strip() != PHASE_1_APPROVED_COMMIT:
        raise RuntimeError(
            f"required golden tag {PHASE_1_TAG!r} must resolve to {PHASE_1_APPROVED_COMMIT}, "
            f"got {resolved.stdout.strip() or resolved.stderr.strip() or 'unavailable'}"
        )
    result = subprocess.run(
        ["git", "show", f"{PHASE_1_TAG}:juniper_auto/model/moe.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"required golden evidence is unavailable: cannot read "
            f"{PHASE_1_TAG}:juniper_auto/model/moe.py from git: {result.stderr}"
        )
    source = result.stdout
    spec = importlib.util.spec_from_loader("_phase1_golden_moe", loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_phase1_golden_moe"] = module
    try:
        exec(compile(source, "<phase-1-architecture:juniper_auto/model/moe.py>", "exec"), module.__dict__)
    finally:
        del sys.modules["_phase1_golden_moe"]
    return module


@pytest.fixture(scope="module")
def phase1_golden_moe():
    return _load_phase1_golden_moe_module()


def test_every_required_ci_workflow_fetches_the_golden_phase1_tag(repo_root):
    for name in ("phase-0-validation.yml", "phase-1-validation.yml", "phase-2-validation.yml"):
        workflow = yaml.load(
            (repo_root / ".github" / "workflows" / name).read_text(),
            Loader=yaml.BaseLoader,
        )
        checkout_steps = [
            step
            for job in workflow["jobs"].values()
            for step in job["steps"]
            if step.get("uses", "").startswith("actions/checkout@")
        ]
        assert len(checkout_steps) == 1, f"{name}: expected exactly one checkout step"
        checkout = checkout_steps[0]
        assert checkout.get("with", {}).get("fetch-depth") == "0", name
        assert checkout.get("with", {}).get("fetch-tags") == "true", name


def _paired_layers(phase1_golden_moe, **cfg_overrides):
    d_model = cfg_overrides.get("d_model", 32)
    cfg_overrides.setdefault("n_query_heads", 1)
    cfg_overrides.setdefault("n_kv_heads", 1)
    cfg_overrides.setdefault("head_dim", d_model)
    cfg = make_tiny_sparse_config(**cfg_overrides)
    new_layer = MoELayer(cfg)
    golden_layer = phase1_golden_moe.MoELayer(cfg)
    golden_layer.load_state_dict(new_layer.state_dict())
    return new_layer, golden_layer, cfg


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
@pytest.mark.parametrize(
    "n_tokens,n_experts,top_k,padding",
    [
        (9, 4, 2, "none"),
        (11, 6, 2, "trailing"),
        (11, 6, 2, "leading"),
        (10, 5, 3, "scattered"),
        (7, 8, 1, "none"),
    ],
)
def test_refactored_reference_dispatch_matches_phase1_golden_bit_for_bit(
    phase1_golden_moe, seed, n_tokens, n_experts, top_k, padding
):
    torch.manual_seed(seed)
    new_layer, golden_layer, cfg = _paired_layers(
        phase1_golden_moe, n_routed_experts=n_experts, top_k=top_k, d_model=8, expert_ffn_dim=8
    )
    x = torch.randn(1, n_tokens, cfg.core.d_model)
    valid = torch.ones(1, n_tokens, dtype=torch.bool)
    if padding == "trailing":
        valid[0, -2:] = False
    elif padding == "leading":
        valid[0, :2] = False
    elif padding == "scattered":
        valid[0, ::3] = False

    out_new, lb_new, z_new, diag_new = new_layer(x, valid_mask=valid, return_diagnostics=True)
    out_gold, lb_gold, z_gold, diag_gold = golden_layer(x, valid_mask=valid, return_diagnostics=True)

    # Phase 2 independent review tightened padding semantics: valid-token
    # behavior remains the Phase 1 golden oracle, while padding positions no
    # longer execute experts and therefore have a zero MoE contribution.
    flat_valid = valid.unsqueeze(-1).expand_as(out_new)
    if valid.all():
        assert torch.equal(out_new, out_gold)
    else:
        # Compacting valid rows changes GEMM batching and can change the last
        # few floating-point bits without changing valid-token semantics.
        torch.testing.assert_close(out_new[flat_valid], out_gold[flat_valid], atol=1e-6, rtol=1e-6)
    if not valid.all():
        assert torch.equal(out_new[~flat_valid], torch.zeros_like(out_new[~flat_valid]))
    assert torch.equal(lb_new, lb_gold)
    assert torch.equal(z_new, z_gold)
    assert torch.equal(diag_new.router_logits, diag_gold.router_logits)
    assert torch.equal(diag_new.router_probs, diag_gold.router_probs)
    assert torch.equal(diag_new.topk_idx, diag_gold.topk_idx)
    assert torch.equal(diag_new.topk_weights, diag_gold.topk_weights)
    assert torch.equal(diag_new.assignment_counts_per_expert, diag_gold.assignment_counts_per_expert)


def test_refactored_reference_dispatch_matches_phase1_golden_under_gradients(phase1_golden_moe):
    torch.manual_seed(7)
    new_layer, golden_layer, cfg = _paired_layers(
        phase1_golden_moe, n_routed_experts=6, top_k=2, d_model=6, expert_ffn_dim=6
    )
    x = torch.randn(2, 5, cfg.core.d_model)
    valid = torch.ones(2, 5, dtype=torch.bool)
    valid[1, -1] = False

    x_new = x.clone().requires_grad_(True)
    x_gold = x.clone().requires_grad_(True)

    out_new, lb_new, z_new, _ = new_layer(x_new, valid_mask=valid)
    (out_new[valid].sum() + lb_new + z_new).backward()

    out_gold, lb_gold, z_gold, _ = golden_layer(x_gold, valid_mask=valid)
    (out_gold[valid].sum() + lb_gold + z_gold).backward()

    flat_valid = valid.unsqueeze(-1).expand_as(out_new)
    torch.testing.assert_close(out_new[flat_valid], out_gold[flat_valid], atol=1e-6, rtol=1e-6)
    grad_valid = valid.unsqueeze(-1).expand_as(x_new.grad)
    torch.testing.assert_close(x_new.grad[grad_valid], x_gold.grad[grad_valid], atol=1e-6, rtol=1e-6)
    assert torch.equal(x_new.grad[~grad_valid], torch.zeros_like(x_new.grad[~grad_valid]))
    torch.testing.assert_close(
        new_layer.router.weight.grad, golden_layer.router.weight.grad, atol=1e-6, rtol=1e-6
    )
    for p_new, p_gold in zip(new_layer.shared_expert.parameters(), golden_layer.shared_expert.parameters()):
        torch.testing.assert_close(p_new.grad, p_gold.grad, atol=1e-6, rtol=1e-6)


# --------------------------------------------------------------------------
# Reference vs optimized (Phase 2) numerical equivalence
# --------------------------------------------------------------------------

_EQUIVALENCE_TOLERANCE = dict(atol=1e-5, rtol=1e-5)


def _layer(backend="reference", **overrides):
    overrides.setdefault("n_routed_experts", 8)
    overrides.setdefault("top_k", 2)
    overrides.setdefault("d_model", 8)
    overrides.setdefault("expert_ffn_dim", 8)
    d_model = overrides["d_model"]
    overrides.setdefault("n_query_heads", 1)
    overrides.setdefault("n_kv_heads", 1)
    overrides.setdefault("head_dim", d_model)
    cfg = make_tiny_sparse_config(**overrides)
    return MoELayer(cfg, backend=backend), cfg


@pytest.mark.parametrize("seed", list(range(8)))
@pytest.mark.parametrize(
    "batch,seq_len,n_experts,top_k,padding",
    [
        (1, 9, 8, 2, "none"),
        (2, 11, 6, 2, "trailing"),
        (3, 7, 5, 3, "leading"),
        (2, 13, 8, 2, "scattered"),
        (1, 6, 4, 1, "none"),
        (4, 5, 8, 2, "almost-all-padding"),
    ],
)
def test_reference_and_optimized_forward_agree(seed, batch, seq_len, n_experts, top_k, padding):
    torch.manual_seed(seed)
    layer, cfg = _layer(n_routed_experts=n_experts, top_k=top_k, d_model=8, expert_ffn_dim=8)
    x = torch.randn(batch, seq_len, cfg.core.d_model)
    valid = torch.ones(batch, seq_len, dtype=torch.bool)
    if padding == "trailing":
        valid[:, -2:] = False
    elif padding == "leading":
        valid[:, :2] = False
    elif padding == "scattered":
        valid[..., ::3] = False
    elif padding == "almost-all-padding":
        valid[:] = False
        valid[0, 0] = True

    out_ref, lb_ref, z_ref, diag_ref = layer(x, valid_mask=valid, backend="reference", return_diagnostics=True)
    out_opt, lb_opt, z_opt, diag_opt = layer(x, valid_mask=valid, backend="optimized", return_diagnostics=True)

    # Routing itself is shared code (routing.py) -- these must match exactly,
    # not just within tolerance, regardless of dispatch backend.
    assert torch.equal(diag_ref.router_logits, diag_opt.router_logits)
    assert torch.equal(diag_ref.topk_idx, diag_opt.topk_idx)
    assert torch.equal(diag_ref.topk_weights, diag_opt.topk_weights)
    assert torch.equal(diag_ref.assignment_counts_per_expert, diag_opt.assignment_counts_per_expert)
    assert torch.equal(lb_ref, lb_opt)
    assert torch.equal(z_ref, z_opt)

    # Only the final combination (execution/summation order) can legitimately
    # differ between dispatch backends.
    torch.testing.assert_close(out_ref, out_opt, **_EQUIVALENCE_TOLERANCE)
    torch.testing.assert_close(
        out_ref[valid.reshape(batch, seq_len)], out_opt[valid.reshape(batch, seq_len)], **_EQUIVALENCE_TOLERANCE
    )


def test_reference_and_optimized_backward_agree():
    torch.manual_seed(11)
    layer, cfg = _layer(n_routed_experts=8, top_k=2, d_model=8, expert_ffn_dim=8)
    x = torch.randn(2, 9, cfg.core.d_model)
    valid = torch.ones(2, 9, dtype=torch.bool)
    valid[1, -3:] = False

    x_ref = x.clone().requires_grad_(True)
    out_ref, lb_ref, z_ref, _ = layer(x_ref, valid_mask=valid, backend="reference")
    (out_ref.sum() + lb_ref + z_ref).backward()
    router_grad_ref = layer.router.weight.grad.clone()
    shared_grad_ref = [p.grad.clone() for p in layer.shared_expert.parameters()]
    routed_grad_ref = [
        None if p.grad is None else p.grad.clone() for e in layer.routed_experts for p in e.parameters()
    ]
    layer.zero_grad(set_to_none=True)

    x_opt = x.clone().requires_grad_(True)
    out_opt, lb_opt, z_opt, _ = layer(x_opt, valid_mask=valid, backend="optimized")
    (out_opt.sum() + lb_opt + z_opt).backward()
    router_grad_opt = layer.router.weight.grad.clone()
    shared_grad_opt = [p.grad.clone() for p in layer.shared_expert.parameters()]
    routed_grad_opt = [
        None if p.grad is None else p.grad.clone() for e in layer.routed_experts for p in e.parameters()
    ]

    torch.testing.assert_close(x_ref.grad, x_opt.grad, **_EQUIVALENCE_TOLERANCE)
    torch.testing.assert_close(router_grad_ref, router_grad_opt, **_EQUIVALENCE_TOLERANCE)
    for g_ref, g_opt in zip(shared_grad_ref, shared_grad_opt):
        torch.testing.assert_close(g_ref, g_opt, **_EQUIVALENCE_TOLERANCE)
    for g_ref, g_opt in zip(routed_grad_ref, routed_grad_opt):
        assert (g_ref is None) == (g_opt is None)
        if g_ref is not None:
            torch.testing.assert_close(g_ref, g_opt, **_EQUIVALENCE_TOLERANCE)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA mixed-precision hardware")
def test_reference_and_optimized_agree_under_real_cuda_fp16_autocast():
    torch.manual_seed(13)
    layer, cfg = _layer(n_routed_experts=8, top_k=2, d_model=16, expert_ffn_dim=16)
    layer = layer.to("cuda")
    x = torch.randn(2, 10, cfg.core.d_model, device="cuda")
    valid = torch.ones(2, 10, dtype=torch.bool, device="cuda")
    valid[1, -2:] = False

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        out_ref, lb_ref, z_ref, diag_ref = layer(x, valid_mask=valid, backend="reference", return_diagnostics=True)
        out_opt, lb_opt, z_opt, diag_opt = layer(x, valid_mask=valid, backend="optimized", return_diagnostics=True)

    assert diag_ref.router_logits.dtype == torch.float32
    assert diag_opt.router_logits.dtype == torch.float32
    assert torch.equal(diag_ref.topk_idx, diag_opt.topk_idx)
    torch.testing.assert_close(out_ref.float(), out_opt.float(), atol=2e-2, rtol=2e-2)
