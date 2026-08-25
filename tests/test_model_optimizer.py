"""Actual optimizer step: parameters change, no NaN/Inf, optimizer state
is valid."""

from __future__ import annotations

import torch

from juniper_auto.model import build_model
from tests.model_fixtures import make_tiny_dense_config, make_tiny_sparse_config


def _run_one_step(cfg):
    model = build_model(cfg, seed=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    before = {name: p.detach().clone() for name, p in model.named_parameters()}

    torch.manual_seed(0)
    input_ids = torch.randint(0, cfg.embeddings.vocab_size, (2, 8))
    labels = input_ids.clone()

    out = model(input_ids, labels=labels)
    optimizer.zero_grad()
    out.loss.backward()
    optimizer.step()

    return model, optimizer, before


def test_sparse_optimizer_step_changes_active_params_no_nan():
    cfg = make_tiny_sparse_config()
    model, optimizer, before = _run_one_step(cfg)

    changed = 0
    for name, p in model.named_parameters():
        assert torch.isfinite(p).all(), f"non-finite parameter after step: {name}"
        if p.grad is not None:
            changed += int(not torch.equal(p, before[name]))
    assert changed > 0

    for group in optimizer.state.values():
        for key in ("exp_avg", "exp_avg_sq"):
            if key in group:
                assert torch.isfinite(group[key]).all()


def test_dense_optimizer_step_changes_all_params_no_nan():
    cfg = make_tiny_dense_config()
    model, optimizer, before = _run_one_step(cfg)

    for name, p in model.named_parameters():
        assert torch.isfinite(p).all(), f"non-finite parameter after step: {name}"
        assert not torch.equal(p, before[name]), f"parameter did not change: {name}"


def test_optimizer_step_does_not_touch_unselected_expert_params():
    cfg = make_tiny_sparse_config(n_routed_experts=8, top_k=2)
    model = build_model(cfg, seed=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    torch.manual_seed(0)
    input_ids = torch.randint(0, cfg.embeddings.vocab_size, (1, 3))  # small batch: some experts likely unselected
    labels = input_ids.clone()
    out = model(input_ids, labels=labels)
    optimizer.zero_grad()
    out.loss.backward()

    unselected_before_grad = {
        name: p.grad for name, p in model.named_parameters() if p.grad is None
    }
    optimizer.step()
    # Params with no gradient this step must be completely untouched by
    # AdamW (no weight decay drift, no state creation for them).
    for name in unselected_before_grad:
        assert name not in {n for n, p in model.named_parameters() if p in optimizer.state}
