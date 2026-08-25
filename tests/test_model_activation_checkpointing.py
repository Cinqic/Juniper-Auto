"""Optional per-block activation (gradient) checkpointing: off by default,
and when enabled must not change forward or backward numerics."""

from __future__ import annotations

import torch

from juniper_auto.model import build_model
from tests.model_fixtures import make_tiny_sparse_config


def test_gradient_checkpointing_off_by_default():
    cfg = make_tiny_sparse_config()
    model = build_model(cfg, seed=0)
    assert model.gradient_checkpointing is False


def test_gradient_checkpointing_matches_non_checkpointed_forward_and_backward():
    cfg = make_tiny_sparse_config()
    model_a = build_model(cfg, seed=1)
    model_b = build_model(cfg, seed=1)
    model_b.set_gradient_checkpointing(True)
    model_a.train()
    model_b.train()

    torch.manual_seed(0)
    input_ids = torch.randint(0, cfg.embeddings.vocab_size, (2, 6))
    labels = input_ids.clone()

    out_a = model_a(input_ids, labels=labels)
    out_b = model_b(input_ids, labels=labels)
    torch.testing.assert_close(out_a.loss, out_b.loss, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out_a.logits, out_b.logits, atol=1e-5, rtol=1e-5)

    out_a.loss.backward()
    out_b.loss.backward()
    for (name_a, pa), (name_b, pb) in zip(model_a.named_parameters(), model_b.named_parameters()):
        assert name_a == name_b
        assert (pa.grad is None) == (pb.grad is None), name_a
        if pa.grad is not None:
            torch.testing.assert_close(pa.grad, pb.grad, atol=1e-4, rtol=1e-4)


def test_gradient_checkpointing_has_no_effect_in_eval_mode():
    cfg = make_tiny_sparse_config()
    model = build_model(cfg, seed=0)
    model.set_gradient_checkpointing(True)
    model.eval()  # checkpointing only applies during training, per set_gradient_checkpointing's docstring
    input_ids = torch.randint(0, cfg.embeddings.vocab_size, (1, 5))
    with torch.no_grad():
        out = model(input_ids)
    assert torch.isfinite(out.logits).all()


def test_gradient_checkpointing_rejects_diagnostics():
    cfg = make_tiny_sparse_config()
    model = build_model(cfg, seed=0)
    model.set_gradient_checkpointing(True)
    model.train()
    input_ids = torch.randint(0, cfg.embeddings.vocab_size, (1, 4))
    try:
        model(input_ids, return_diagnostics=True)
        assert False, "expected ValueError"
    except ValueError:
        pass
