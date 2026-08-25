"""Tiny-overfit harness smoke tests on tiny (non-frozen) configs: loss
decreases, no non-finite events, and the harness correctly resumes from
its own checkpoint payload."""

from __future__ import annotations

import torch

from juniper_auto.training.tiny_overfit import (
    TinyOverfitConfig,
    TinyOverfitHarness,
    compute_token_accuracy,
    run_tiny_overfit,
)
from tests.model_fixtures import make_tiny_dense_config, make_tiny_sparse_config


def _run_cfg(**overrides):
    defaults = dict(
        seed=0,
        vocab_size=97,
        seq_len=8,
        n_sequences=4,
        batch_size=2,
        lr=5e-3,
        max_steps=40,
        grad_clip_norm=1.0,
        use_amp=False,
        device="cpu",
    )
    defaults.update(overrides)
    return TinyOverfitConfig(**defaults)


def test_compute_token_accuracy_hand_example():
    logits = torch.zeros(1, 3, 4)
    logits[0, 0, 1] = 10.0  # predicts token 1 at position 0 -> compared to label[1]
    logits[0, 1, 2] = 10.0  # predicts token 2 at position 1 -> compared to label[2]
    labels = torch.tensor([[0, 1, 3]])  # label[1]=1 (correct), label[2]=3 (wrong, predicted 2)
    acc = compute_token_accuracy(logits, labels)
    assert acc == 0.5


def test_dense_tiny_overfit_reduces_loss_and_stays_finite():
    cfg = make_tiny_dense_config()
    result = run_tiny_overfit(cfg, _run_cfg())
    assert result.any_nonfinite_event is False
    assert result.best_lm_loss < result.starting_lm_loss
    assert result.steps_run == 40
    assert result.global_valid_token_count == 40 * 2 * 7  # steps * batch * (seq_len - 1)


def test_sparse_tiny_overfit_reduces_loss_and_stays_finite():
    cfg = make_tiny_sparse_config()
    result = run_tiny_overfit(cfg, _run_cfg(max_steps=60))
    assert result.any_nonfinite_event is False
    assert result.best_lm_loss < result.starting_lm_loss


def test_harness_checkpoint_and_resume_continues_from_same_step():
    cfg = make_tiny_sparse_config()
    run_cfg = _run_cfg(max_steps=10)
    harness = TinyOverfitHarness(cfg, run_cfg)
    for _ in range(5):
        harness.train_step()
    assert harness.global_step == 5

    payload = harness.checkpoint_payload(git_commit="deadbeef", dataset_identity="synthetic Phase 1 engineering data")

    # A real resume reconstructs the harness from the *same* recorded
    # training config (including seed) -- the sampler pool itself is
    # regenerated deterministically from that seed, and only its position
    # (cursor/shuffle-order/step) comes from the checkpoint.
    fresh = TinyOverfitHarness(cfg, run_cfg)
    fresh.load_checkpoint_payload(payload)
    assert fresh.global_step == 5
    assert fresh.global_valid_token_count == harness.global_valid_token_count
    for p_a, p_b in zip(harness.model.parameters(), fresh.model.parameters()):
        torch.testing.assert_close(p_a, p_b)
