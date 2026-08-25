"""Interrupted/resumed vs uninterrupted training equivalence (Phase 1
instructions section 31): on a deterministic CPU configuration, resuming
from a mid-run checkpoint must reproduce the exact same subsequent batches,
losses, and final parameters as never having stopped at all."""

from __future__ import annotations

import torch

from juniper_auto.training.tiny_overfit import TinyOverfitConfig, TinyOverfitHarness
from tests.model_fixtures import make_tiny_sparse_config


def _run_cfg(seed):
    return TinyOverfitConfig(
        seed=seed,
        vocab_size=61,
        seq_len=6,
        n_sequences=5,
        batch_size=2,
        lr=1e-2,
        max_steps=20,
        grad_clip_norm=1.0,
        use_amp=False,
        device="cpu",
    )


def test_uninterrupted_vs_resumed_training_matches_exactly_on_cpu():
    cfg = make_tiny_sparse_config()

    # Run A: uninterrupted, 20 steps straight through.
    harness_a = TinyOverfitHarness(cfg, _run_cfg(seed=42))
    history_a = []
    for _ in range(20):
        record, _ = harness_a.train_step()
        history_a.append(record)

    # Run B: 10 steps, checkpoint, "terminate" (drop the harness entirely,
    # simulating a process restart), reconstruct a genuinely fresh harness
    # from the *same recorded training config* (exactly what a real resume
    # does -- the seed comes from the checkpoint's own training_config, not
    # from process memory), load the checkpoint, then resume for 10 more
    # steps.
    harness_b = TinyOverfitHarness(cfg, _run_cfg(seed=42))
    for _ in range(10):
        harness_b.train_step()
    payload = harness_b.checkpoint_payload(
        git_commit="deadbeef", dataset_identity="synthetic Phase 1 engineering data"
    )
    del harness_b

    # Negative control constructed alongside: a same-seed harness that is
    # trained for 10 steps *without* loading the checkpoint would start
    # from pristine initial weights, not from the actual post-10-step
    # state -- so a match below cannot be explained by "same seed" alone.
    harness_b_fresh_no_resume = TinyOverfitHarness(cfg, _run_cfg(seed=42))
    fresh_history = [harness_b_fresh_no_resume.train_step()[0] for _ in range(10)]

    harness_b_resumed = TinyOverfitHarness(cfg, _run_cfg(seed=42))
    harness_b_resumed.load_checkpoint_payload(payload)
    assert harness_b_resumed.global_step == 10

    history_b_resumed = []
    for _ in range(10):
        record, _ = harness_b_resumed.train_step()
        history_b_resumed.append(record)

    # The no-checkpoint negative control must NOT match run A's steps
    # 11-20 -- proving the match below actually depends on checkpoint
    # restoration, not merely on reusing the same seed.
    assert [r.loss for r in fresh_history] != [r.loss for r in history_a[10:]]

    # Steps 11-20 of run A must exactly match the 10 post-resume steps of run B.
    for step_a, step_b in zip(history_a[10:], history_b_resumed):
        assert step_a.step == step_b.step
        assert step_a.loss == step_b.loss
        assert step_a.lm_loss == step_b.lm_loss
        assert step_a.token_accuracy == step_b.token_accuracy

    assert harness_a.global_step == harness_b_resumed.global_step
    assert harness_a.global_valid_token_count == harness_b_resumed.global_valid_token_count

    for p_a, p_b in zip(harness_a.model.parameters(), harness_b_resumed.model.parameters()):
        torch.testing.assert_close(p_a, p_b, atol=0, rtol=0)

    for (name_a, state_a), (name_b, state_b) in zip(
        harness_a.optimizer.state.items(), harness_b_resumed.optimizer.state.items()
    ):
        for key in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(state_a[key], state_b[key], atol=0, rtol=0)


def test_resume_without_restoring_sampler_state_diverges():
    # Negative control proving the equivalence test above is actually
    # sensitive to correct resume behavior: if the data-stream position is
    # NOT restored (a "missing sampler state" bug), the resumed run draws
    # different batches and its losses diverge from the uninterrupted run.
    cfg = make_tiny_sparse_config()

    harness_a = TinyOverfitHarness(cfg, _run_cfg(seed=7))
    history_a = []
    for _ in range(16):
        record, _ = harness_a.train_step()
        history_a.append(record)

    harness_b = TinyOverfitHarness(cfg, _run_cfg(seed=7))
    for _ in range(8):
        harness_b.train_step()
    payload = harness_b.checkpoint_payload(
        git_commit="deadbeef", dataset_identity="synthetic Phase 1 engineering data"
    )

    from juniper_auto.training.checkpoint import restore_from_checkpoint

    broken = TinyOverfitHarness(cfg, _run_cfg(seed=7))
    resumed = restore_from_checkpoint(payload, model=broken.model, optimizer=broken.optimizer)
    broken.global_step = resumed["global_step"]
    broken.global_valid_token_count = resumed["global_valid_token_count"]
    # Deliberately skip: broken.stream.load_state_dict(resumed["sampler_state"])

    history_b_broken = []
    for _ in range(8):
        record, _ = broken.train_step()
        history_b_broken.append(record)

    losses_a = [r.loss for r in history_a[8:]]
    losses_b_broken = [r.loss for r in history_b_broken]
    assert losses_a != losses_b_broken
