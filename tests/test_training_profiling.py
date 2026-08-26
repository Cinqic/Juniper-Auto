"""Profiling methodology labels and training-path coverage."""

from __future__ import annotations

import torch

from juniper_auto.model import build_model
from juniper_auto.training.profiling import profile_inference, profile_training_step
from tests.model_fixtures import make_tiny_dense_config


def test_inference_profile_labels_full_sequence_work_as_prefill():
    cfg = make_tiny_dense_config()
    model = build_model(cfg, seed=0)
    result = profile_inference(
        model,
        vocab_size=cfg.embeddings.vocab_size,
        batch_size=1,
        seq_len=8,
        device="cpu",
        precision_label="fp32",
        warmup_iters=0,
        measured_iters=1,
    )
    assert result.prefill_latency_seconds > 0
    assert result.prefill_tokens_per_second > 0
    assert result.host_peak_rss_bytes is not None
    assert not hasattr(result, "tokens_per_second")


def test_training_profile_executes_accumulation_clipping_and_optimizer(monkeypatch):
    cfg = make_tiny_dense_config()
    model = build_model(cfg, seed=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    first_before = next(model.parameters()).detach().clone()
    real_clip = torch.nn.utils.clip_grad_norm_
    calls = []

    def recording_clip(parameters, max_norm, *args, **kwargs):
        calls.append(max_norm)
        return real_clip(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", recording_clip)
    result = profile_training_step(
        model,
        optimizer,
        vocab_size=cfg.embeddings.vocab_size,
        microbatch_size=1,
        seq_len=8,
        grad_accumulation_steps=2,
        device="cpu",
        use_amp=False,
        grad_clip_norm=0.75,
        warmup_iters=0,
        measured_iters=1,
    )
    assert calls == [0.75]
    assert result.grad_accumulation_steps == 2
    assert result.grad_clip_norm == 0.75
    assert result.numerical_finite is True
    assert not torch.equal(first_before, next(model.parameters()).detach())
