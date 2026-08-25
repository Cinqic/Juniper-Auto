"""Checkpoint build/save/load/validate, including the failure cases the
checkpoint format must reject loudly rather than silently accept."""

from __future__ import annotations

import copy
import os

import torch

from juniper_auto.model import build_model
from juniper_auto.training.checkpoint import (
    CheckpointValidationError,
    build_checkpoint_payload,
    load_checkpoint,
    restore_from_checkpoint,
    save_checkpoint,
    validate_checkpoint_payload,
)
from juniper_auto.training.state import capture_rng_state
from tests.model_fixtures import make_tiny_sparse_config


def _make_payload(cfg, seed=0):
    model = build_model(cfg, seed=seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, cfg.embeddings.vocab_size, (1, 4))
    out = model(x, labels=x)
    out.loss.backward()
    optimizer.step()
    return build_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scaler=None,
        scheduler=None,
        rng_state=capture_rng_state(),
        sampler_state={"cursor": 3, "step": 3},
        global_step=3,
        global_valid_token_count=12,
        architecture_id=cfg.architecture_id,
        architecture_config_dict=cfg.model_dump(),
        training_config={"lr": 1e-3},
        git_commit="deadbeef",
        dataset_identity="synthetic Phase 1 engineering data",
        tokenizer_identity="not-yet-created",
    ), model, optimizer


def test_round_trip_save_and_load(tmp_path):
    cfg = make_tiny_sparse_config()
    payload, model, optimizer = _make_payload(cfg)
    path = tmp_path / "checkpoint.pt"
    checksum = save_checkpoint(path, payload)
    assert len(checksum) == 64
    assert path.is_file()
    assert not path.with_name(path.name + ".tmp").exists()  # no leftover temp file

    loaded = load_checkpoint(path, expected_architecture_id=cfg.architecture_id)
    assert loaded["global_step"] == 3
    assert loaded["global_valid_token_count"] == 12
    assert loaded["architecture_id"] == cfg.architecture_id


def test_restore_from_checkpoint_restores_model_and_optimizer(tmp_path):
    cfg = make_tiny_sparse_config()
    payload, model, optimizer = _make_payload(cfg)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, payload)
    loaded = load_checkpoint(path)

    fresh_model = build_model(cfg, seed=999)
    fresh_optimizer = torch.optim.AdamW(fresh_model.parameters(), lr=1e-3)
    resumed = restore_from_checkpoint(loaded, model=fresh_model, optimizer=fresh_optimizer)

    assert resumed["global_step"] == 3
    for p_a, p_b in zip(model.parameters(), fresh_model.parameters()):
        torch.testing.assert_close(p_a, p_b)


def test_load_checkpoint_missing_file_raises_validation_error(tmp_path):
    try:
        load_checkpoint(tmp_path / "does-not-exist.pt")
        assert False, "expected CheckpointValidationError"
    except CheckpointValidationError:
        pass


def test_load_checkpoint_corrupted_file_raises_validation_error(tmp_path):
    path = tmp_path / "corrupt.pt"
    path.write_bytes(b"not a real torch checkpoint file, just garbage bytes")
    try:
        load_checkpoint(path)
        assert False, "expected CheckpointValidationError"
    except CheckpointValidationError:
        pass


def test_validate_rejects_missing_required_field():
    cfg = make_tiny_sparse_config()
    payload, _, _ = _make_payload(cfg)
    broken = dict(payload)
    del broken["rng_state"]
    try:
        validate_checkpoint_payload(broken)
        assert False, "expected CheckpointValidationError"
    except CheckpointValidationError as e:
        assert "rng_state" in str(e)


def test_validate_rejects_missing_sampler_state():
    cfg = make_tiny_sparse_config()
    payload, _, _ = _make_payload(cfg)
    broken = dict(payload)
    del broken["sampler_state"]
    try:
        validate_checkpoint_payload(broken)
        assert False, "expected CheckpointValidationError"
    except CheckpointValidationError as e:
        assert "sampler_state" in str(e)


def test_validate_rejects_wrong_architecture_id():
    cfg = make_tiny_sparse_config()
    payload, _, _ = _make_payload(cfg)
    try:
        validate_checkpoint_payload(payload, expected_architecture_id="some-other-architecture")
        assert False, "expected CheckpointValidationError"
    except CheckpointValidationError as e:
        assert "architecture_id" in str(e)


def test_validate_rejects_unsupported_format_version():
    cfg = make_tiny_sparse_config()
    payload, _, _ = _make_payload(cfg)
    broken = dict(payload)
    broken["checkpoint_format_version"] = 999
    try:
        validate_checkpoint_payload(broken)
        assert False, "expected CheckpointValidationError"
    except CheckpointValidationError as e:
        assert "format version" in str(e)


def test_load_checkpoint_wrong_architecture_is_rejected(tmp_path):
    cfg = make_tiny_sparse_config()
    payload, _, _ = _make_payload(cfg)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, payload)
    try:
        load_checkpoint(path, expected_architecture_id="not-the-real-architecture")
        assert False, "expected CheckpointValidationError"
    except CheckpointValidationError:
        pass


def test_atomic_write_leaves_no_partial_file_under_final_name_on_crash(tmp_path, monkeypatch):
    cfg = make_tiny_sparse_config()
    payload, _, _ = _make_payload(cfg)
    path = tmp_path / "checkpoint.pt"

    original_replace = os.replace

    def failing_replace(src, dst):
        raise RuntimeError("simulated crash between write and rename")

    monkeypatch.setattr(os, "replace", failing_replace)
    try:
        save_checkpoint(path, payload)
        assert False, "expected the simulated crash to propagate"
    except RuntimeError:
        pass
    monkeypatch.setattr(os, "replace", original_replace)

    assert not path.exists()  # the canonical filename was never created
