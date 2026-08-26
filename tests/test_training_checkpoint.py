"""Checkpoint build/save/load/validate, including the failure cases the
checkpoint format must reject loudly rather than silently accept."""

from __future__ import annotations

import copy
import os

import pytest
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


def test_restore_returns_sequence_curriculum_state_and_preserves_empty_dict():
    cfg = make_tiny_sparse_config()
    payload, _, _ = _make_payload(cfg)
    payload["sequence_curriculum_state"] = {"current_length": 128, "stage": 2}
    resumed = restore_from_checkpoint(payload, model=build_model(cfg, seed=999))
    assert resumed["sequence_curriculum_state"] == {"current_length": 128, "stage": 2}

    model = build_model(cfg, seed=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    empty_payload = build_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scaler=None,
        scheduler=None,
        rng_state=capture_rng_state(),
        sampler_state={},
        global_step=0,
        global_valid_token_count=0,
        architecture_id=cfg.architecture_id,
        architecture_config_dict=cfg.model_dump(),
        training_config={},
        git_commit="deadbeef",
        dataset_identity="synthetic",
        tokenizer_identity="not-yet-created",
        sequence_curriculum_state={},
    )
    assert empty_payload["sequence_curriculum_state"] == {}


def test_non_null_scheduler_and_grad_scaler_state_round_trip():
    cfg = make_tiny_sparse_config()
    model = build_model(cfg, seed=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.25)
    optimizer.step()
    scheduler.step()
    optimizer.step()
    scheduler.step()
    scaler = torch.amp.GradScaler("cpu", enabled=True)
    scaler.load_state_dict(
        {
            "scale": 128.0,
            "growth_factor": 2.0,
            "backoff_factor": 0.5,
            "growth_interval": 17,
            "_growth_tracker": 9,
        }
    )
    payload = build_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=scheduler,
        rng_state=capture_rng_state(),
        sampler_state={"cursor": 4},
        global_step=2,
        global_valid_token_count=8,
        architecture_id=cfg.architecture_id,
        architecture_config_dict=cfg.model_dump(),
        training_config={"scheduler": "StepLR"},
        git_commit="deadbeef",
        dataset_identity="synthetic",
        tokenizer_identity="not-yet-created",
    )

    restored_model = build_model(cfg, seed=999)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=9e-3)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, step_size=1, gamma=0.9)
    restored_scaler = torch.amp.GradScaler("cpu", enabled=True)
    restore_from_checkpoint(
        payload,
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        scaler=restored_scaler,
    )
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    assert restored_scaler.state_dict() == scaler.state_dict()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA FP16 GradScaler")
def test_real_cuda_grad_scaler_state_restores_after_optimizer_step():
    cfg = make_tiny_sparse_config()
    model = build_model(cfg, seed=0, device="cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    ids = torch.randint(0, cfg.embeddings.vocab_size, (2, 6), device="cuda")
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = model(ids, labels=ids).loss
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

    payload = build_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=None,
        rng_state=capture_rng_state(),
        sampler_state={"cursor": 1},
        global_step=1,
        global_valid_token_count=10,
        architecture_id=cfg.architecture_id,
        architecture_config_dict=cfg.model_dump(),
        training_config={"precision": "fp16"},
        git_commit="deadbeef",
        dataset_identity="synthetic",
        tokenizer_identity="not-yet-created",
    )
    restored_model = build_model(cfg, seed=999, device="cuda")
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=9e-3)
    restored_scaler = torch.amp.GradScaler("cuda", enabled=True)
    restore_from_checkpoint(
        payload,
        model=restored_model,
        optimizer=restored_optimizer,
        scaler=restored_scaler,
    )
    assert restored_scaler.state_dict() == scaler.state_dict()
    for original, restored in zip(model.parameters(), restored_model.parameters()):
        torch.testing.assert_close(original, restored)


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
