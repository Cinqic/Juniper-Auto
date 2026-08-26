"""Versioned Phase 1 training checkpoint format: build, save (atomically),
load, and validate.

A checkpoint bundles everything required to resume training bit-for-bit
reproducibly on the same hardware/software stack (see
docs/phases/phase-1-architecture.md): model weights, optimizer/scheduler/
GradScaler state, every RNG stream, the sampler/data-stream state, step
counters, and enough identity metadata (architecture id + full config,
git commit, dataset/tokenizer identity, checkpoint format version) that a
checkpoint from the wrong architecture or an incompatible format is
rejected loudly rather than silently loaded.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from juniper_auto.training.state import RNGState, rng_state_from_dict, rng_state_to_dict
from juniper_auto.util.hashing import sha256_file

CHECKPOINT_FORMAT_VERSION = 1

REQUIRED_CHECKPOINT_FIELDS = {
    "checkpoint_format_version",
    "architecture_id",
    "architecture_config",
    "training_config",
    "git_commit",
    "dataset_identity",
    "tokenizer_identity",
    "global_step",
    "global_valid_token_count",
    "sequence_curriculum_state",
    "model_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "scaler_state_dict",
    "rng_state",
    "sampler_state",
}


class CheckpointValidationError(Exception):
    pass


def build_checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: "torch.cuda.amp.GradScaler | None",
    scheduler: Any | None,
    rng_state: RNGState,
    sampler_state: dict,
    global_step: int,
    global_valid_token_count: int,
    architecture_id: str,
    architecture_config_dict: dict,
    training_config: dict,
    git_commit: str,
    dataset_identity: str,
    tokenizer_identity: str,
    sequence_curriculum_state: dict | None = None,
) -> dict:
    return {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "architecture_id": architecture_id,
        "architecture_config": architecture_config_dict,
        "training_config": training_config,
        "git_commit": git_commit,
        "dataset_identity": dataset_identity,
        "tokenizer_identity": tokenizer_identity,
        "global_step": global_step,
        "global_valid_token_count": global_valid_token_count,
        # Phase 1 does not implement a sequence-length curriculum; this is
        # recorded honestly rather than omitted, so a future phase that
        # does implement one has a defined field to populate.
        "sequence_curriculum_state": (
            {"status": "not-implemented-phase-1"}
            if sequence_curriculum_state is None
            else sequence_curriculum_state
        ),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "rng_state": rng_state_to_dict(rng_state),
        "sampler_state": sampler_state,
    }


def save_checkpoint(path: str | Path, payload: dict) -> str:
    """Writes to a temporary file in the same directory and atomically
    renames it into place, so a crash mid-write never leaves a partially
    written file under the canonical checkpoint filename. Returns the
    saved file's SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)
    return sha256_file(path)


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device | None = None,
    expected_architecture_id: str | None = None,
) -> dict:
    """Loads and validates a Phase 1 checkpoint. Raises
    CheckpointValidationError (not a raw torch/pickle exception) on a
    missing file, a corrupted/truncated file, a missing required field, an
    unsupported format version, or an architecture-id mismatch.

    `weights_only=False` is required because this payload carries plain
    Python metadata (dicts, RNG state tuples) alongside tensors, not only
    model weights -- so this must only ever be pointed at checkpoints this
    project produced itself, never at an untrusted file.
    """
    path = Path(path)
    if not path.is_file():
        raise CheckpointValidationError(f"checkpoint file does not exist: {path}")
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except Exception as exc:
        raise CheckpointValidationError(f"checkpoint file is corrupted or unreadable: {path}: {exc}") from exc
    validate_checkpoint_payload(payload, expected_architecture_id=expected_architecture_id)
    return payload


def validate_checkpoint_payload(payload: Any, *, expected_architecture_id: str | None = None) -> None:
    if not isinstance(payload, dict):
        raise CheckpointValidationError(f"checkpoint payload must be a dict, got {type(payload)!r}")

    missing = REQUIRED_CHECKPOINT_FIELDS - set(payload.keys())
    if missing:
        raise CheckpointValidationError(f"checkpoint missing required fields: {sorted(missing)}")

    version = payload["checkpoint_format_version"]
    if version != CHECKPOINT_FORMAT_VERSION:
        raise CheckpointValidationError(
            f"unsupported checkpoint format version: {version!r} (expected {CHECKPOINT_FORMAT_VERSION})"
        )

    if expected_architecture_id is not None and payload["architecture_id"] != expected_architecture_id:
        raise CheckpointValidationError(
            f"checkpoint architecture_id {payload['architecture_id']!r} does not match "
            f"expected {expected_architecture_id!r}"
        )


def restore_from_checkpoint(
    payload: dict,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: "torch.cuda.amp.GradScaler | None" = None,
    restore_rng: bool = True,
) -> dict:
    """Applies model/optimizer/scheduler/scaler state (and, by default,
    every RNG stream) in place from an already-validated checkpoint
    payload. Returns the remaining resumable training-loop state
    (global_step, global_valid_token_count, sampler_state, and sequence
    curriculum state) for the caller to restore into its own loop variables
    and data stream.
    """
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if scaler is not None and payload["scaler_state_dict"] is not None:
        scaler.load_state_dict(payload["scaler_state_dict"])

    if restore_rng:
        from juniper_auto.training.state import restore_rng_state

        restore_rng_state(rng_state_from_dict(payload["rng_state"]))

    return {
        "global_step": payload["global_step"],
        "global_valid_token_count": payload["global_valid_token_count"],
        "sampler_state": payload["sampler_state"],
        "sequence_curriculum_state": payload["sequence_curriculum_state"],
    }
