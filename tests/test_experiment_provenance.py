"""Canonical experiment artifacts must be attributable to clean code."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import scripts.run_phase1_experiment as experiment_script


def _args(**overrides):
    values = {
        "command": "probe",
        "result_id": "exp-test",
        "allow_dirty": False,
        "overwrite": False,
        "output": "unused.json",
        "func": object(),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_dirty_tree_is_refused_for_canonical_result(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment_script, "_git_status_porcelain", lambda: " M juniper_auto/model/model.py\n")
    output = tmp_path / "result.json"
    with pytest.raises(RuntimeError, match="dirty working tree"):
        experiment_script._write(
            output,
            {"experiment": "probe"},
            args=_args(),
            config_paths=[experiment_script.SPARSE_PATH],
            seed=7,
        )
    assert not output.exists()


def test_dirty_override_is_explicitly_noncanonical(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment_script, "_git_status_porcelain", lambda: "?? diagnostic.patch\n")
    monkeypatch.setattr(experiment_script, "_git_commit", lambda: "abc123")
    output = tmp_path / "diagnostic.json"
    experiment_script._write(
        output,
        {"experiment": "probe"},
        args=_args(allow_dirty=True),
        config_paths=[experiment_script.SPARSE_PATH],
        seed=7,
    )
    result = json.loads(output.read_text())
    assert result["git_commit"] == "abc123"
    assert result["git_worktree_clean"] is False
    assert result["canonical_result"] is False
    assert result["git_status_porcelain"] == ["?? diagnostic.patch"]


def test_clean_result_records_config_command_seed_and_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment_script, "_git_status_porcelain", lambda: "")
    monkeypatch.setattr(experiment_script, "_git_commit", lambda: "def456")
    output = tmp_path / "result.json"
    experiment_script._write(
        output,
        {"experiment": "probe"},
        args=_args(result_id="exp-0099"),
        config_paths=[experiment_script.SPARSE_PATH],
        seed=11,
    )
    result = json.loads(output.read_text())
    assert result["result_identity"] == "exp-0099"
    assert result["git_worktree_clean"] is True
    assert result["canonical_result"] is True
    assert result["seed"] == 11
    assert result["command"]
    assert result["architecture_configs"][0]["architecture_id"] == "ja150m-v0.1"
    assert len(result["architecture_configs"][0]["config_sha256"]) == 64


def test_existing_result_is_not_silently_overwritten(tmp_path, monkeypatch):
    output = tmp_path / "result.json"
    output.write_text("historical evidence\n")
    with pytest.raises(FileExistsError, match="--overwrite"):
        experiment_script._write(
            output,
            {"experiment": "probe"},
            args=_args(),
            config_paths=[experiment_script.SPARSE_PATH],
            seed=0,
        )
    assert output.read_text() == "historical evidence\n"


def test_unresolvable_head_is_refused_for_canonical_result(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment_script, "_git_status_porcelain", lambda: "")
    monkeypatch.setattr(experiment_script, "_git_commit", lambda: "unknown")
    with pytest.raises(RuntimeError, match="resolvable Git HEAD"):
        experiment_script._write(
            tmp_path / "result.json",
            {"experiment": "probe"},
            args=_args(),
            config_paths=[experiment_script.SPARSE_PATH],
            seed=0,
        )
