import importlib.util
from pathlib import Path

import pytest


def _validator_module(repo_root):
    path = repo_root / "scripts/validate_repo.py"
    spec = importlib.util.spec_from_file_location("phase0_validator_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_dependency_gate_rejects_missing_declared_dependency(repo_root, tmp_path, monkeypatch):
    validator = _validator_module(repo_root)
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="x"\nversion="0"\ndependencies=["missing-package>=1"]\n'
    )
    (tmp_path / "requirements-lock.txt").write_text(
        "# pip-compile --generate-hashes\npytest==8.4.2 \\\n+    --hash=sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
    )
    monkeypatch.setattr(validator, "REPO_ROOT", Path(tmp_path))
    with pytest.raises(validator.GateFailure, match="missing from lock"):
        validator.gate_dependency_consistency()


def test_manifest_gate_rejects_missing_categories(repo_root, tmp_path, monkeypatch):
    validator = _validator_module(repo_root)
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "frozen-artifacts.yaml").write_text(
        "architecture:\n  sparse:\n    id: ja150m-v0.1\n    status: frozen\n"
    )
    monkeypatch.setattr(validator, "REPO_ROOT", Path(tmp_path))
    with pytest.raises(validator.GateFailure, match="categories missing"):
        validator.gate_frozen_artifact_manifest()


def test_hash_gate_rejects_stale_protected_artifact(repo_root, tmp_path, monkeypatch):
    validator = _validator_module(repo_root)
    import yaml
    from juniper_auto.util.hashing import PHASE_0_HASHED_ARTIFACTS

    for relative in PHASE_0_HASHED_ARTIFACTS:
        protected = tmp_path / relative
        protected.parent.mkdir(parents=True, exist_ok=True)
        protected.write_text("current")
    manifests = tmp_path / "manifests"
    manifests.mkdir(exist_ok=True)
    (manifests / "phase-0-artifact-hashes.yaml").write_text(
        yaml.safe_dump({"algorithm": "sha256", "sha256": {path: "0" * 64 for path in PHASE_0_HASHED_ARTIFACTS}})
    )
    monkeypatch.setattr(validator, "REPO_ROOT", Path(tmp_path))
    with pytest.raises(validator.GateFailure, match="stale artifact hashes"):
        validator.gate_artifact_hashes()
