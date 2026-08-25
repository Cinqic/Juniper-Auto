#!/usr/bin/env python3
"""Single canonical Phase 0 validation entrypoint.

Usage:
    python scripts/validate_repo.py --all

Runs every Phase 0 local validation gate in order, stopping at the first
failure and printing which gate failed and why. Returns 0 if every gate
passes, non-zero otherwise. Safe to run from a fresh clone: it does not
assume anything beyond a populated venv with requirements-lock.txt
installed and the package installed (`pip install -e . --no-deps`).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

PROHIBITED_TRACKED_PATTERNS = [
    ".env",
    ".venv/",
    "__pycache__/",
    ".pytest_cache/",
    "*.pyc",
    "id_rsa",
    "id_ed25519",
    "*.pem",
    "*.key",
]


class GateFailure(Exception):
    pass


def _header(name: str) -> None:
    print(f"\n=== [{name}] ===")


def gate_environment_sanity() -> None:
    _header("environment sanity")
    import platform

    py = platform.python_version()
    print(f"python: {py}")
    if not py.startswith("3.12"):
        raise GateFailure(f"expected Python 3.12.x, found {py}")

    try:
        import torch
    except ImportError as e:
        raise GateFailure(f"torch is not importable: {e}") from e
    print(f"torch: {torch.__version__} (cuda available: {torch.cuda.is_available()})")

    # Explicitly prove CPU works regardless of CUDA availability -- CI has no GPU.
    x = torch.randn(2, 2, device="cpu")
    _ = (x @ x).sum()
    print("torch CPU tensor op: OK")


def gate_imports() -> None:
    _header("imports")
    import importlib

    modules = [
        "juniper_auto",
        "juniper_auto.config",
        "juniper_auto.accounting",
        "juniper_auto.foundation",
        "juniper_auto.util",
    ]
    for m in modules:
        importlib.import_module(m)
        print(f"import {m}: OK")


def gate_config_validation() -> None:
    _header("config validation")
    from juniper_auto.config import load_architecture_config

    for rel in [
        "configs/architecture/ja150m-v0.1.yaml",
        "configs/architecture/ja150m-v0.1-dense.yaml",
    ]:
        path = REPO_ROOT / rel
        cfg = load_architecture_config(path)
        print(f"{rel}: valid ({cfg.architecture_id}, kind={cfg.kind})")


def gate_parameter_accounting() -> None:
    _header("parameter accounting")
    from juniper_auto.config import assert_frozen_v01, load_architecture_config
    from juniper_auto.config.frozen import FrozenValueMismatch

    sparse = load_architecture_config(REPO_ROOT / "configs/architecture/ja150m-v0.1.yaml")
    dense = load_architecture_config(REPO_ROOT / "configs/architecture/ja150m-v0.1-dense.yaml")

    try:
        assert_frozen_v01(sparse)
        assert_frozen_v01(dense)
    except FrozenValueMismatch as e:
        raise GateFailure(str(e)) from e

    from juniper_auto.accounting import standard_active_parameter_breakdown, total_parameter_breakdown

    print(f"sparse total: {total_parameter_breakdown(sparse).total} (expected 150031360)")
    print(f"sparse active: {standard_active_parameter_breakdown(sparse).total} (expected 79252480)")
    print(f"dense total: {total_parameter_breakdown(dense).total} (expected 79191040)")


def gate_artifact_hashes() -> None:
    _header("artifact hashes")
    import yaml

    from juniper_auto.util.hashing import compute_hashes

    manifest_path = REPO_ROOT / "manifests" / "phase-0-artifact-hashes.yaml"
    if not manifest_path.is_file():
        raise GateFailure(f"missing hash manifest: {manifest_path}")

    with manifest_path.open() as f:
        manifest = yaml.safe_load(f)

    recorded = manifest["sha256"]
    actual = compute_hashes(REPO_ROOT)

    if set(recorded.keys()) != set(actual.keys()):
        raise GateFailure(
            f"hash manifest artifact list does not match code's hashed-artifact list:\n"
            f"  manifest only: {sorted(set(recorded) - set(actual))}\n"
            f"  code only: {sorted(set(actual) - set(recorded))}"
        )

    stale = [rel for rel in recorded if recorded[rel] != actual[rel]]
    if stale:
        raise GateFailure(f"stale artifact hashes (file changed since manifest was generated): {stale}")

    print(f"{len(actual)} artifact hashes verified against manifests/phase-0-artifact-hashes.yaml")


def gate_frozen_artifact_manifest() -> None:
    _header("frozen artifact manifest")
    import yaml

    manifest_path = REPO_ROOT / "manifests" / "frozen-artifacts.yaml"
    if not manifest_path.is_file():
        raise GateFailure(f"missing manifest: {manifest_path}")
    with manifest_path.open() as f:
        manifest = yaml.safe_load(f)
    if manifest["architecture"]["sparse"]["status"] != "frozen":
        raise GateFailure("architecture.sparse status must be 'frozen'")
    if manifest["architecture"]["sparse"]["id"] != "ja150m-v0.1":
        raise GateFailure("architecture.sparse id must be 'ja150m-v0.1'")
    print("manifests/frozen-artifacts.yaml: parses and architecture entries present")


def gate_time_schema() -> None:
    _header("time-accounting schema")
    import csv

    path = REPO_ROOT / "docs" / "time" / "phase-hours.csv"
    if not path.is_file():
        raise GateFailure(f"missing time-accounting file: {path}")
    required_columns = {
        "phase",
        "date",
        "start_time",
        "end_time",
        "engineering_hours",
        "self_review_hours",
        "independent_review_hours",
        "active_human_hours",
        "ai_assisted_engineering_hours",
        "gpu_hours",
        "cpu_data_processing_hours",
        "major_task",
        "commit_or_experiment_id",
        "outcome",
        "blocker_or_failure",
        "approximation_flag",
        "notes",
    }
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        missing = required_columns - fieldnames
        if missing:
            raise GateFailure(f"phase-hours.csv missing required columns: {sorted(missing)}")
        rows = list(reader)
    print(f"phase-hours.csv: schema OK, {len(rows)} rows")


def gate_repository_integrity() -> None:
    _header("repository integrity")
    result = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    tracked = result.stdout.splitlines()

    violations = []
    for f in tracked:
        lower = f.lower()
        if (
            lower.endswith(".env")
            or "/.venv/" in f
            or lower.startswith(".venv/")
            or "__pycache__" in f
            or lower.endswith(".pyc")
            or lower.endswith(".pem")
            or lower.endswith(".key")
            or f.endswith("id_rsa")
            or f.endswith("id_ed25519")
            or lower.endswith(".pt")
            or lower.endswith(".safetensors")
        ):
            violations.append(f)

    if violations:
        raise GateFailure(f"prohibited files are tracked in git: {violations}")
    print(f"{len(tracked)} tracked files scanned, no prohibited artifacts found")

    # Absolute host-specific path check across versioned config/scripts/docs.
    suspicious = []
    home = str(Path.home())
    for f in tracked:
        if f.startswith(".git/"):
            continue
        full = REPO_ROOT / f
        if not full.is_file():
            continue
        try:
            text = full.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if home in text and "environment-specification.md" not in f and "phase-0-sonnet-self-review.md" not in f:
            suspicious.append(f)
    if suspicious:
        raise GateFailure(
            f"files reference the current user's absolute home path outside "
            f"documented illustrative contexts: {suspicious}"
        )
    print("no unjustified absolute host-specific paths found")


def gate_deterministic_probe() -> None:
    _header("deterministic foundation probe")
    from juniper_auto.config import load_architecture_config
    from juniper_auto.foundation import run_foundation_probe

    cfg = load_architecture_config(REPO_ROOT / "configs/architecture/ja150m-v0.1.yaml")
    r1 = run_foundation_probe(cfg, seed=1234, device="cpu")
    r2 = run_foundation_probe(cfg, seed=1234, device="cpu")
    if r1.output_checksum != r2.output_checksum:
        raise GateFailure(
            f"FoundationProbe not deterministic under fixed seed: {r1.output_checksum} != {r2.output_checksum}"
        )
    print(f"FoundationProbe deterministic under seed=1234 on cpu: checksum={r1.output_checksum}")


def gate_pytest() -> None:
    _header("pytest suite")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q"],
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        raise GateFailure(f"pytest exited with code {result.returncode}")


GATES = [
    ("environment sanity", gate_environment_sanity),
    ("imports", gate_imports),
    ("config validation", gate_config_validation),
    ("parameter accounting", gate_parameter_accounting),
    ("frozen artifact manifest", gate_frozen_artifact_manifest),
    ("time-accounting schema", gate_time_schema),
    ("repository integrity", gate_repository_integrity),
    ("artifact hashes", gate_artifact_hashes),
    ("deterministic foundation probe", gate_deterministic_probe),
    ("pytest suite", gate_pytest),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="run all Phase 0 validation gates (currently the only mode)")
    args = parser.parse_args()

    if not args.all:
        parser.print_help()
        print("\nNo gates selected. Pass --all to run the full Phase 0 validation suite.")
        return 2

    print(textwrap.dedent(f"""\
        Juniper Auto Phase 0 repository validation
        Repo root: {REPO_ROOT}
        Gates: {len(GATES)}
    """))

    for name, gate_fn in GATES:
        try:
            gate_fn()
        except GateFailure as e:
            print(f"\nFAILED at gate '{name}': {e}", file=sys.stderr)
            return 1
        except Exception as e:  # unexpected error inside a gate
            print(f"\nFAILED at gate '{name}' with unexpected error: {e!r}", file=sys.stderr)
            return 1

    print("\nAll Phase 0 validation gates passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
