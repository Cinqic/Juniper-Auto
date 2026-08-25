#!/usr/bin/env python3
"""Single canonical Phase 1 validation entrypoint.

Usage:
    python scripts/validate_phase1.py --all

Runs every Phase 1 local validation gate in order, stopping at the first
failure. Assumes the Phase 0 baseline (scripts/validate_repo.py --all)
also passes -- this script re-checks a subset directly (imports, official
parameter counts) but is not a substitute for running Phase 0's validator,
which this script's first gate invokes. Safe to run from a fresh clone
with a populated venv and `pip install -e . --no-deps`; no GPU required
(hardware-specific CUDA experiments are recorded as Phase 1 artifacts
under docs/experiments/results/, not re-run here).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


class GateFailure(Exception):
    pass


def _header(name: str) -> None:
    print(f"\n=== [{name}] ===")


def gate_phase0_baseline() -> None:
    _header("phase 0 baseline")
    result = subprocess.run([sys.executable, "scripts/validate_repo.py", "--all"], cwd=REPO_ROOT)
    if result.returncode != 0:
        raise GateFailure("scripts/validate_repo.py --all did not pass -- Phase 1 cannot build on a failing Phase 0 baseline")


def gate_phase1_imports() -> None:
    _header("phase 1 imports")
    import importlib

    modules = [
        "juniper_auto.model",
        "juniper_auto.model.norm",
        "juniper_auto.model.rope",
        "juniper_auto.model.attention",
        "juniper_auto.model.ffn",
        "juniper_auto.model.moe",
        "juniper_auto.model.block",
        "juniper_auto.model.losses",
        "juniper_auto.model.model",
        "juniper_auto.model.inspection",
        "juniper_auto.training",
        "juniper_auto.training.state",
        "juniper_auto.training.checkpoint",
        "juniper_auto.training.tiny_overfit",
        "juniper_auto.training.profiling",
    ]
    for m in modules:
        importlib.import_module(m)
        print(f"import {m}: OK")


def gate_official_model_construction_and_parameter_counts() -> None:
    _header("official model construction and parameter counts")
    from juniper_auto.config import load_architecture_config
    from juniper_auto.config.frozen import FROZEN_STANDARD_ACTIVE_PARAMETERS, FROZEN_TOTAL_PARAMETERS
    from juniper_auto.model import build_model
    from juniper_auto.model.inspection import (
        bias_audit,
        dropout_audit,
        pytorch_parameter_breakdown,
        qk_norm_parameter_count,
        total_parameters,
        verify_weight_tying,
    )
    from juniper_auto.accounting import standard_active_parameter_breakdown, total_parameter_breakdown

    sparse_cfg = load_architecture_config(REPO_ROOT / "configs/architecture/ja150m-v0.1.yaml")
    dense_cfg = load_architecture_config(REPO_ROOT / "configs/architecture/ja150m-v0.1-dense.yaml")

    sparse_model = build_model(sparse_cfg, seed=0, device="cpu")
    dense_model = build_model(dense_cfg, seed=0, device="cpu")

    checks = [
        ("sparse total (method B)", total_parameters(sparse_model), FROZEN_TOTAL_PARAMETERS["ja150m-v0.1"]),
        (
            "sparse total (method A)",
            total_parameter_breakdown(sparse_cfg).total,
            FROZEN_TOTAL_PARAMETERS["ja150m-v0.1"],
        ),
        (
            "sparse active (method A)",
            standard_active_parameter_breakdown(sparse_cfg).total,
            FROZEN_STANDARD_ACTIVE_PARAMETERS["ja150m-v0.1"],
        ),
        ("dense total (method B)", total_parameters(dense_model), FROZEN_TOTAL_PARAMETERS["ja150m-v0.1-dense"]),
        (
            "dense total (method A)",
            total_parameter_breakdown(dense_cfg).total,
            FROZEN_TOTAL_PARAMETERS["ja150m-v0.1-dense"],
        ),
    ]
    for label, actual, expected in checks:
        if actual != expected:
            raise GateFailure(f"{label}: expected {expected}, got {actual}")
        print(f"{label}: {actual} OK")

    method_b_sparse = pytorch_parameter_breakdown(sparse_model)
    method_a_sparse = total_parameter_breakdown(sparse_cfg).as_dict()
    for key in method_a_sparse:
        if method_a_sparse[key] != method_b_sparse[key]:
            raise GateFailure(f"sparse breakdown mismatch in {key}: method A={method_a_sparse[key]} method B={method_b_sparse[key]}")

    if qk_norm_parameter_count(sparse_model) != 2560:
        raise GateFailure(f"expected exactly 2560 QK-Norm parameters, got {qk_norm_parameter_count(sparse_model)}")
    print("QK-Norm parameter count: 2560 OK")

    if not verify_weight_tying(sparse_model) or not verify_weight_tying(dense_model):
        raise GateFailure("weight tying is not true object-identity sharing")
    print("weight tying: OK")

    bias_offenders = bias_audit(sparse_model) + bias_audit(dense_model)
    if bias_offenders:
        raise GateFailure(f"unintended bias parameters found: {bias_offenders}")
    dropout_offenders = dropout_audit(sparse_model) + dropout_audit(dense_model)
    if dropout_offenders:
        raise GateFailure(f"unintended dropout modules found: {dropout_offenders}")
    print("bias/dropout audit: OK")

    layer_kinds = sparse_model.layer_kinds
    dense_positions = {i + 1 for i, k in enumerate(layer_kinds) if k == "dense"}
    moe_positions = {i + 1 for i, k in enumerate(layer_kinds) if k == "moe"}
    if dense_positions != {1, 5, 10, 15, 20} or moe_positions != set(range(1, 21)) - {1, 5, 10, 15, 20}:
        raise GateFailure(f"sparse layer placement does not match the frozen spec: dense={sorted(dense_positions)}")
    print("layer placement: OK")


def gate_forward_backward_smoke() -> None:
    _header("forward/backward smoke (official models, tiny batch, CPU)")
    import torch

    from juniper_auto.config import load_architecture_config
    from juniper_auto.model import build_model

    for rel in ["configs/architecture/ja150m-v0.1.yaml", "configs/architecture/ja150m-v0.1-dense.yaml"]:
        cfg = load_architecture_config(REPO_ROOT / rel)
        model = build_model(cfg, seed=0, device="cpu")
        input_ids = torch.randint(0, cfg.embeddings.vocab_size, (1, 4))
        labels = input_ids.clone()
        out = model(input_ids, labels=labels)
        if not torch.isfinite(out.loss):
            raise GateFailure(f"{cfg.architecture_id}: non-finite loss on smoke forward pass")
        out.loss.backward()
        if not any(p.grad is not None for p in model.parameters()):
            raise GateFailure(f"{cfg.architecture_id}: no gradients produced on smoke backward pass")
        print(f"{cfg.architecture_id}: forward+backward OK, loss={out.loss.item():.4f}")


def gate_checkpoint_round_trip_smoke() -> None:
    _header("checkpoint round-trip smoke (tiny config, CPU)")
    import tempfile

    from juniper_auto.training.checkpoint import load_checkpoint, save_checkpoint
    from juniper_auto.training.tiny_overfit import TinyOverfitConfig, TinyOverfitHarness

    sys.path.insert(0, str(REPO_ROOT / "tests"))
    from model_fixtures import make_tiny_sparse_config  # noqa: E402

    cfg = make_tiny_sparse_config()
    run_cfg = TinyOverfitConfig(
        seed=0, vocab_size=cfg.embeddings.vocab_size, seq_len=6, n_sequences=4, batch_size=2,
        lr=1e-3, max_steps=3, grad_clip_norm=1.0, use_amp=False, device="cpu",
    )
    harness = TinyOverfitHarness(cfg, run_cfg)
    harness.train_step()
    payload = harness.checkpoint_payload(git_commit="ci-smoke", dataset_identity="synthetic Phase 1 engineering data")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "smoke.pt"
        save_checkpoint(path, payload)
        loaded = load_checkpoint(path, expected_architecture_id=cfg.architecture_id)
        if loaded["global_step"] != 1:
            raise GateFailure("checkpoint round trip did not preserve global_step")
    print("checkpoint save/load round trip: OK")


def gate_pytest() -> None:
    _header("pytest suite (full)")
    result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"], cwd=REPO_ROOT)
    if result.returncode != 0:
        raise GateFailure(f"pytest exited with code {result.returncode}")


def gate_phase1_documentation() -> None:
    _header("phase 1 documentation")
    required = [
        "docs/phases/phase-1-architecture.md",
        "docs/phases/phase-1-requirements-traceability.md",
        "docs/phases/phase-1-sonnet-self-review.md",
        "docs/architecture/reference-model-implementation.md",
        "docs/recovery/phase-1.md",
        "docs/adr/0008-moe-auxiliary-loss-semantics.md",
    ]
    missing = [rel for rel in required if not (REPO_ROOT / rel).is_file()]
    if missing:
        raise GateFailure(f"missing required Phase 1 documentation: {missing}")
    print(f"{len(required)} required Phase 1 documents present")


def gate_phase1_artifact_hashes() -> None:
    _header("phase 1 artifact hashes")
    import yaml

    from juniper_auto.util.hashing import PHASE_1_HASHED_ARTIFACTS, PHASE_1_TEST_FILES, compute_hashes

    test_manifest_path = REPO_ROOT / "manifests" / "phase-1-test-manifest.yaml"
    if not test_manifest_path.is_file():
        raise GateFailure(f"missing Phase 1 test manifest: {test_manifest_path}")
    with test_manifest_path.open() as f:
        test_manifest = yaml.safe_load(f)
    actual_test_hashes = compute_hashes(REPO_ROOT, PHASE_1_TEST_FILES)
    if test_manifest["sha256"] != actual_test_hashes:
        raise GateFailure("manifests/phase-1-test-manifest.yaml is stale relative to the actual test files")
    print(f"phase-1-test-manifest.yaml: {len(actual_test_hashes)} test file hashes verified")

    manifest_path = REPO_ROOT / "manifests" / "phase-1-artifact-hashes.yaml"
    if not manifest_path.is_file():
        raise GateFailure(f"missing hash manifest: {manifest_path}")
    with manifest_path.open() as f:
        manifest = yaml.safe_load(f)
    recorded = manifest["sha256"]
    actual = compute_hashes(REPO_ROOT, PHASE_1_HASHED_ARTIFACTS)
    if set(recorded.keys()) != set(actual.keys()):
        raise GateFailure(
            f"hash manifest artifact list does not match code's hashed-artifact list:\n"
            f"  manifest only: {sorted(set(recorded) - set(actual))}\n"
            f"  code only: {sorted(set(actual) - set(recorded))}"
        )
    stale = [rel for rel in recorded if recorded[rel] != actual[rel]]
    if stale:
        raise GateFailure(f"stale Phase 1 artifact hashes (file changed since manifest was generated): {stale}")
    print(f"{len(actual)} Phase 1 artifact hashes verified against manifests/phase-1-artifact-hashes.yaml")


def gate_phase1_experiment_and_time_records() -> None:
    _header("phase 1 experiment registry and time accounting")
    import yaml

    with (REPO_ROOT / "experiments" / "registry.yaml").open() as f:
        entries = yaml.safe_load(f)
    phase1_entries = [e for e in entries if e.get("phase") == "phase-1"]
    if len(phase1_entries) < 6:
        raise GateFailure(f"expected at least 6 phase-1 experiment entries, found {len(phase1_entries)}")
    print(f"{len(phase1_entries)} phase-1 experiment registry entries found")

    import csv

    with (REPO_ROOT / "docs" / "time" / "phase-hours.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    phase1_rows = [r for r in rows if r["phase"] == "phase-1"]
    if not phase1_rows:
        raise GateFailure("no phase-1 rows found in docs/time/phase-hours.csv")
    print(f"{len(phase1_rows)} phase-1 time-accounting rows found")


GATES = [
    ("phase 0 baseline", gate_phase0_baseline),
    ("phase 1 imports", gate_phase1_imports),
    ("official model construction and parameter counts", gate_official_model_construction_and_parameter_counts),
    ("forward/backward smoke", gate_forward_backward_smoke),
    ("checkpoint round-trip smoke", gate_checkpoint_round_trip_smoke),
    ("pytest suite", gate_pytest),
    ("phase 1 documentation", gate_phase1_documentation),
    ("phase 1 artifact hashes", gate_phase1_artifact_hashes),
    ("phase 1 experiment/time records", gate_phase1_experiment_and_time_records),
]


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="run all Phase 1 validation gates (currently the only mode)")
    args = parser.parse_args()

    if not args.all:
        parser.print_help()
        print("\nNo gates selected. Pass --all to run the full Phase 1 validation suite.")
        return 2

    print(textwrap.dedent(f"""\
        Juniper Auto Phase 1 repository validation
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

    print("\nAll Phase 1 validation gates passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
