#!/usr/bin/env python3
"""Single canonical Phase 2 validation entrypoint.

Usage:
    python scripts/validate_phase2.py --all

Runs every Phase 2 local validation gate in order, stopping at the first
failure, after first requiring the Phase 0 and Phase 1 baselines
(scripts/validate_repo.py --all, scripts/validate_phase1.py --all) to pass.
Safe to run from a fresh clone with a populated venv and
`pip install -e . --no-deps`; no GPU required (FLOWBOX-specific CUDA
experiments -- exp-0015's real-CUDA case and exp-0021 -- are recorded as
Phase 2 artifacts under docs/experiments/results/, not re-run here; the
CUDA-gated tests in the pytest suite below run for real whenever CUDA is
actually available and skip cleanly otherwise).
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


def gate_phase1_baseline() -> None:
    _header("phase 1 baseline (includes phase 0)")
    result = subprocess.run([sys.executable, "scripts/validate_phase1.py", "--all"], cwd=REPO_ROOT)
    if result.returncode != 0:
        raise GateFailure(
            "scripts/validate_phase1.py --all did not pass -- Phase 2 cannot build on a failing Phase 1 baseline"
        )


def gate_phase2_imports() -> None:
    _header("phase 2 imports")
    import importlib

    modules = [
        "juniper_auto.model.routing",
        "juniper_auto.model.moe_dispatch",
        "juniper_auto.model.moe_ablations",
        "juniper_auto.model.moe_diagnostics",
        "juniper_auto.model.moe",
        "juniper_auto.model.block",
        "juniper_auto.model.model",
        "juniper_auto.analysis",
        "juniper_auto.analysis.context_sensitivity",
    ]
    for m in modules:
        importlib.import_module(m)
        print(f"import {m}: OK")


def gate_frozen_architecture_unchanged() -> None:
    _header("frozen architecture and parameter counts unchanged")
    from juniper_auto.config import load_architecture_config
    from juniper_auto.config.frozen import FROZEN_STANDARD_ACTIVE_PARAMETERS, FROZEN_TOTAL_PARAMETERS
    from juniper_auto.model import build_model
    from juniper_auto.model.inspection import total_parameters
    from juniper_auto.accounting import standard_active_parameter_breakdown, total_parameter_breakdown

    sparse_cfg = load_architecture_config(REPO_ROOT / "configs/architecture/ja150m-v0.1.yaml")
    dense_cfg = load_architecture_config(REPO_ROOT / "configs/architecture/ja150m-v0.1-dense.yaml")
    sparse_model = build_model(sparse_cfg, seed=0, device="cpu")
    dense_model = build_model(dense_cfg, seed=0, device="cpu")

    checks = [
        ("sparse total", total_parameters(sparse_model), FROZEN_TOTAL_PARAMETERS["ja150m-v0.1"], 150_031_360),
        (
            "sparse standard active",
            standard_active_parameter_breakdown(sparse_cfg).total,
            FROZEN_STANDARD_ACTIVE_PARAMETERS["ja150m-v0.1"],
            79_252_480,
        ),
        ("dense total", total_parameters(dense_model), FROZEN_TOTAL_PARAMETERS["ja150m-v0.1-dense"], 79_191_040),
    ]
    for label, actual, expected_frozen, expected_literal in checks:
        if actual != expected_frozen or actual != expected_literal:
            raise GateFailure(
                f"{label}: expected {expected_literal} (frozen constant {expected_frozen}), got {actual} -- "
                "Phase 2 must not change frozen architecture parameter counts"
            )
        print(f"{label}: {actual} OK (unchanged from Phase 1)")

    dense_positions = {i + 1 for i, k in enumerate(sparse_model.layer_kinds) if k == "dense"}
    if dense_positions != {1, 5, 10, 15, 20}:
        raise GateFailure(f"sparse dense-anchor layer placement drifted: {sorted(dense_positions)}")
    print("layer placement: OK (unchanged from Phase 1)")


def gate_reference_backend_available() -> None:
    _header("reference dispatch backend available and default")
    from juniper_auto.model.moe import MoELayer
    from tests.model_fixtures import make_tiny_sparse_config

    cfg = make_tiny_sparse_config(n_routed_experts=4, top_k=2, d_model=8, expert_ffn_dim=8, n_query_heads=1, n_kv_heads=1, head_dim=8)
    expected_phase1_commit = "073acf46e04241ed35d00bc4b4c29ac463ee744d"
    resolved = subprocess.run(
        ["git", "rev-list", "-n", "1", "phase-1-architecture"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if resolved.returncode != 0 or resolved.stdout.strip() != expected_phase1_commit:
        raise GateFailure(
            "required phase-1-architecture golden tag is missing or moved: "
            f"expected {expected_phase1_commit}, got {resolved.stdout.strip() or resolved.stderr.strip() or 'unavailable'}"
        )
    layer = MoELayer(cfg)
    if layer.backend != "reference":
        raise GateFailure(f"MoELayer default backend changed from 'reference' to {layer.backend!r} without a superseding ADR")
    print("MoELayer default backend: reference OK")


def gate_reference_optimized_equivalence() -> None:
    _header("reference vs optimized numerical equivalence (official architecture)")
    import torch

    from juniper_auto.config import load_architecture_config
    from juniper_auto.model import build_model

    cfg = load_architecture_config(REPO_ROOT / "configs/architecture/ja150m-v0.1.yaml")
    model = build_model(cfg, seed=0, device="cpu")
    moe_positions = [i for i, kind in enumerate(model.layer_kinds) if kind == "moe"]
    layer = model.layers[moe_positions[0]].moe

    torch.manual_seed(0)
    x = torch.randn(2, 32, cfg.core.d_model)
    valid = torch.ones(2, 32, dtype=torch.bool)
    valid[1, -5:] = False

    out_ref, lb_ref, z_ref, diag_ref = layer(x, valid_mask=valid, backend="reference", return_diagnostics=True)
    out_opt, lb_opt, z_opt, diag_opt = layer(x, valid_mask=valid, backend="optimized", return_diagnostics=True)

    if not torch.equal(diag_ref.topk_idx, diag_opt.topk_idx):
        raise GateFailure("reference/optimized routing (topk_idx) diverged on the official architecture")
    max_diff = (out_ref - out_opt).abs().max().item()
    if max_diff >= 1e-4:
        raise GateFailure(f"reference/optimized output diverged beyond tolerance: max_abs_diff={max_diff}")
    print(f"reference vs optimized: routing identical, max_abs_output_diff={max_diff:.2e} OK")


def gate_dropless_invariants() -> None:
    _header("dropless invariants (official architecture, one MoE layer)")
    import torch

    from juniper_auto.config import load_architecture_config
    from juniper_auto.model import build_model

    cfg = load_architecture_config(REPO_ROOT / "configs/architecture/ja150m-v0.1.yaml")
    model = build_model(cfg, seed=0, device="cpu")
    moe_positions = [i for i, kind in enumerate(model.layer_kinds) if kind == "moe"]
    layer = model.layers[moe_positions[0]].moe

    torch.manual_seed(0)
    batch, seq_len = 2, 24
    x = torch.randn(batch, seq_len, cfg.core.d_model)
    valid = torch.ones(batch, seq_len, dtype=torch.bool)
    valid[0, -7:] = False

    _, _, _, diag = layer(x, valid_mask=valid, return_diagnostics=True)
    n_valid = int(valid.sum().item())
    if diag.assignment_counts_per_expert.sum().item() != n_valid * cfg.moe.top_k:
        raise GateFailure("dropless assignment-count invariant violated on the official architecture")
    if not torch.allclose(diag.topk_weights.sum(dim=-1), torch.ones(batch * seq_len), atol=1e-4):
        raise GateFailure("renormalized top-k weights do not sum to 1 on the official architecture")
    print(f"dropless invariants OK: {n_valid} valid tokens x top_k={cfg.moe.top_k} assignments")


def gate_diagnostics_and_ablations_smoke() -> None:
    _header("diagnostics and ablation smoke (official architecture)")
    import torch

    from juniper_auto.config import load_architecture_config
    from juniper_auto.model import build_model
    from juniper_auto.model.moe_ablations import MoEAblationConfig
    from juniper_auto.model.moe_diagnostics import assemble_full_trace, collect_model_expert_gradient_norms

    cfg = load_architecture_config(REPO_ROOT / "configs/architecture/ja150m-v0.1.yaml")
    model = build_model(cfg, seed=0, device="cpu")
    model.eval()
    input_ids = torch.randint(0, cfg.embeddings.vocab_size, (1, 8))

    out = model(input_ids, return_diagnostics=True, return_trace=True)
    trace = assemble_full_trace(model.layer_kinds, out.diagnostics, token_ids=input_ids)
    n_moe_layers = sum(1 for k in model.layer_kinds if k == "moe")
    if len(trace) != n_moe_layers * 8:
        raise GateFailure(f"full-model trace record count mismatch: expected {n_moe_layers * 8}, got {len(trace)}")
    if not all(
        record.weights_normalized
        and record.routed_assignment_count == 2
        and record.shared_expert_activated
        and record.reconstruction_position == record.flat_token_index
        and record.token_id is not None
        for record in trace
    ):
        raise GateFailure("full-model trace is missing a required routing/normalization/reconstruction field")

    out_normal = model(input_ids)
    out_ablated = model(input_ids, ablation=MoEAblationConfig(mode="disable_shared_expert"))
    if torch.allclose(out_normal.logits, out_ablated.logits):
        raise GateFailure("disable_shared_expert ablation had no effect at the model level")
    out_normal_again = model(input_ids)
    if not torch.equal(out_normal.logits, out_normal_again.logits):
        raise GateFailure("ablation state leaked into a subsequent ablation=None model-level call")

    model.train()
    model.zero_grad(set_to_none=True)
    train_out = model(input_ids, labels=input_ids)
    train_out.loss.backward()
    gradient_records = collect_model_expert_gradient_norms(model)
    if len(gradient_records) != n_moe_layers:
        raise GateFailure("expert-gradient telemetry did not return one record per MoE layer")
    if any(record.router is None or record.shared_expert is None for record in gradient_records):
        raise GateFailure("router/shared expert gradient telemetry is missing after backward")
    print(
        f"diagnostics/trace/ablation/gradient smoke OK: {len(trace)} trace records, "
        f"{len(gradient_records)} gradient records, ablation isolated to eval"
    )


def gate_context_sensitivity_infrastructure() -> None:
    _header("context-sensitivity probe infrastructure")
    import torch

    from juniper_auto.analysis.context_sensitivity import run_untrained_official_model_probe
    from juniper_auto.analysis.context_sensitivity import validate_context_probe_templates
    from juniper_auto.config import load_architecture_config
    from juniper_auto.model import build_model

    cfg = load_architecture_config(REPO_ROOT / "configs/architecture/ja150m-v0.1.yaml")
    validate_context_probe_templates()
    model = build_model(cfg, seed=0, device="cpu")
    torch.manual_seed(0)
    probe_token = 100
    contexts = []
    for _ in range(4):
        seq = torch.randint(0, cfg.embeddings.vocab_size, (16,)).tolist()
        seq[4] = probe_token
        contexts.append(seq)
    metrics_by_layer = run_untrained_official_model_probe(model, probe_token, contexts, seed=0)
    if not metrics_by_layer:
        raise GateFailure("context-sensitivity probe returned no per-layer metrics")
    if "ENGINEERING/PROXY TEST" not in run_untrained_official_model_probe.__doc__:
        raise GateFailure("context-sensitivity untrained-model probe lost its required proxy-test labeling")
    print(f"context-sensitivity harness OK: {len(metrics_by_layer)} MoE layers measured")


def gate_reproducibility() -> None:
    _header("reproducibility (fixed seed, fixed config, official architecture)")
    import torch

    from juniper_auto.config import load_architecture_config
    from juniper_auto.model import build_model

    cfg = load_architecture_config(REPO_ROOT / "configs/architecture/ja150m-v0.1.yaml")

    def run_once():
        model = build_model(cfg, seed=0, device="cpu")
        moe_positions = [i for i, k in enumerate(model.layer_kinds) if k == "moe"]
        layer = model.layers[moe_positions[0]].moe
        torch.manual_seed(0)
        x = torch.randn(1, 16, cfg.core.d_model)
        _, _, _, diag = layer(x, return_diagnostics=True)
        return diag

    diag_a = run_once()
    diag_b = run_once()
    if not torch.equal(diag_a.topk_idx, diag_b.topk_idx) or not torch.equal(diag_a.topk_weights, diag_b.topk_weights):
        raise GateFailure("same seed/config/backend did not reproduce identical routing")
    print("reproducibility OK: identical routing across two independent runs")


def gate_pytest() -> None:
    _header("pytest suite (full)")
    result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"], cwd=REPO_ROOT)
    if result.returncode != 0:
        raise GateFailure(f"pytest exited with code {result.returncode}")


def gate_phase2_documentation() -> None:
    _header("phase 2 documentation")
    required = [
        "docs/phases/phase-2-moe.md",
        "docs/phases/phase-2-requirements-traceability.md",
        "docs/phases/phase-2-sonnet-self-review.md",
        "docs/recovery/phase-2.md",
        "docs/architecture/moe-routing-diagnostics.md",
        "docs/adr/0009-moe-dispatch-backend-selection.md",
    ]
    missing = [rel for rel in required if not (REPO_ROOT / rel).is_file()]
    if missing:
        raise GateFailure(f"missing required Phase 2 documentation: {missing}")
    print(f"{len(required)} required Phase 2 documents present")


def gate_phase2_artifact_hashes() -> None:
    _header("phase 2 artifact hashes")
    import yaml

    from juniper_auto.util.hashing import PHASE_2_HASHED_ARTIFACTS, PHASE_2_TEST_FILES, compute_hashes

    test_manifest_path = REPO_ROOT / "manifests" / "phase-2-test-manifest.yaml"
    if not test_manifest_path.is_file():
        raise GateFailure(f"missing Phase 2 test manifest: {test_manifest_path}")
    with test_manifest_path.open() as f:
        test_manifest = yaml.safe_load(f)
    actual_test_hashes = compute_hashes(REPO_ROOT, PHASE_2_TEST_FILES)
    if test_manifest["sha256"] != actual_test_hashes:
        raise GateFailure("manifests/phase-2-test-manifest.yaml is stale relative to the actual test files")
    print(f"phase-2-test-manifest.yaml: {len(actual_test_hashes)} test file hashes verified")

    manifest_path = REPO_ROOT / "manifests" / "phase-2-artifact-hashes.yaml"
    if not manifest_path.is_file():
        raise GateFailure(f"missing hash manifest: {manifest_path}")
    with manifest_path.open() as f:
        manifest = yaml.safe_load(f)
    recorded = manifest["sha256"]
    actual = compute_hashes(REPO_ROOT, PHASE_2_HASHED_ARTIFACTS)
    if set(recorded.keys()) != set(actual.keys()):
        raise GateFailure(
            "hash manifest artifact list does not match code's hashed-artifact list:\n"
            f"  manifest only: {sorted(set(recorded) - set(actual))}\n"
            f"  code only: {sorted(set(actual) - set(recorded))}"
        )
    stale = [rel for rel in recorded if recorded[rel] != actual[rel]]
    if stale:
        raise GateFailure(f"stale Phase 2 artifact hashes (file changed since manifest was generated): {stale}")
    print(f"{len(actual)} Phase 2 artifact hashes verified against manifests/phase-2-artifact-hashes.yaml")


def gate_phase2_experiment_and_time_records() -> None:
    _header("phase 2 experiment registry and time accounting")
    import yaml

    with (REPO_ROOT / "experiments" / "registry.yaml").open() as f:
        entries = yaml.safe_load(f)
    phase2_entries = [e for e in entries if e.get("phase") == "phase-2"]
    if len(phase2_entries) < 7:
        raise GateFailure(f"expected at least 7 phase-2 experiment entries, found {len(phase2_entries)}")
    print(f"{len(phase2_entries)} phase-2 experiment registry entries found")

    import csv

    with (REPO_ROOT / "docs" / "time" / "phase-hours.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    phase2_rows = [r for r in rows if r["phase"] == "phase-2"]
    if not phase2_rows:
        raise GateFailure("no phase-2 rows found in docs/time/phase-hours.csv")
    print(f"{len(phase2_rows)} phase-2 time-accounting rows found")


GATES = [
    ("phase 1 baseline", gate_phase1_baseline),
    ("phase 2 imports", gate_phase2_imports),
    ("frozen architecture unchanged", gate_frozen_architecture_unchanged),
    ("reference backend available and default", gate_reference_backend_available),
    ("reference/optimized equivalence", gate_reference_optimized_equivalence),
    ("dropless invariants", gate_dropless_invariants),
    ("diagnostics/trace/ablation smoke", gate_diagnostics_and_ablations_smoke),
    ("context-sensitivity infrastructure", gate_context_sensitivity_infrastructure),
    ("reproducibility", gate_reproducibility),
    ("pytest suite", gate_pytest),
    ("phase 2 documentation", gate_phase2_documentation),
    ("phase 2 artifact hashes", gate_phase2_artifact_hashes),
    ("phase 2 experiment/time records", gate_phase2_experiment_and_time_records),
]


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="run all Phase 2 validation gates (currently the only mode)")
    args = parser.parse_args()

    if not args.all:
        parser.print_help()
        print("\nNo gates selected. Pass --all to run the full Phase 2 validation suite.")
        return 2

    print(textwrap.dedent(f"""\
        Juniper Auto Phase 2 repository validation
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

    print("\nAll Phase 2 validation gates passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
