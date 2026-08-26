#!/usr/bin/env python3
"""Run a single Phase 2 experiment against the real official architectures
and write a JSON result artifact, mirroring scripts/run_phase1_experiment.py's
provenance conventions.

Usage:
    python scripts/run_phase2_experiment.py equivalence --output <path>
    python scripts/run_phase2_experiment.py routing-trace --output <path>
    python scripts/run_phase2_experiment.py context-sensitivity-baseline --output <path>
    python scripts/run_phase2_experiment.py detector-validation --output <path>
    python scripts/run_phase2_experiment.py ablation-validation --output <path>
    python scripts/run_phase2_experiment.py reproducibility --output <path>
    python scripts/run_phase2_experiment.py flowbox-moe-profile --output <path>

These are real, executed runs against the official ja150m-v0.1 architecture
(or, where noted, tiny synthetic configs deliberately constructed to
exercise a specific code path) -- not estimates.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from juniper_auto.analysis.context_sensitivity import run_untrained_official_model_probe  # noqa: E402
from juniper_auto.config import load_architecture_config  # noqa: E402
from juniper_auto.model import build_model  # noqa: E402
from juniper_auto.model.moe_ablations import MoEAblationConfig  # noqa: E402
from juniper_auto.model.moe_diagnostics import (  # noqa: E402
    RoutingWindowAccumulator,
    assemble_full_trace,
    detect_dead_experts,
    detect_dominant_experts,
    detect_router_saturation,
    detect_routing_collapse,
    detect_starved_experts,
    export_trace_json,
)
from juniper_auto.util.environment import describe_environment  # noqa: E402
from juniper_auto.util.hashing import sha256_file  # noqa: E402

SPARSE_PATH = REPO_ROOT / "configs/architecture/ja150m-v0.1.yaml"


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=5, check=True
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _git_status_porcelain() -> str:
    out = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=10, check=True,
    )
    return out.stdout


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _asdict(obj):
    if dataclasses.is_dataclass(obj):
        return {k: _asdict(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, list):
        return [_asdict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _asdict(v) for k, v in obj.items()}
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    return obj


def _config_identity(path: Path) -> dict:
    cfg = load_architecture_config(path)
    return {
        "architecture_id": cfg.architecture_id,
        "config_path": str(path.relative_to(REPO_ROOT)),
        "config_sha256": sha256_file(path),
    }


def _write(path: Path, payload: dict, *, args, config_paths: list[Path], seed) -> None:
    if path.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing result artifact without --overwrite: {path}")

    status = _git_status_porcelain()
    clean = status == ""
    if not clean and not args.allow_dirty:
        raise RuntimeError(
            "refusing to produce a canonical experiment result from a dirty working tree; "
            "commit/stash the changes or pass --allow-dirty for an explicitly non-canonical diagnostic result"
        )

    commit = _git_commit()
    if commit == "unknown" and not args.allow_dirty:
        raise RuntimeError("refusing to produce a canonical experiment result without a resolvable Git HEAD")

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "result_identity": args.result_id or path.stem,
        "git_commit": commit,
        "git_worktree_clean": clean,
        "canonical_result": clean,
        "git_status_porcelain": status.splitlines(),
        "architecture_configs": [_config_identity(p) for p in config_paths],
        "environment": describe_environment().as_dict(),
        "command": shlex.join(sys.argv),
        "seed": seed,
        **payload,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    print(f"wrote {path}")


def _official_sparse_moe_layer(seed: int, device: str = "cpu"):
    cfg = load_architecture_config(SPARSE_PATH)
    model = build_model(cfg, seed=seed, device=device)
    moe_positions = [i for i, kind in enumerate(model.layer_kinds) if kind == "moe"]
    return model, model.layers[moe_positions[0]].moe, cfg


def cmd_equivalence(args) -> None:
    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    _, layer, cfg = _official_sparse_moe_layer(seed=0, device=device)

    cases = []
    max_abs_diff = 0.0
    all_routing_identical = True
    for seed in range(args.n_cases):
        torch.manual_seed(seed)
        batch, seq_len = 2, 64
        x = torch.randn(batch, seq_len, cfg.core.d_model, device=device)
        valid = torch.ones(batch, seq_len, dtype=torch.bool, device=device)
        valid[:, seed % seq_len :] = False if seed % 3 == 0 else valid[:, seed % seq_len :]

        out_ref, lb_ref, z_ref, diag_ref = layer(x, valid_mask=valid, backend="reference", return_diagnostics=True)
        out_opt, lb_opt, z_opt, diag_opt = layer(x, valid_mask=valid, backend="optimized", return_diagnostics=True)

        routing_identical = bool(torch.equal(diag_ref.topk_idx, diag_opt.topk_idx)) and bool(
            torch.equal(diag_ref.topk_weights, diag_opt.topk_weights)
        )
        all_routing_identical = all_routing_identical and routing_identical
        abs_diff = (out_ref - out_opt).abs().max().item()
        max_abs_diff = max(max_abs_diff, abs_diff)
        cases.append({
            "seed": seed, "batch": batch, "seq_len": seq_len,
            "routing_identical": routing_identical, "max_abs_output_diff": abs_diff,
            "load_balance_loss_identical": bool(torch.equal(lb_ref, lb_opt)),
            "router_z_loss_identical": bool(torch.equal(z_ref, z_opt)),
        })

    tolerance = 1e-4
    gate_passed = bool(all_routing_identical and max_abs_diff < tolerance)
    result = {
        "experiment": "reference-vs-optimized-moe-equivalence",
        "architecture_id": cfg.architecture_id,
        "device": device,
        "n_cases": args.n_cases,
        "cases": cases,
        "max_abs_output_diff_across_all_cases": max_abs_diff,
        "gate": {"routing_must_be_identical": True, "max_abs_output_diff_tolerance": tolerance},
        "gate_passed": gate_passed,
    }
    _write(Path(args.output), result, args=args, config_paths=[SPARSE_PATH], seed=0)


def cmd_routing_trace(args) -> None:
    device = "cpu"
    model, _, cfg = _official_sparse_moe_layer(seed=0, device=device)
    torch.manual_seed(0)
    batch, seq_len = 2, 16
    input_ids = torch.randint(0, cfg.embeddings.vocab_size, (batch, seq_len))
    attention_mask = torch.ones(batch, seq_len, dtype=torch.bool)
    attention_mask[1, -3:] = False

    out = model(input_ids, attention_mask=attention_mask, return_diagnostics=True, return_trace=True)
    full_trace = assemble_full_trace(model.layer_kinds, out.diagnostics)

    trace_path = Path(args.output).with_suffix(".trace.json")
    export_trace_json(full_trace, trace_path)

    n_moe_layers = sum(1 for k in model.layer_kinds if k == "moe")
    result = {
        "experiment": "official-model-routing-trace",
        "architecture_id": cfg.architecture_id,
        "batch": batch,
        "seq_len": seq_len,
        "n_moe_layers": n_moe_layers,
        "n_trace_records": len(full_trace),
        "expected_trace_records": n_moe_layers * batch * seq_len,
        "trace_artifact": str(trace_path.relative_to(REPO_ROOT)) if trace_path.is_relative_to(REPO_ROOT) else str(trace_path),
        "gate": {"n_trace_records_must_equal_expected": True},
        "gate_passed": len(full_trace) == n_moe_layers * batch * seq_len,
    }
    _write(Path(args.output), result, args=args, config_paths=[SPARSE_PATH], seed=0)


def cmd_context_sensitivity_baseline(args) -> None:
    device = "cpu"
    model, _, cfg = _official_sparse_moe_layer(seed=0, device=device)
    torch.manual_seed(args.seed)
    probe_token = 100
    seq_len = 32
    contexts = []
    for _ in range(args.n_contexts):
        seq = torch.randint(0, cfg.embeddings.vocab_size, (seq_len,)).tolist()
        seq[seq_len // 2] = probe_token
        contexts.append(seq)

    metrics_by_layer = run_untrained_official_model_probe(model, probe_token, contexts, seed=args.seed)
    result = {
        "experiment": "context-sensitivity-untrained-baseline",
        "label": "ENGINEERING/PROXY TEST -- NOT SEMANTIC SPECIALIZATION EVIDENCE. Model is untrained (random init); "
        "these numbers describe an untrained router's response to different hidden-state inputs, not learned "
        "context-aware routing.",
        "architecture_id": cfg.architecture_id,
        "probe_token_id": probe_token,
        "n_contexts": args.n_contexts,
        "metrics_by_moe_layer": {str(k): v for k, v in metrics_by_layer.items()},
        "gate": {"harness_must_run_without_error_on_official_model": True},
        "gate_passed": len(metrics_by_layer) > 0,
    }
    _write(Path(args.output), result, args=args, config_paths=[SPARSE_PATH], seed=args.seed)


def cmd_detector_validation(args) -> None:
    healthy_shares = torch.full((8,), 1.0 / 8)
    dead_case = torch.tensor([0.0, 0.15, 0.15, 0.14, 0.14, 0.14, 0.14, 0.14])
    dominant_case = torch.tensor([0.7, 0.043, 0.043, 0.043, 0.043, 0.043, 0.043, 0.042])
    collapse_shares = torch.tensor([0.9, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01])

    cases = {
        "healthy_shares_no_detections": {
            "dead": detect_dead_experts(healthy_shares),
            "dominant": detect_dominant_experts(healthy_shares),
            "starved": detect_starved_experts(healthy_shares),
            "collapse": detect_routing_collapse(mean_normalized_entropy=0.98, load_shares=healthy_shares),
            "saturation": detect_router_saturation(mean_logit_abs=1.5, mean_top1_margin=0.2),
        },
        "dead_expert_case": {"dead": detect_dead_experts(dead_case)},
        "dominant_expert_case": {"dominant": detect_dominant_experts(dominant_case)},
        "collapse_case": {
            "collapse": detect_routing_collapse(mean_normalized_entropy=0.15, load_shares=collapse_shares)
        },
        "saturation_case": {"saturation": detect_router_saturation(mean_logit_abs=40.0, mean_top1_margin=0.2)},
    }

    gate_passed = bool(
        cases["healthy_shares_no_detections"]["dead"] == []
        and cases["healthy_shares_no_detections"]["dominant"] == []
        and cases["healthy_shares_no_detections"]["starved"] == []
        and cases["healthy_shares_no_detections"]["collapse"] is False
        and cases["healthy_shares_no_detections"]["saturation"] is False
        and cases["dead_expert_case"]["dead"] == [0]
        and cases["dominant_expert_case"]["dominant"] == [0]
        and cases["collapse_case"]["collapse"] is True
        and cases["saturation_case"]["saturation"] is True
    )

    result = {
        "experiment": "routing-health-detector-validation",
        "cases": {k: _asdict(v) for k, v in cases.items()},
        "gate": {"every_synthetic_case_must_match_its_intended_detection": True},
        "gate_passed": gate_passed,
    }
    _write(Path(args.output), result, args=args, config_paths=[SPARSE_PATH], seed=0)


def cmd_ablation_validation(args) -> None:
    device = "cpu"
    _, layer, cfg = _official_sparse_moe_layer(seed=0, device=device)
    torch.manual_seed(0)
    batch, seq_len = 1, 12
    x = torch.randn(batch, seq_len, cfg.core.d_model, device=device)
    valid = torch.ones(batch, seq_len, dtype=torch.bool, device=device)

    out_normal, _, _, _ = layer(x, valid_mask=valid)

    modes = [
        MoEAblationConfig(mode="disable_routed_expert", expert_id=0),
        MoEAblationConfig(mode="disable_shared_expert"),
        MoEAblationConfig(mode="replace_routed_expert", expert_id=0, replacement_expert_id=1),
        MoEAblationConfig(mode="uniform_router"),
        MoEAblationConfig(mode="random_router", seed=42),
        MoEAblationConfig(mode="zero_expert_output", expert_ids=(2, 3)),
    ]
    ablation_results = []
    for ablation in modes:
        out_a, _, _, _ = layer(x, valid_mask=valid, ablation=ablation)
        out_b, _, _, _ = layer(x, valid_mask=valid, ablation=ablation)
        out_normal_again, _, _, _ = layer(x, valid_mask=valid)
        ablation_results.append({
            "mode": ablation.mode,
            "differs_from_normal": bool(not torch.equal(out_a, out_normal)),
            "reproducible_across_repeated_calls": bool(torch.equal(out_a, out_b)),
            "does_not_leak_into_next_normal_call": bool(torch.equal(out_normal, out_normal_again)),
        })

    gate_passed = all(
        r["differs_from_normal"] and r["reproducible_across_repeated_calls"] and r["does_not_leak_into_next_normal_call"]
        for r in ablation_results
    )
    result = {
        "experiment": "ablation-validation",
        "architecture_id": cfg.architecture_id,
        "ablation_results": ablation_results,
        "gate": {
            "every_mode_must_change_output": True,
            "every_mode_must_be_reproducible": True,
            "no_mode_may_leak_into_normal_inference": True,
        },
        "gate_passed": gate_passed,
    }
    _write(Path(args.output), result, args=args, config_paths=[SPARSE_PATH], seed=0)


def cmd_reproducibility(args) -> None:
    device = "cpu"

    def run_once():
        model, layer, cfg = _official_sparse_moe_layer(seed=0, device=device)
        torch.manual_seed(0)
        x = torch.randn(2, 16, cfg.core.d_model, device=device)
        valid = torch.ones(2, 16, dtype=torch.bool, device=device)
        _, lb, z, diag = layer(x, valid_mask=valid, return_diagnostics=True)
        return diag, lb, z

    diag_a, lb_a, z_a = run_once()
    diag_b, lb_b, z_b = run_once()

    result = {
        "experiment": "reproducibility",
        "topk_idx_identical": bool(torch.equal(diag_a.topk_idx, diag_b.topk_idx)),
        "topk_weights_identical": bool(torch.equal(diag_a.topk_weights, diag_b.topk_weights)),
        "assignment_counts_identical": bool(
            torch.equal(diag_a.assignment_counts_per_expert, diag_b.assignment_counts_per_expert)
        ),
        "entropy_identical": bool(torch.equal(diag_a.entropy, diag_b.entropy)),
        "load_balance_loss_identical": bool(torch.equal(lb_a, lb_b)),
        "router_z_loss_identical": bool(torch.equal(z_a, z_b)),
        "gate": {"same_seed_same_config_must_reproduce_exactly": True},
        "gate_passed": None,
    }
    result["gate_passed"] = all(
        result[k] for k in ["topk_idx_identical", "topk_weights_identical", "assignment_counts_identical",
                             "entropy_identical", "load_balance_loss_identical", "router_z_loss_identical"]
    )
    _write(Path(args.output), result, args=args, config_paths=[SPARSE_PATH], seed=0)


def cmd_flowbox_moe_profile(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, layer, cfg = _official_sparse_moe_layer(seed=0, device=device)

    shapes = [(1, 128), (1, 512), (1, 1024), (1, 4096), (4, 256)]
    profiles = []
    for batch, seq_len in shapes:
        x = torch.randn(batch, seq_len, cfg.core.d_model, device=device)
        valid = torch.ones(batch, seq_len, dtype=torch.bool, device=device)

        shape_result = {"batch": batch, "seq_len": seq_len}
        for backend in ["reference", "optimized"]:
            with torch.no_grad():
                for _ in range(2):
                    layer(x, valid_mask=valid, backend=backend)
            _sync(device)
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            n_iters = 5
            with torch.no_grad():
                for _ in range(n_iters):
                    layer(x, valid_mask=valid, backend=backend)
            _sync(device)
            elapsed = (time.perf_counter() - start) / n_iters
            tokens_per_second = (batch * seq_len) / elapsed
            peak_vram = torch.cuda.max_memory_allocated() if device == "cuda" else None
            shape_result[backend] = {
                "latency_seconds": elapsed,
                "tokens_per_second": tokens_per_second,
                "peak_vram_bytes": peak_vram,
            }
            if device == "cuda":
                torch.cuda.empty_cache()
        profiles.append(shape_result)

    result = {
        "experiment": "flowbox-moe-dispatch-profile",
        "architecture_id": cfg.architecture_id,
        "device": device,
        "device_name": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "shapes_profiled": [{"batch": b, "seq_len": s} for b, s in shapes],
        "note": "prefill-only latency/throughput (single forward pass per iteration, no autoregressive decode "
        "loop exists in this reference model), warmup + CUDA sync + peak-memory-stat reset before each "
        "timed/measured block, cache emptied between backends.",
        "profiles": profiles,
    }
    _write(Path(args.output), result, args=args, config_paths=[SPARSE_PATH], seed=0)


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--output", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--allow-dirty", action="store_true")
    p.add_argument("--result-id", default=None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("equivalence")
    p.add_argument("--n-cases", type=int, default=20)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    _add_common_args(p)
    p.set_defaults(func=cmd_equivalence)

    p = sub.add_parser("routing-trace")
    _add_common_args(p)
    p.set_defaults(func=cmd_routing_trace)

    p = sub.add_parser("context-sensitivity-baseline")
    p.add_argument("--n-contexts", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    _add_common_args(p)
    p.set_defaults(func=cmd_context_sensitivity_baseline)

    p = sub.add_parser("detector-validation")
    _add_common_args(p)
    p.set_defaults(func=cmd_detector_validation)

    p = sub.add_parser("ablation-validation")
    _add_common_args(p)
    p.set_defaults(func=cmd_ablation_validation)

    p = sub.add_parser("reproducibility")
    _add_common_args(p)
    p.set_defaults(func=cmd_reproducibility)

    p = sub.add_parser("flowbox-moe-profile")
    _add_common_args(p)
    p.set_defaults(func=cmd_flowbox_moe_profile)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
