#!/usr/bin/env python3
"""Run a single Phase 1 experiment against the real official architectures
and write a JSON result artifact.

Usage:
    python scripts/run_phase1_experiment.py param-verification --output <path>
    python scripts/run_phase1_experiment.py dense-overfit --output <path> [--device cuda|cpu]
    python scripts/run_phase1_experiment.py sparse-overfit --output <path> [--device cuda|cpu]
    python scripts/run_phase1_experiment.py resume-equivalence --output <path> [--device cpu]
    python scripts/run_phase1_experiment.py profile --architecture sparse|dense --output <path>

These are real, executed runs against ja150m-v0.1 / ja150m-v0.1-dense --
not estimates. Results (including any negative/failed outcome) are
written verbatim to the given JSON path for the experiment registry to
reference.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from juniper_auto.config import load_architecture_config  # noqa: E402
from juniper_auto.accounting import standard_active_parameter_breakdown, total_parameter_breakdown  # noqa: E402
from juniper_auto.model import build_model  # noqa: E402
from juniper_auto.model.inspection import pytorch_parameter_breakdown, total_parameters, verify_weight_tying  # noqa: E402
from juniper_auto.training.checkpoint import restore_from_checkpoint  # noqa: E402
from juniper_auto.training.profiling import profile_checkpoint_io, profile_inference, profile_training_step  # noqa: E402
from juniper_auto.training.tiny_overfit import TinyOverfitConfig, TinyOverfitHarness, run_tiny_overfit  # noqa: E402
from juniper_auto.util.environment import describe_environment  # noqa: E402
from juniper_auto.util.hashing import sha256_file  # noqa: E402

SPARSE_PATH = REPO_ROOT / "configs/architecture/ja150m-v0.1.yaml"
DENSE_PATH = REPO_ROOT / "configs/architecture/ja150m-v0.1-dense.yaml"


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
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    return out.stdout


def _asdict(obj):
    if dataclasses.is_dataclass(obj):
        return {k: _asdict(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, list):
        return [_asdict(v) for v in obj]
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


def _write(path: Path, payload: dict, *, args, config_paths: list[Path], seed: int) -> None:
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
        "run_configuration": {
            key: value
            for key, value in vars(args).items()
            if key not in {"func", "output", "allow_dirty", "overwrite", "result_id"}
        },
        "seed": seed,
        **payload,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    print(f"wrote {path}")


def cmd_param_verification(args) -> None:
    sparse_cfg = load_architecture_config(SPARSE_PATH)
    dense_cfg = load_architecture_config(DENSE_PATH)

    sparse_model = build_model(sparse_cfg, seed=0)
    dense_model = build_model(dense_cfg, seed=0)

    method_a_sparse_total = total_parameter_breakdown(sparse_cfg).as_dict()
    method_a_sparse_active = standard_active_parameter_breakdown(sparse_cfg).as_dict()
    method_a_dense_total = total_parameter_breakdown(dense_cfg).as_dict()

    method_b_sparse = pytorch_parameter_breakdown(sparse_model)
    method_b_dense = pytorch_parameter_breakdown(dense_model)

    result = {
        "experiment": "param-verification",
        "sparse": {
            "method_a_total_breakdown": method_a_sparse_total,
            "method_a_standard_active_breakdown": method_a_sparse_active,
            "method_b_total_breakdown": method_b_sparse,
            "method_b_total_via_model_parameters": total_parameters(sparse_model),
            "totals_match": method_a_sparse_total["total"] == method_b_sparse["total"] == total_parameters(sparse_model),
            "expected_total": 150_031_360,
            "expected_standard_active": 79_252_480,
            "weight_tying_verified": verify_weight_tying(sparse_model),
        },
        "dense": {
            "method_a_total_breakdown": method_a_dense_total,
            "method_b_total_breakdown": method_b_dense,
            "method_b_total_via_model_parameters": total_parameters(dense_model),
            "totals_match": method_a_dense_total["total"] == method_b_dense["total"] == total_parameters(dense_model),
            "expected_total": 79_191_040,
            "weight_tying_verified": verify_weight_tying(dense_model),
        },
    }
    result["gate"] = {
        "sparse_total_expected": 150_031_360,
        "sparse_standard_active_expected": 79_252_480,
        "dense_total_expected": 79_191_040,
        "require_weight_tying": True,
    }
    result["gate_passed"] = bool(
        method_a_sparse_total["total"] == method_b_sparse["total"] == 150_031_360
        and method_a_sparse_active["total"] == 79_252_480
        and method_a_dense_total["total"] == method_b_dense["total"] == 79_191_040
        and verify_weight_tying(sparse_model)
        and verify_weight_tying(dense_model)
    )
    _write(Path(args.output), result, args=args, config_paths=[SPARSE_PATH, DENSE_PATH], seed=0)


def cmd_overfit(args, architecture_path: Path) -> None:
    cfg = load_architecture_config(architecture_path)
    device = args.device
    use_amp = device == "cuda"
    run_cfg = TinyOverfitConfig(
        seed=args.seed,
        vocab_size=cfg.embeddings.vocab_size,
        seq_len=args.seq_len,
        n_sequences=args.n_sequences,
        batch_size=args.batch_size,
        lr=args.lr,
        max_steps=args.max_steps,
        grad_clip_norm=args.grad_clip_norm,
        use_amp=use_amp,
        device=device,
    )
    result = run_tiny_overfit(cfg, run_cfg)
    loss_reduction_fraction = 1.0 - (result.ending_lm_loss / result.starting_lm_loss)
    gate = {
        "max_ending_lm_loss": args.max_ending_lm_loss,
        "min_ending_token_accuracy": args.min_ending_token_accuracy,
        "min_loss_reduction_fraction": args.min_loss_reduction_fraction,
        "require_no_nonfinite_event": True,
        "require_final_parameters_finite": True,
    }
    gate_passed = bool(
        result.ending_lm_loss <= args.max_ending_lm_loss
        and result.ending_token_accuracy >= args.min_ending_token_accuracy
        and loss_reduction_fraction >= args.min_loss_reduction_fraction
        and not result.any_nonfinite_event
        and result.final_parameters_finite
    )
    _write(
        Path(args.output),
        {
            "experiment": f"{cfg.kind}-tiny-overfit",
            "architecture_id": cfg.architecture_id,
            "result": _asdict(result),
            "loss_reduction_fraction": loss_reduction_fraction,
            "gate": gate,
            "gate_passed": gate_passed,
        },
        args=args,
        config_paths=[architecture_path],
        seed=args.seed,
    )


def cmd_resume_equivalence(args) -> None:
    cfg = load_architecture_config(SPARSE_PATH if args.architecture == "sparse" else DENSE_PATH)
    device = args.device

    def run_cfg(seed):
        return TinyOverfitConfig(
            seed=seed,
            vocab_size=cfg.embeddings.vocab_size,
            seq_len=args.seq_len,
            n_sequences=args.n_sequences,
            batch_size=args.batch_size,
            lr=args.lr,
            max_steps=args.max_steps,
            grad_clip_norm=args.grad_clip_norm,
            use_amp=False,
            device=device,
        )

    total_steps = args.max_steps
    split = total_steps // 2

    harness_a = TinyOverfitHarness(cfg, run_cfg(args.seed))
    history_a = [harness_a.train_step()[0] for _ in range(total_steps)]

    harness_b = TinyOverfitHarness(cfg, run_cfg(args.seed))
    for _ in range(split):
        harness_b.train_step()
    payload = harness_b.checkpoint_payload(
        git_commit=_git_commit(), dataset_identity="synthetic Phase 1 engineering data"
    )
    del harness_b

    harness_resumed = TinyOverfitHarness(cfg, run_cfg(args.seed))
    harness_resumed.load_checkpoint_payload(payload)
    sampler_state_after_restore = harness_resumed.stream.state_dict()
    expected_stream = type(harness_resumed.stream)(
        seed=args.seed,
        vocab_size=cfg.embeddings.vocab_size,
        seq_len=args.seq_len,
        n_sequences=args.n_sequences,
        batch_size=args.batch_size,
    )
    expected_stream.load_state_dict(payload["sampler_state"])
    next_batch_identity_match = torch.equal(
        expected_stream.next_batch(), harness_resumed.stream.next_batch()
    )
    harness_resumed.stream.load_state_dict(sampler_state_after_restore)
    history_resumed = [harness_resumed.train_step()[0] for _ in range(total_steps - split)]

    losses_a = [r.loss for r in history_a[split:]]
    losses_resumed = [r.loss for r in history_resumed]
    exact_loss_match = losses_a == losses_resumed

    max_param_diff = max(
        (p_a - p_b).abs().max().item()
        for p_a, p_b in zip(harness_a.model.parameters(), harness_resumed.model.parameters())
    )
    optimizer_state_exact_match = True
    for state_a, state_b in zip(harness_a.optimizer.state.values(), harness_resumed.optimizer.state.values()):
        if state_a.keys() != state_b.keys():
            optimizer_state_exact_match = False
            break
        for key in state_a:
            a, b = state_a[key], state_b[key]
            if isinstance(a, torch.Tensor):
                if not torch.equal(a, b):
                    optimizer_state_exact_match = False
                    break
            elif a != b:
                optimizer_state_exact_match = False
                break

    result = {
        "experiment": "resume-equivalence",
        "architecture_id": cfg.architecture_id,
        "device": device,
        "total_steps": total_steps,
        "split_at_step": split,
        "exact_loss_match": exact_loss_match,
        "losses_uninterrupted_tail": losses_a,
        "losses_resumed": losses_resumed,
        "max_parameter_abs_diff": max_param_diff,
        "global_step_match": harness_a.global_step == harness_resumed.global_step,
        "global_valid_token_count_match": harness_a.global_valid_token_count == harness_resumed.global_valid_token_count,
        "optimizer_state_exact_match": optimizer_state_exact_match,
        "next_batch_identity_match": next_batch_identity_match,
        "sequence_curriculum_state_preserved": harness_resumed.sequence_curriculum_state
        == payload["sequence_curriculum_state"],
    }
    result["gate"] = {
        "require_exact_loss_match": True,
        "max_parameter_abs_diff": 0.0,
        "require_optimizer_state_exact_match": True,
        "require_step_token_and_next_batch_identity_match": True,
        "require_sequence_curriculum_state_preserved": True,
    }
    result["gate_passed"] = bool(
        exact_loss_match
        and max_param_diff == 0.0
        and result["global_step_match"]
        and result["global_valid_token_count_match"]
        and optimizer_state_exact_match
        and next_batch_identity_match
        and result["sequence_curriculum_state_preserved"]
    )
    config_path = SPARSE_PATH if args.architecture == "sparse" else DENSE_PATH
    _write(Path(args.output), result, args=args, config_paths=[config_path], seed=args.seed)


def cmd_profile(args) -> None:
    cfg = load_architecture_config(SPARSE_PATH if args.architecture == "sparse" else DENSE_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    result: dict = {"experiment": "flowbox-profile", "architecture_id": cfg.architecture_id, "device": device}

    inference_results = []
    for precision_label, dtype in [("fp32", torch.float32), ("fp16", torch.float16)]:
        if dtype == torch.float16 and device != "cuda":
            continue
        model = build_model(cfg, seed=0, device=device, dtype=dtype)
        prof = profile_inference(
            model,
            vocab_size=cfg.embeddings.vocab_size,
            batch_size=args.inference_batch_size,
            seq_len=args.inference_seq_len,
            device=device,
            precision_label=precision_label,
        )
        inference_results.append(_asdict(prof))
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    result["inference"] = inference_results

    model = build_model(cfg, seed=0, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    training_prof = profile_training_step(
        model,
        optimizer,
        vocab_size=cfg.embeddings.vocab_size,
        microbatch_size=args.train_microbatch_size,
        seq_len=args.train_seq_len,
        grad_accumulation_steps=args.grad_accumulation_steps,
        device=device,
        use_amp=(device == "cuda"),
        activation_checkpointing=args.activation_checkpointing,
    )
    result["training_step"] = _asdict(training_prof)
    # Release the full training model + AdamW optimizer state before the
    # next stage builds another full model -- otherwise both are resident
    # on the GPU simultaneously and an OOM here would misrepresent a
    # profiling-script memory-hygiene bug as a hardware limitation.
    del model, optimizer
    if device == "cuda":
        torch.cuda.empty_cache()

    checkpoint_harness = TinyOverfitHarness(
        cfg,
        TinyOverfitConfig(
            seed=0,
            vocab_size=cfg.embeddings.vocab_size,
            seq_len=8,
            n_sequences=4,
            batch_size=2,
            lr=1e-4,
            max_steps=1,
            grad_clip_norm=1.0,
            use_amp=False,
            device=device,
        ),
    )
    checkpoint_harness.train_step()
    payload = checkpoint_harness.checkpoint_payload(
        git_commit=_git_commit(), dataset_identity="synthetic Phase 1 engineering data"
    )
    ckpt_path = Path(args.output).with_suffix(".checkpoint_probe.pt")
    ckpt_prof = profile_checkpoint_io(payload, ckpt_path)
    result["checkpoint_io"] = _asdict(ckpt_prof)
    ckpt_path.unlink(missing_ok=True)
    del checkpoint_harness, payload
    if device == "cuda":
        torch.cuda.empty_cache()

    # Attempt the full official-model batch-1 4096-token reference inference
    # path; record success or the actual limitation honestly rather than
    # redesigning the architecture around it.
    context_probe: dict = {"batch_size": 1, "seq_len": cfg.attention.context_length}
    try:
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        probe_model = build_model(cfg, seed=0, device=device, dtype=torch.float16 if device == "cuda" else None)
        probe_model.eval()
        input_ids = torch.randint(0, cfg.embeddings.vocab_size, (1, cfg.attention.context_length), device=device)
        with torch.no_grad():
            probe_model(input_ids)
        if device == "cuda":
            torch.cuda.synchronize(device)
        context_probe["status"] = "success"
        context_probe["peak_vram_bytes"] = torch.cuda.max_memory_allocated(device) if device == "cuda" else None
    except torch.cuda.OutOfMemoryError as exc:
        context_probe["status"] = "oom"
        context_probe["detail"] = str(exc)
    except Exception as exc:  # pragma: no cover - defensive, records honestly rather than crashing the run
        context_probe["status"] = "error"
        context_probe["detail"] = repr(exc)
    result["full_context_4096_batch1_probe"] = context_probe

    result["gate"] = {
        "require_fp32_and_fp16_prefill_profiles_on_cuda": True,
        "require_finite_training_step": True,
        "require_checkpoint_write_and_load": True,
        "require_full_context_4096_batch1_success": True,
    }
    precisions = {entry["precision_label"] for entry in inference_results}
    result["gate_passed"] = bool(
        device == "cuda"
        and precisions == {"fp32", "fp16"}
        and result["training_step"]["numerical_finite"]
        and result["checkpoint_io"]["file_size_bytes"] > 0
        and context_probe.get("status") == "success"
    )

    config_path = SPARSE_PATH if args.architecture == "sparse" else DENSE_PATH
    _write(Path(args.output), result, args=args, config_paths=[config_path], seed=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_result_args(p):
        p.add_argument("--output", required=True)
        p.add_argument("--result-id")
        p.add_argument("--allow-dirty", action="store_true")
        p.add_argument("--overwrite", action="store_true")

    p = sub.add_parser("param-verification")
    add_result_args(p)
    p.set_defaults(func=cmd_param_verification)

    for name, path in [("dense-overfit", DENSE_PATH), ("sparse-overfit", SPARSE_PATH)]:
        p = sub.add_parser(name)
        add_result_args(p)
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        p.add_argument("--seed", type=int, default=1234)
        p.add_argument("--seq-len", type=int, default=32)
        p.add_argument("--n-sequences", type=int, default=8)
        p.add_argument("--batch-size", type=int, default=4)
        p.add_argument("--lr", type=float, default=3e-3)
        p.add_argument("--max-steps", type=int, default=300)
        p.add_argument("--grad-clip-norm", type=float, default=1.0)
        p.add_argument("--max-ending-lm-loss", type=float, default=0.01)
        p.add_argument("--min-ending-token-accuracy", type=float, default=0.99)
        p.add_argument("--min-loss-reduction-fraction", type=float, default=0.95)
        p.set_defaults(func=lambda a, path=path: cmd_overfit(a, path))

    p = sub.add_parser("resume-equivalence")
    add_result_args(p)
    p.add_argument("--architecture", choices=["sparse", "dense"], default="sparse")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--n-sequences", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.set_defaults(func=cmd_resume_equivalence)

    p = sub.add_parser("profile")
    add_result_args(p)
    p.add_argument("--architecture", choices=["sparse", "dense"], required=True)
    p.add_argument("--inference-batch-size", type=int, default=1)
    p.add_argument("--inference-seq-len", type=int, default=512)
    p.add_argument("--train-microbatch-size", type=int, default=2)
    p.add_argument("--train-seq-len", type=int, default=512)
    p.add_argument("--grad-accumulation-steps", type=int, default=4)
    p.add_argument("--activation-checkpointing", action="store_true")
    p.set_defaults(func=cmd_profile)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
