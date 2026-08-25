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


def _asdict(obj):
    if dataclasses.is_dataclass(obj):
        return {k: _asdict(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, list):
        return [_asdict(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    return obj


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "git_commit": _git_commit(),
        "environment": describe_environment().as_dict(),
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
    _write(Path(args.output), result)


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
    _write(
        Path(args.output),
        {
            "experiment": f"{cfg.kind}-tiny-overfit",
            "architecture_id": cfg.architecture_id,
            "result": _asdict(result),
        },
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
    history_resumed = [harness_resumed.train_step()[0] for _ in range(total_steps - split)]

    losses_a = [r.loss for r in history_a[split:]]
    losses_resumed = [r.loss for r in history_resumed]
    exact_loss_match = losses_a == losses_resumed

    max_param_diff = max(
        (p_a - p_b).abs().max().item()
        for p_a, p_b in zip(harness_a.model.parameters(), harness_resumed.model.parameters())
    )

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
    }
    _write(Path(args.output), result)


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

    _write(Path(args.output), result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("param-verification")
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_param_verification)

    for name, path in [("dense-overfit", DENSE_PATH), ("sparse-overfit", SPARSE_PATH)]:
        p = sub.add_parser(name)
        p.add_argument("--output", required=True)
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        p.add_argument("--seed", type=int, default=1234)
        p.add_argument("--seq-len", type=int, default=32)
        p.add_argument("--n-sequences", type=int, default=8)
        p.add_argument("--batch-size", type=int, default=4)
        p.add_argument("--lr", type=float, default=3e-3)
        p.add_argument("--max-steps", type=int, default=300)
        p.add_argument("--grad-clip-norm", type=float, default=1.0)
        p.set_defaults(func=lambda a, path=path: cmd_overfit(a, path))

    p = sub.add_parser("resume-equivalence")
    p.add_argument("--output", required=True)
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
    p.add_argument("--output", required=True)
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
