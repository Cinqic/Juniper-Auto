"""FLOWBOX hardware profiling utilities: inference throughput/latency,
training step timing (forward/backward/optimizer separately), and
checkpoint I/O timing. Always synchronizes CUDA before/after timed
sections and runs untimed warmup iterations first, so steady-state timing
is not contaminated by one-time CUDA context/kernel-compilation cost.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch


def _sync(device: str | torch.device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _host_rss_bytes() -> int | None:
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024  # Linux: ru_maxrss is KB
    except Exception:
        return None


@dataclass
class InferenceProfileResult:
    precision_label: str
    batch_size: int
    seq_len: int
    warmup_iters: int
    measured_iters: int
    prefill_latency_seconds: float
    prefill_tokens_per_second: float
    peak_vram_bytes: int | None
    host_peak_rss_bytes: int | None


def profile_inference(
    model: torch.nn.Module,
    *,
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    device: str,
    precision_label: str,
    warmup_iters: int = 3,
    measured_iters: int = 10,
) -> InferenceProfileResult:
    """Profiles `model` exactly as passed in (dtype/device already set by
    the caller, e.g. via `build_model(cfg, device=..., dtype=...)`) --
    `precision_label` is only a record-keeping label, not something this
    function enforces."""
    model.eval()
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    with torch.no_grad():
        for _ in range(warmup_iters):
            model(input_ids)
        _sync(device)

        if torch.device(device).type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        start = time.perf_counter()
        for _ in range(measured_iters):
            model(input_ids)
        _sync(device)
        elapsed = time.perf_counter() - start

    per_iter_seconds = elapsed / measured_iters
    tokens_per_second = (batch_size * seq_len) / per_iter_seconds
    peak_vram = torch.cuda.max_memory_allocated(device) if torch.device(device).type == "cuda" else None

    return InferenceProfileResult(
        precision_label=precision_label,
        batch_size=batch_size,
        seq_len=seq_len,
        warmup_iters=warmup_iters,
        measured_iters=measured_iters,
        prefill_latency_seconds=per_iter_seconds,
        prefill_tokens_per_second=tokens_per_second,
        peak_vram_bytes=peak_vram,
        host_peak_rss_bytes=_host_rss_bytes(),
    )


@dataclass
class TrainingStepProfileResult:
    microbatch_size: int
    seq_len: int
    grad_accumulation_steps: int
    activation_checkpointing: bool
    use_amp: bool
    grad_clip_norm: float
    warmup_iters: int
    measured_iters: int
    forward_seconds: float
    backward_seconds: float
    optimizer_step_seconds: float
    total_step_seconds: float
    training_tokens_per_second: float
    peak_vram_bytes: int | None
    host_peak_rss_bytes: int | None
    numerical_finite: bool


def profile_training_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    vocab_size: int,
    microbatch_size: int,
    seq_len: int,
    grad_accumulation_steps: int,
    device: str,
    use_amp: bool,
    activation_checkpointing: bool = False,
    grad_clip_norm: float = 1.0,
    warmup_iters: int = 2,
    measured_iters: int = 5,
) -> TrainingStepProfileResult:
    model.train()
    if hasattr(model, "set_gradient_checkpointing"):
        model.set_gradient_checkpointing(activation_checkpointing)

    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and torch.device(device).type == "cuda"))

    def make_microbatch():
        ids = torch.randint(0, vocab_size, (microbatch_size, seq_len), device=device)
        return ids, ids.clone()

    def run_full_step() -> tuple[float, float, float]:
        optimizer.zero_grad(set_to_none=True)
        forward_time = 0.0
        backward_time = 0.0
        for _ in range(grad_accumulation_steps):
            ids, labels = make_microbatch()
            _sync(device)
            t0 = time.perf_counter()
            if use_amp and torch.device(device).type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = model(ids, labels=labels)
            else:
                out = model(ids, labels=labels)
            if not torch.isfinite(out.loss):
                raise RuntimeError("non-finite loss encountered during training profile")
            _sync(device)
            t1 = time.perf_counter()

            loss = out.loss / grad_accumulation_steps
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            _sync(device)
            t2 = time.perf_counter()

            forward_time += t1 - t0
            backward_time += t2 - t1

        t3 = time.perf_counter()
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            if not torch.isfinite(grad_norm):
                raise RuntimeError("non-finite gradient norm encountered during training profile")
            scaler.step(optimizer)
            scaler.update()
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            if not torch.isfinite(grad_norm):
                raise RuntimeError("non-finite gradient norm encountered during training profile")
            optimizer.step()
        _sync(device)
        t4 = time.perf_counter()
        return forward_time, backward_time, t4 - t3

    for _ in range(warmup_iters):
        run_full_step()

    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    forward_total = backward_total = optimizer_total = 0.0
    for _ in range(measured_iters):
        f, b, o = run_full_step()
        forward_total += f
        backward_total += b
        optimizer_total += o

    forward_avg = forward_total / measured_iters
    backward_avg = backward_total / measured_iters
    optimizer_avg = optimizer_total / measured_iters
    total_step = forward_avg + backward_avg + optimizer_avg
    tokens_per_step = microbatch_size * seq_len * grad_accumulation_steps
    tokens_per_second = tokens_per_step / total_step
    peak_vram = torch.cuda.max_memory_allocated(device) if torch.device(device).type == "cuda" else None
    parameters_finite = all(torch.isfinite(p).all().item() for p in model.parameters())

    return TrainingStepProfileResult(
        microbatch_size=microbatch_size,
        seq_len=seq_len,
        grad_accumulation_steps=grad_accumulation_steps,
        activation_checkpointing=activation_checkpointing,
        use_amp=use_amp,
        grad_clip_norm=grad_clip_norm,
        warmup_iters=warmup_iters,
        measured_iters=measured_iters,
        forward_seconds=forward_avg,
        backward_seconds=backward_avg,
        optimizer_step_seconds=optimizer_avg,
        total_step_seconds=total_step,
        training_tokens_per_second=tokens_per_second,
        peak_vram_bytes=peak_vram,
        host_peak_rss_bytes=_host_rss_bytes(),
        numerical_finite=parameters_finite,
    )


@dataclass
class CheckpointIOProfileResult:
    checkpoint_path: str
    file_size_bytes: int
    write_seconds: float
    load_seconds: float


def profile_checkpoint_io(payload: dict, path: str | Path) -> CheckpointIOProfileResult:
    from juniper_auto.training.checkpoint import load_checkpoint, save_checkpoint

    path = Path(path)
    start = time.perf_counter()
    save_checkpoint(path, payload)
    write_seconds = time.perf_counter() - start

    start = time.perf_counter()
    load_checkpoint(path)
    load_seconds = time.perf_counter() - start

    return CheckpointIOProfileResult(
        checkpoint_path=str(path),
        file_size_bytes=path.stat().st_size,
        write_seconds=write_seconds,
        load_seconds=load_seconds,
    )
