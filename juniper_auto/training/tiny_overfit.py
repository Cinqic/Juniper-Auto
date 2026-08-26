"""Tiny-batch overfit harness: proves a full model can memorize a small,
deterministic synthetic training set, exercising forward, backward,
optimizer update, gradient clipping, and (optionally) FP16 AMP + dynamic
loss scaling end to end. Also the harness used for the checkpoint/resume
equivalence experiment (see juniper_auto.training.checkpoint).

This is Phase 1 engineering-test training, not a production optimization
recipe: hyperparameters here are explicitly recorded per run rather than
frozen as a later phase's pretraining configuration.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from juniper_auto.config.schema import ArchitectureConfig
from juniper_auto.model import build_model
from juniper_auto.training.checkpoint import build_checkpoint_payload, restore_from_checkpoint
from juniper_auto.training.state import SyntheticSequenceStream, capture_rng_state
from juniper_auto.util.seed import apply_seed


@dataclass
class TinyOverfitConfig:
    seed: int
    vocab_size: int
    seq_len: int
    n_sequences: int
    batch_size: int
    lr: float
    max_steps: int
    grad_clip_norm: float
    weight_decay: float = 0.0
    use_amp: bool = False
    device: str = "cpu"


@dataclass
class TinyOverfitStepRecord:
    step: int
    loss: float
    lm_loss: float
    load_balance_loss: float
    router_z_loss: float
    token_accuracy: float
    grad_norm: float


@dataclass
class TinyOverfitResult:
    config: TinyOverfitConfig
    architecture_id: str
    starting_lm_loss: float
    ending_lm_loss: float
    best_lm_loss: float
    ending_token_accuracy: float
    best_token_accuracy: float
    steps_run: int
    elapsed_seconds: float
    peak_vram_bytes: int | None
    any_nonfinite_event: bool
    final_parameters_finite: bool
    global_valid_token_count: int
    history: list[TinyOverfitStepRecord] = field(default_factory=list)


def compute_token_accuracy(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> float:
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]
    preds = shift_logits.argmax(dim=-1)
    valid = shift_labels != ignore_index
    total = int(valid.sum().item())
    if total == 0:
        return float("nan")
    correct = int(((preds == shift_labels) & valid).sum().item())
    return correct / total


class TinyOverfitHarness:
    """Owns the model/optimizer/scaler/data-stream for one tiny-overfit run
    and supports mid-run checkpointing so the same harness code drives both
    the plain overfit experiment and the interrupted/resumed equivalence
    experiment."""

    def __init__(self, architecture_cfg: ArchitectureConfig, run_cfg: TinyOverfitConfig):
        self.architecture_cfg = architecture_cfg
        self.run_cfg = run_cfg
        apply_seed(run_cfg.seed, deterministic_algorithms=True)

        self.model = build_model(architecture_cfg, device=run_cfg.device, seed=run_cfg.seed)
        self.model.train()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=run_cfg.lr, weight_decay=run_cfg.weight_decay
        )
        self.stream = SyntheticSequenceStream(
            seed=run_cfg.seed,
            vocab_size=run_cfg.vocab_size,
            seq_len=run_cfg.seq_len,
            n_sequences=run_cfg.n_sequences,
            batch_size=run_cfg.batch_size,
        )

        self.device_type = torch.device(run_cfg.device).type
        self.use_amp = run_cfg.use_amp and self.device_type == "cuda"
        self.scaler = torch.amp.GradScaler(self.device_type, enabled=self.use_amp) if self.device_type == "cuda" else None

        self.global_step = 0
        self.global_valid_token_count = 0
        self.sequence_curriculum_state = {"status": "not-implemented-phase-1"}

    def train_step(self) -> tuple[TinyOverfitStepRecord, bool]:
        batch = self.stream.next_batch().to(self.run_cfg.device)
        labels = batch.clone()

        self.optimizer.zero_grad(set_to_none=True)
        if self.use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = self.model(batch, labels=labels)
            self.scaler.scale(out.loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.run_cfg.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            out = self.model(batch, labels=labels)
            out.loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.run_cfg.grad_clip_norm)
            self.optimizer.step()

        token_accuracy = compute_token_accuracy(out.logits.detach(), labels)
        n_valid_tokens = int(labels[..., 1:].numel())  # synthetic stream has no padding
        self.global_valid_token_count += n_valid_tokens
        self.global_step += 1
        nonfinite = not bool(
            torch.isfinite(out.loss).item()
            and torch.isfinite(out.lm_loss).item()
            and torch.isfinite(grad_norm).item()
        )

        record = TinyOverfitStepRecord(
            step=self.global_step,
            loss=float(out.loss.item()),
            lm_loss=float(out.lm_loss.item()),
            load_balance_loss=float(out.load_balance_loss.item()),
            router_z_loss=float(out.router_z_loss.item()),
            token_accuracy=token_accuracy,
            grad_norm=float(grad_norm),
        )
        return record, nonfinite

    def checkpoint_payload(self, *, git_commit: str, dataset_identity: str, tokenizer_identity: str = "not-yet-created") -> dict:
        training_config = {
            "seed": self.run_cfg.seed,
            "vocab_size": self.run_cfg.vocab_size,
            "seq_len": self.run_cfg.seq_len,
            "n_sequences": self.run_cfg.n_sequences,
            "batch_size": self.run_cfg.batch_size,
            "lr": self.run_cfg.lr,
            "grad_clip_norm": self.run_cfg.grad_clip_norm,
            "weight_decay": self.run_cfg.weight_decay,
            "use_amp": self.run_cfg.use_amp,
            "device": self.run_cfg.device,
            "optimizer": "AdamW",
        }
        return build_checkpoint_payload(
            model=self.model,
            optimizer=self.optimizer,
            scaler=self.scaler,
            scheduler=None,
            rng_state=capture_rng_state(),
            sampler_state=self.stream.state_dict(),
            global_step=self.global_step,
            global_valid_token_count=self.global_valid_token_count,
            architecture_id=self.architecture_cfg.architecture_id,
            architecture_config_dict=self.architecture_cfg.model_dump(),
            training_config=training_config,
            git_commit=git_commit,
            dataset_identity=dataset_identity,
            tokenizer_identity=tokenizer_identity,
            sequence_curriculum_state=self.sequence_curriculum_state,
        )

    def load_checkpoint_payload(self, payload: dict) -> None:
        resumed = restore_from_checkpoint(payload, model=self.model, optimizer=self.optimizer, scaler=self.scaler)
        self.global_step = resumed["global_step"]
        self.global_valid_token_count = resumed["global_valid_token_count"]
        self.sequence_curriculum_state = resumed["sequence_curriculum_state"]
        self.stream.load_state_dict(resumed["sampler_state"])


def run_tiny_overfit(architecture_cfg: ArchitectureConfig, run_cfg: TinyOverfitConfig) -> TinyOverfitResult:
    harness = TinyOverfitHarness(architecture_cfg, run_cfg)
    if harness.device_type == "cuda":
        torch.cuda.reset_peak_memory_stats(run_cfg.device)

    history: list[TinyOverfitStepRecord] = []
    any_nonfinite = False
    start = time.perf_counter()
    for _ in range(run_cfg.max_steps):
        record, nonfinite = harness.train_step()
        history.append(record)
        any_nonfinite = any_nonfinite or nonfinite
    elapsed = time.perf_counter() - start

    peak_vram = torch.cuda.max_memory_allocated(run_cfg.device) if harness.device_type == "cuda" else None
    lm_losses = [r.lm_loss for r in history]
    accuracies = [r.token_accuracy for r in history]
    final_parameters_finite = all(torch.isfinite(p).all().item() for p in harness.model.parameters())

    return TinyOverfitResult(
        config=run_cfg,
        architecture_id=architecture_cfg.architecture_id,
        starting_lm_loss=lm_losses[0],
        ending_lm_loss=lm_losses[-1],
        best_lm_loss=min(lm_losses),
        ending_token_accuracy=accuracies[-1],
        best_token_accuracy=max(accuracies),
        steps_run=len(history),
        elapsed_seconds=elapsed,
        peak_vram_bytes=peak_vram,
        any_nonfinite_event=any_nonfinite,
        final_parameters_finite=final_parameters_finite,
        global_valid_token_count=harness.global_valid_token_count,
        history=history,
    )
