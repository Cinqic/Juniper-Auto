"""FoundationProbe -- a minimal deterministic dry run.

THIS IS NOT THE JUNIPER AUTO 150M MODEL. It is a Phase 0 foundation probe
whose only job is to prove, end to end, that:

  - imports resolve,
  - PyTorch is available and usable,
  - architecture configuration loads and validates,
  - an explicit seed can be applied and its effects are reproducible,
  - a tiny deterministic tensor operation executes and produces a stable
    result under a fixed seed,
  - structured logging functions,
  - none of the above secretly requires a GPU.

Model implementation (the actual ja150m-v0.1 Transformer) is Phase 1 scope.
Do not extend this class into a real model -- add a new module under a
future `model/` implementation instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from juniper_auto.config.schema import ArchitectureConfig
from juniper_auto.util.environment import EnvironmentIdentity, describe_environment
from juniper_auto.util.logging import LogContext, current_git_commit, get_logger, log_event
from juniper_auto.util.seed import SeedReport, apply_seed
import logging as _logging


@dataclass(frozen=True)
class FoundationProbeResult:
    seed_report: SeedReport
    environment: EnvironmentIdentity
    architecture_id: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    output_checksum: float
    device: str


class FoundationProbe(nn.Module):
    """A tiny, architecturally-unrelated linear+nonlinearity stack sized
    from the loaded config's `d_model`, used only to exercise the tensor
    pipeline end to end. It intentionally does not implement attention,
    MoE routing, RoPE, or any other ja150m-v0.1 mechanism."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.linear_in = nn.Linear(d_model, d_model, bias=False)
        self.activation = nn.GELU()
        self.linear_out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_out(self.activation(self.linear_in(x)))


def run_foundation_probe(
    cfg: ArchitectureConfig,
    *,
    seed: int,
    batch_size: int = 2,
    seq_len: int = 8,
    device: str | None = None,
) -> FoundationProbeResult:
    """Run the foundation probe under an explicit seed and return a result
    whose `output_checksum` is expected to be identical across runs with the
    same seed, device, and PyTorch/CUDA version (see
    juniper_auto.util.seed for the determinism caveats)."""
    logger = get_logger("juniper_auto.foundation.probe")
    seed_report = apply_seed(seed)
    env = describe_environment()

    resolved_device = device or ("cuda" if env.cuda_available else "cpu")
    torch_device = torch.device(resolved_device)

    log_event(
        logger,
        _logging.INFO,
        "foundation_probe.start",
        LogContext(
            phase="phase-0",
            run_id=f"foundation-probe-seed-{seed}",
            git_commit=current_git_commit(),
            config_id=cfg.architecture_id,
            architecture_id=cfg.architecture_id,
            seed=seed,
            env_id=env.short_id(),
            extra={"device": resolved_device},
        ),
    )

    model = FoundationProbe(cfg.core.d_model).to(torch_device)
    model.eval()

    x = torch.randn(batch_size, seq_len, cfg.core.d_model, device=torch_device)
    with torch.no_grad():
        y = model(x)

    checksum = float(y.sum().item())

    log_event(
        logger,
        _logging.INFO,
        "foundation_probe.complete",
        LogContext(
            phase="phase-0",
            run_id=f"foundation-probe-seed-{seed}",
            git_commit=current_git_commit(),
            config_id=cfg.architecture_id,
            architecture_id=cfg.architecture_id,
            seed=seed,
            env_id=env.short_id(),
            extra={"device": resolved_device, "output_checksum": checksum},
        ),
    )

    return FoundationProbeResult(
        seed_report=seed_report,
        environment=env,
        architecture_id=cfg.architecture_id,
        input_shape=tuple(x.shape),
        output_shape=tuple(y.shape),
        output_checksum=checksum,
        device=resolved_device,
    )
