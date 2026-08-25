"""Deterministic seed framework.

Seeds must always be explicit and recorded -- this module never silently
generates a hidden seed. It distinguishes strict determinism (CPU-only,
where PyTorch guarantees bitwise reproducibility for the ops we use) from
best-effort reproducibility (CUDA, where cuDNN/kernel nondeterminism means
"same seed" reduces but does not guarantee eliminate run-to-run variance).
Do not claim universal bitwise GPU determinism -- it has not been
established for this project.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class SeedReport:
    seed: int
    python_random_seeded: bool
    numpy_seeded: bool
    torch_cpu_seeded: bool
    torch_cuda_seeded: bool
    cuda_available: bool
    deterministic_algorithms_requested: bool
    strict_determinism_claim: str


def apply_seed(seed: int, *, deterministic_algorithms: bool = True) -> SeedReport:
    """Apply an explicit seed across Python, NumPy, and PyTorch (CPU and,
    if available, CUDA). Returns a SeedReport recording exactly what was
    seeded, so callers/logs never have to guess.

    `seed` must be provided explicitly by the caller -- there is no default
    seed and no fallback to system entropy here.
    """
    if not isinstance(seed, int):
        raise TypeError(f"seed must be an explicit int, got {type(seed)!r}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.manual_seed_all(seed)

    if deterministic_algorithms:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)

    if cuda_available:
        claim = (
            "best-effort: CPU-side operations are expected to be bitwise "
            "reproducible under this seed; CUDA/cuDNN kernel-level "
            "determinism is not guaranteed and has not been independently "
            "verified for this project"
        )
    else:
        claim = (
            "best-effort: CPU-only execution; standard PyTorch CPU ops are "
            "expected to be bitwise reproducible under this seed for the "
            "operations exercised by the Phase 0 foundation probe, but this "
            "is not a universal determinism guarantee across all ops"
        )

    return SeedReport(
        seed=seed,
        python_random_seeded=True,
        numpy_seeded=True,
        torch_cpu_seeded=True,
        torch_cuda_seeded=cuda_available,
        cuda_available=cuda_available,
        deterministic_algorithms_requested=deterministic_algorithms,
        strict_determinism_claim=claim,
    )
