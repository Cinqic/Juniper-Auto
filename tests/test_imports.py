"""Import checks -- imports every Phase 0 module from a clean interpreter
state via subprocess, so a broken import doesn't hide behind pytest's own
already-warm import cache."""

import subprocess
import sys

CRITICAL_MODULES = [
    "juniper_auto",
    "juniper_auto.config",
    "juniper_auto.config.schema",
    "juniper_auto.config.loader",
    "juniper_auto.config.frozen",
    "juniper_auto.accounting",
    "juniper_auto.accounting.parameter_count",
    "juniper_auto.foundation",
    "juniper_auto.foundation.probe",
    "juniper_auto.util",
    "juniper_auto.util.seed",
    "juniper_auto.util.logging",
    "juniper_auto.util.environment",
]


def test_all_critical_modules_import_cleanly(repo_root):
    code = "import " + "; import ".join(CRITICAL_MODULES) + "; print('OK')"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"import failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    assert "OK" in result.stdout


def test_torch_importable_without_gpu_requirement(repo_root):
    """Import and use torch on CPU explicitly, proving no hidden GPU
    requirement exists for CI (which runs CPU-only)."""
    code = (
        "import torch;"
        "x = torch.randn(4, 4, device='cpu');"
        "y = (x @ x).sum();"
        "print(float(y))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=60,
        env={"CUDA_VISIBLE_DEVICES": "", **_clean_env()},
    )
    assert result.returncode == 0, f"CPU-only torch failed:\nstdout={result.stdout}\nstderr={result.stderr}"


def _clean_env():
    import os

    return dict(os.environ)
