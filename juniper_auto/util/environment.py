"""Environment identity capture -- used by logging, the foundation probe,
and `scripts/validate_repo.py`'s environment-sanity gate.

This reports the *actual* running environment; it does not assert that the
environment matches any particular expectation (that is the caller's job).
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EnvironmentIdentity:
    python_version: str
    platform: str
    torch_version: str
    cuda_available: bool
    cuda_device_name: str | None
    cuda_driver_version: str | None

    def as_dict(self) -> dict:
        return {
            "python_version": self.python_version,
            "platform": self.platform,
            "torch_version": self.torch_version,
            "cuda_available": self.cuda_available,
            "cuda_device_name": self.cuda_device_name,
            "cuda_driver_version": self.cuda_driver_version,
        }

    def short_id(self) -> str:
        gpu = self.cuda_device_name or "cpu-only"
        return f"py{self.python_version}-torch{self.torch_version}-{gpu}"


def describe_environment() -> EnvironmentIdentity:
    cuda_available = torch.cuda.is_available()
    device_name = None
    driver_version = None
    if cuda_available:
        try:
            device_name = torch.cuda.get_device_name(0)
        except Exception:
            device_name = "unknown-cuda-device"
        driver_version = getattr(torch.version, "cuda", None)

    return EnvironmentIdentity(
        python_version=platform.python_version(),
        platform=f"{platform.system()}-{platform.release()}",
        torch_version=torch.__version__,
        cuda_available=cuda_available,
        cuda_device_name=device_name,
        cuda_driver_version=driver_version,
    )
