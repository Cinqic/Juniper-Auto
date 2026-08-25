"""Deterministic YAML loading for architecture configuration."""

from __future__ import annotations

from pathlib import Path

import yaml

from juniper_auto.config.schema import ArchitectureConfig


def load_architecture_config(path: str | Path) -> ArchitectureConfig:
    """Load and schema-validate an architecture config from a YAML file.

    Raises pydantic.ValidationError if the file is malformed, and
    yaml.YAMLError if it is not valid YAML.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")
    return ArchitectureConfig.model_validate(raw)
