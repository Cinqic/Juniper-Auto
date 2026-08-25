"""Frozen v0.1 value assertions.

`juniper_auto.config.schema` validates general structure. This module
asserts the exact frozen numeric values for architecture ids
`ja150m-v0.1` and `ja150m-v0.1-dense`, so that any accidental drift in the
committed YAML configs fails loudly instead of silently changing the
research target. See docs/research/project-charter.md and
docs/research/project-governance.md rule 4 (frozen-artifact versioning).
"""

from __future__ import annotations

from juniper_auto.accounting.parameter_count import (
    standard_active_parameter_breakdown,
    total_parameter_breakdown,
)
from juniper_auto.config.schema import ArchitectureConfig

FROZEN_TOTAL_PARAMETERS = {
    "ja150m-v0.1": 150_031_360,
    "ja150m-v0.1-dense": 79_191_040,
}

FROZEN_STANDARD_ACTIVE_PARAMETERS = {
    "ja150m-v0.1": 79_252_480,
    # The dense control has no total/active distinction (see
    # standard_active_parameter_breakdown docstring).
    "ja150m-v0.1-dense": 79_191_040,
}

FROZEN_CORE_VALUES: dict[str, dict] = {
    "ja150m-v0.1": {
        "core.d_model": 512,
        "core.n_layers": 20,
        "core.dense_layers": [1, 5, 10, 15, 20],
        "attention.n_query_heads": 8,
        "attention.n_kv_heads": 2,
        "attention.head_dim": 64,
        "attention.context_length": 4096,
        "dense_ffn.dim": 1536,
        "moe.n_routed_experts": 8,
        "moe.n_shared_experts": 1,
        "moe.top_k": 2,
        "moe.expert_ffn_dim": 512,
        "embeddings.vocab_size": 36864,
    },
    "ja150m-v0.1-dense": {
        "core.d_model": 512,
        "core.n_layers": 20,
        "attention.n_query_heads": 8,
        "attention.n_kv_heads": 2,
        "attention.head_dim": 64,
        "attention.context_length": 4096,
        "dense_ffn.dim": 1536,
        "embeddings.vocab_size": 36864,
    },
}


class FrozenValueMismatch(ValueError):
    pass


def _get_path(cfg: ArchitectureConfig, dotted: str):
    obj = cfg
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def assert_frozen_v01(cfg: ArchitectureConfig) -> None:
    """Raise FrozenValueMismatch if `cfg` deviates from the frozen v0.1 spec
    for its architecture_id. No-op for unrecognized architecture ids."""
    arch_id = cfg.architecture_id
    if arch_id not in FROZEN_CORE_VALUES:
        return

    for dotted_path, expected in FROZEN_CORE_VALUES[arch_id].items():
        actual = _get_path(cfg, dotted_path)
        if actual != expected:
            raise FrozenValueMismatch(
                f"{arch_id}: {dotted_path} is frozen at {expected!r}, config has {actual!r}"
            )

    total = total_parameter_breakdown(cfg).total
    expected_total = FROZEN_TOTAL_PARAMETERS[arch_id]
    if total != expected_total:
        raise FrozenValueMismatch(
            f"{arch_id}: total parameters must be {expected_total}, computed {total}"
        )

    active = standard_active_parameter_breakdown(cfg).total
    expected_active = FROZEN_STANDARD_ACTIVE_PARAMETERS[arch_id]
    if active != expected_active:
        raise FrozenValueMismatch(
            f"{arch_id}: standard active parameters must be {expected_active}, computed {active}"
        )
