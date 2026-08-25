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
        "core.moe_layers": [2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19],
        "attention.kind": "causal_gqa",
        "attention.n_query_heads": 8,
        "attention.n_kv_heads": 2,
        "attention.head_dim": 64,
        "attention.qk_norm": True,
        "attention.qk_norm_placement": "before_rope",
        "attention.qk_norm_kind": "per_head_rmsnorm",
        "attention.attention_scale": 0.125,
        "attention.context_length": 4096,
        "attention.future_context_target": 16384,
        "attention.future_context_advertised": False,
        "attention.sliding_window": None,
        "attention.attention_bias": False,
        "attention.causal": True,
        "dense_ffn.kind": "swiglu",
        "dense_ffn.activation": "silu",
        "dense_ffn.dim": 1536,
        "dense_ffn.expansion": 3.0,
        "dense_ffn.bias": False,
        "moe.n_routed_experts": 8,
        "moe.n_shared_experts": 1,
        "moe.top_k": 2,
        "moe.shared_expert_always_active": True,
        "moe.shared_expert_gated": False,
        "moe.expert_ffn_dim": 512,
        "moe.expert_kind": "swiglu",
        "moe.expert_activation": "silu",
        "moe.expert_bias": False,
        "moe.router_bias": False,
        "moe.router_input_dim": 512,
        "moe.router_output_dim": 8,
        "moe.router_logits_dtype": "fp32",
        "moe.router_softmax_dtype": "fp32",
        "moe.routing_kind": "token_choice",
        "moe.dropless": True,
        "moe.token_dropping_allowed": False,
        "moe.renormalize_top_k_weights": True,
        "moe.expert_output_combination": "sum",
        "moe.load_balance_loss_coefficient": 0.01,
        "moe.router_z_loss_coefficient": 0.001,
        "moe.training_router_jitter_policy": "experiment_only",
        "moe.training_router_jitter_magnitude": None,
        "moe.evaluation_router_jitter": False,
        "moe.inference_router_jitter": False,
        "normalization.kind": "rmsnorm",
        "normalization.placement": "pre_norm",
        "normalization.epsilon": 1e-5,
        "normalization.reduction_dtype": "fp32",
        "normalization.attention_norm": True,
        "normalization.ffn_or_moe_norm": True,
        "normalization.final_norm": True,
        "normalization.layernorm_bias": False,
        "position_encoding.kind": "rope",
        "position_encoding.theta": 100000,
        "position_encoding.initial_scaling": 1.0,
        "position_encoding.rotary_fraction": 1.0,
        "position_encoding.rotary_dim": 64,
        "residual.kind": "additive",
        "residual.scale": 1.0,
        "residual.rezero": False,
        "residual.deepnorm": False,
        "residual.learned_gates": False,
        "embeddings.kind": "learned",
        "embeddings.vocab_size": 36864,
        "embeddings.dim": 512,
        "embeddings.tie_lm_head": True,
        "embeddings.output_bias": False,
        "embeddings.embedding_scale": 1.0,
        "embeddings.logit_softcap": None,
        "dropout.embedding": 0.0,
        "dropout.attention": 0.0,
        "dropout.ffn": 0.0,
        "dropout.residual": 0.0,
        "initialization.distribution": "normal",
        "initialization.mean": 0.0,
        "initialization.base_std": 0.02,
        "initialization.router_std": 0.02,
        "initialization.embedding_std": 0.02,
        "initialization.residual_output_projection_std": 0.0031622776601683794,
        "precision.training_mixed_precision": "fp16",
        "precision.training_param_master_dtype": "fp32",
        "precision.inference_default_dtype": "fp16",
    },
    "ja150m-v0.1-dense": {
        "core.d_model": 512,
        "core.n_layers": 20,
        "core.dense_layers": list(range(1, 21)),
        "core.moe_layers": [],
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

    expected_values = dict(FROZEN_CORE_VALUES[arch_id])
    if arch_id == "ja150m-v0.1-dense":
        # The dense control freezes every shared architecture policy to the
        # same independent constants as the sparse reference. Only its layer
        # partition and absent MoE section differ.
        shared_sections = {
            "attention", "dense_ffn", "normalization", "position_encoding",
            "residual", "embeddings", "dropout", "initialization", "precision",
        }
        expected_values.update(
            {
                path: value
                for path, value in FROZEN_CORE_VALUES["ja150m-v0.1"].items()
                if path.split(".", 1)[0] in shared_sections
            }
        )

    for dotted_path, expected in expected_values.items():
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
