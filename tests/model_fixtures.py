"""Small, non-frozen architecture configs for exercising model mechanics
without instantiating the full 150M/79M parameter models. Not a test file
itself (no `test_` functions) -- imported by the model/training test
files. Architecture ids here (`test-tiny-*`) are deliberately outside
`KNOWN_ARCHITECTURE_KINDS`, so `assert_frozen_v01` is a no-op for them and
they may vary freely across tests.
"""

from __future__ import annotations

from juniper_auto.config.schema import (
    ArchitectureConfig,
    AttentionConfig,
    CoreConfig,
    DenseFFNConfig,
    DropoutConfig,
    EmbeddingsConfig,
    InitializationConfig,
    MoEConfig,
    NormalizationConfig,
    PositionEncodingConfig,
    PrecisionConfig,
    ResidualConfig,
)


def make_tiny_config(
    *,
    architecture_id: str = "test-tiny-sparse",
    kind: str = "sparse",
    d_model: int = 32,
    n_layers: int = 4,
    dense_layers: list[int] | None = None,
    moe_layers: list[int] | None = None,
    n_query_heads: int = 4,
    n_kv_heads: int = 2,
    head_dim: int = 8,
    attention_scale: float | None = None,
    rotary_dim: int | None = None,
    theta: float = 100000.0,
    vocab_size: int = 97,
    context_length: int = 64,
    n_routed_experts: int = 4,
    n_shared_experts: int = 1,
    top_k: int = 2,
    expert_ffn_dim: int = 32,
    attention_bias: bool = False,
    ffn_bias: bool = False,
    expert_bias: bool = False,
    router_bias: bool = False,
    qk_norm: bool = True,
    epsilon: float = 1e-5,
) -> ArchitectureConfig:
    if dense_layers is None and moe_layers is None:
        if kind == "dense":
            dense_layers = list(range(1, n_layers + 1))
            moe_layers = []
        else:
            dense_layers = [1, n_layers]
            moe_layers = [i for i in range(1, n_layers + 1) if i not in dense_layers]
    assert dense_layers is not None and moe_layers is not None

    moe = None
    if kind == "sparse":
        moe = MoEConfig(
            n_routed_experts=n_routed_experts,
            n_shared_experts=n_shared_experts,
            top_k=top_k,
            shared_expert_always_active=True,
            shared_expert_gated=False,
            expert_ffn_dim=expert_ffn_dim,
            expert_kind="swiglu",
            expert_activation="silu",
            expert_bias=expert_bias,
            router_bias=router_bias,
            router_input_dim=d_model,
            router_output_dim=n_routed_experts,
            router_logits_dtype="fp32",
            router_softmax_dtype="fp32",
            routing_kind="token_choice",
            dropless=True,
            token_dropping_allowed=False,
            renormalize_top_k_weights=True,
            expert_output_combination="sum",
            load_balance_loss_coefficient=0.01,
            router_z_loss_coefficient=0.001,
            training_router_jitter_policy="experiment_only",
            training_router_jitter_magnitude=None,
            evaluation_router_jitter=False,
            inference_router_jitter=False,
        )

    resolved_rotary_dim = rotary_dim or head_dim
    resolved_scale = attention_scale if attention_scale is not None else head_dim**-0.5

    return ArchitectureConfig(
        architecture_id=architecture_id,
        kind=kind,
        core=CoreConfig(d_model=d_model, n_layers=n_layers, dense_layers=dense_layers, moe_layers=moe_layers),
        attention=AttentionConfig(
            kind="causal_gqa",
            n_query_heads=n_query_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            qk_norm=qk_norm,
            qk_norm_placement="before_rope",
            qk_norm_kind="per_head_rmsnorm",
            attention_scale=resolved_scale,
            context_length=context_length,
            future_context_target=context_length,
            future_context_advertised=False,
            sliding_window=None,
            attention_bias=attention_bias,
            causal=True,
        ),
        dense_ffn=DenseFFNConfig(kind="swiglu", activation="silu", dim=d_model * 2, expansion=2.0, bias=ffn_bias),
        moe=moe,
        normalization=NormalizationConfig(
            kind="rmsnorm",
            placement="pre_norm",
            epsilon=epsilon,
            reduction_dtype="fp32",
            attention_norm=True,
            ffn_or_moe_norm=True,
            final_norm=True,
            layernorm_bias=False,
        ),
        position_encoding=PositionEncodingConfig(
            kind="rope",
            theta=theta,
            initial_scaling=1.0,
            rotary_fraction=resolved_rotary_dim / head_dim,
            rotary_dim=resolved_rotary_dim,
        ),
        residual=ResidualConfig(kind="additive", scale=1.0, rezero=False, deepnorm=False, learned_gates=False),
        embeddings=EmbeddingsConfig(
            kind="learned",
            vocab_size=vocab_size,
            dim=d_model,
            tie_lm_head=True,
            output_bias=False,
            embedding_scale=1.0,
            logit_softcap=None,
        ),
        dropout=DropoutConfig(embedding=0.0, attention=0.0, ffn=0.0, residual=0.0),
        initialization=InitializationConfig(
            distribution="normal",
            mean=0.0,
            base_std=0.02,
            router_std=0.02,
            embedding_std=0.02,
            residual_output_projection_std=0.02 / (2 * n_layers) ** 0.5,
        ),
        precision=PrecisionConfig(
            training_mixed_precision="fp16", training_param_master_dtype="fp32", inference_default_dtype="fp16"
        ),
    )


def make_tiny_sparse_config(**overrides) -> ArchitectureConfig:
    overrides.setdefault("architecture_id", "test-tiny-sparse")
    overrides.setdefault("kind", "sparse")
    return make_tiny_config(**overrides)


def make_tiny_dense_config(**overrides) -> ArchitectureConfig:
    overrides.setdefault("architecture_id", "test-tiny-dense")
    overrides.setdefault("kind", "dense")
    return make_tiny_config(**overrides)
