"""Pydantic schema for Juniper Auto architecture configuration.

This schema validates structure (types, ranges, cross-field consistency) for
any architecture configuration -- sparse or dense, this version or a future
one. It deliberately does NOT hard-code the ja150m-v0.1 frozen numeric
values; that is the job of ``juniper_auto.config.frozen``, which checks a
loaded, already-schema-valid config against the exact frozen v0.1 spec.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Known architecture ids and the `kind` each one is required to have. An
# unrecognized architecture_id is allowed to pass schema validation (so this
# schema doesn't need editing for every future architecture), but a *known*
# id must match its expected kind.
KNOWN_ARCHITECTURE_KINDS: dict[str, str] = {
    "ja150m-v0.1": "sparse",
    "ja150m-v0.1-dense": "dense",
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CoreConfig(StrictModel):
    d_model: int = Field(gt=0)
    n_layers: int = Field(gt=0)
    dense_layers: list[int]
    moe_layers: list[int]

    @model_validator(mode="after")
    def _validate_layer_partition(self) -> "CoreConfig":
        all_layers = self.dense_layers + self.moe_layers
        if len(set(all_layers)) != len(all_layers):
            dupes = sorted({x for x in all_layers if all_layers.count(x) > 1})
            raise ValueError(
                f"duplicate/overlapping layer numbers across dense_layers and moe_layers: {dupes}"
            )

        dense_set = set(self.dense_layers)
        moe_set = set(self.moe_layers)
        expected = set(range(1, self.n_layers + 1))
        actual = dense_set | moe_set
        missing = expected - actual
        extra = actual - expected
        if missing:
            raise ValueError(f"layer partition is missing layers: {sorted(missing)}")
        if extra:
            raise ValueError(f"layer partition references out-of-range layers: {sorted(extra)}")
        return self


class AttentionConfig(StrictModel):
    kind: Literal["causal_gqa"]
    n_query_heads: int = Field(gt=0)
    n_kv_heads: int = Field(gt=0)
    head_dim: int = Field(gt=0)
    qk_norm: bool
    qk_norm_placement: Literal["before_rope", "after_rope"]
    qk_norm_kind: str
    attention_scale: float = Field(gt=0)
    context_length: int = Field(gt=0)
    future_context_target: int = Field(gt=0)
    future_context_advertised: bool
    sliding_window: int | None = None
    attention_bias: bool
    causal: bool

    @model_validator(mode="after")
    def _validate_head_config(self) -> "AttentionConfig":
        if self.n_query_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_query_heads ({self.n_query_heads}) must be divisible by "
                f"n_kv_heads ({self.n_kv_heads}) for grouped-query attention"
            )
        if self.future_context_target < self.context_length:
            raise ValueError("future_context_target must be >= context_length")
        return self


class DenseFFNConfig(StrictModel):
    kind: Literal["swiglu"]
    activation: Literal["silu"]
    dim: int = Field(gt=0)
    expansion: float = Field(gt=0)
    bias: bool


class MoEConfig(StrictModel):
    n_routed_experts: int = Field(gt=0)
    n_shared_experts: int = Field(ge=0)
    top_k: int = Field(gt=0)
    shared_expert_always_active: bool
    shared_expert_gated: bool
    expert_ffn_dim: int = Field(gt=0)
    expert_kind: Literal["swiglu"]
    expert_activation: Literal["silu"]
    expert_bias: bool
    router_bias: bool
    router_input_dim: int = Field(gt=0)
    router_output_dim: int = Field(gt=0)
    router_logits_dtype: Literal["fp32", "fp16", "bf16"]
    router_softmax_dtype: Literal["fp32", "fp16", "bf16"]
    routing_kind: Literal["token_choice", "expert_choice"]
    dropless: bool
    token_dropping_allowed: bool
    renormalize_top_k_weights: bool
    expert_output_combination: Literal["sum"]
    load_balance_loss_coefficient: float = Field(ge=0)
    router_z_loss_coefficient: float = Field(ge=0)
    training_router_jitter_policy: Literal["experiment_only"]
    training_router_jitter_magnitude: float | None = Field(default=None, gt=0)
    evaluation_router_jitter: bool
    inference_router_jitter: bool

    @model_validator(mode="after")
    def _validate_moe(self) -> "MoEConfig":
        if self.top_k > self.n_routed_experts:
            raise ValueError(
                f"top_k ({self.top_k}) cannot exceed n_routed_experts ({self.n_routed_experts})"
            )
        if self.router_output_dim != self.n_routed_experts:
            raise ValueError(
                "router_output_dim must equal n_routed_experts "
                f"({self.router_output_dim} != {self.n_routed_experts})"
            )
        return self


class NormalizationConfig(StrictModel):
    kind: Literal["rmsnorm"]
    placement: Literal["pre_norm", "post_norm"]
    epsilon: float = Field(gt=0)
    reduction_dtype: Literal["fp32", "fp16", "bf16"]
    attention_norm: bool
    ffn_or_moe_norm: bool
    final_norm: bool
    layernorm_bias: bool


class PositionEncodingConfig(StrictModel):
    kind: Literal["rope"]
    theta: float = Field(gt=0)
    initial_scaling: float = Field(gt=0)
    rotary_fraction: float = Field(gt=0, le=1)
    rotary_dim: int = Field(gt=0)


class ResidualConfig(StrictModel):
    kind: Literal["additive"]
    scale: float
    rezero: bool
    deepnorm: bool
    learned_gates: bool


class EmbeddingsConfig(StrictModel):
    kind: Literal["learned"]
    vocab_size: int = Field(gt=0)
    dim: int = Field(gt=0)
    tie_lm_head: bool
    output_bias: bool
    embedding_scale: float = Field(gt=0)
    logit_softcap: float | None = None


class DropoutConfig(StrictModel):
    embedding: float = Field(ge=0, le=1)
    attention: float = Field(ge=0, le=1)
    ffn: float = Field(ge=0, le=1)
    residual: float = Field(ge=0, le=1)


class InitializationConfig(StrictModel):
    distribution: Literal["normal"]
    mean: float
    base_std: float = Field(gt=0)
    router_std: float = Field(gt=0)
    embedding_std: float = Field(gt=0)
    residual_output_projection_std: float = Field(gt=0)


class PrecisionConfig(StrictModel):
    training_mixed_precision: Literal["fp16", "bf16", "fp32"]
    training_param_master_dtype: Literal["fp32", "fp16", "bf16"]
    inference_default_dtype: Literal["fp16", "bf16", "fp32"]


class ArchitectureConfig(StrictModel):
    architecture_id: str = Field(pattern=r"^[a-z][a-z0-9.-]+$")
    kind: Literal["sparse", "dense"]
    core: CoreConfig
    attention: AttentionConfig
    dense_ffn: DenseFFNConfig
    moe: MoEConfig | None
    normalization: NormalizationConfig
    position_encoding: PositionEncodingConfig
    residual: ResidualConfig
    embeddings: EmbeddingsConfig
    dropout: DropoutConfig
    initialization: InitializationConfig
    precision: PrecisionConfig

    @model_validator(mode="after")
    def _validate_architecture(self) -> "ArchitectureConfig":
        expected_kind = KNOWN_ARCHITECTURE_KINDS.get(self.architecture_id)
        if expected_kind is not None and expected_kind != self.kind:
            raise ValueError(
                f"invalid frozen architecture id: '{self.architecture_id}' must have "
                f"kind '{expected_kind}', got '{self.kind}'"
            )

        if self.kind == "sparse" and self.moe is None:
            raise ValueError("kind='sparse' requires a non-null 'moe' section")
        if self.kind == "dense" and self.moe is not None:
            raise ValueError("kind='dense' requires 'moe' to be null (no MoE layers)")
        if self.kind == "sparse" and not self.core.moe_layers:
            raise ValueError("kind='sparse' requires at least one moe layer")
        if self.kind == "dense" and self.core.moe_layers:
            raise ValueError("kind='dense' must not declare any moe_layers")

        if self.attention.n_query_heads * self.attention.head_dim != self.core.d_model:
            raise ValueError(
                "n_query_heads * head_dim must equal d_model "
                f"({self.attention.n_query_heads} * {self.attention.head_dim} != {self.core.d_model})"
            )
        if self.embeddings.dim != self.core.d_model:
            raise ValueError("embedding dimension must equal d_model")
        if self.position_encoding.rotary_dim > self.attention.head_dim:
            raise ValueError("rotary_dim cannot exceed attention head_dim")
        if self.kind == "sparse" and self.moe is not None:
            if self.moe.router_input_dim != self.core.d_model:
                raise ValueError("router_input_dim must equal d_model")
        return self
