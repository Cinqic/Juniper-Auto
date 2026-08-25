"""Programmatic parameter accounting for Juniper Auto architectures.

Every count here is *derived* from an ``ArchitectureConfig`` -- nothing is
copied from documentation. This is what
docs/research/project-governance.md rule 13 ("Deterministic computation over
guessing") and rule 9 ("Total vs. active parameter reporting") require in
code. See docs/research/project-charter.md for what "standard active
parameters" means as a convention (not a FLOPs measurement).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from juniper_auto.config.schema import ArchitectureConfig


@dataclass(frozen=True)
class ParameterBreakdown:
    embeddings: int
    attention: int
    dense_ffns: int
    routed_experts: int
    shared_experts: int
    routers: int
    qk_norms: int
    block_norms: int
    final_norm: int

    @property
    def total(self) -> int:
        return (
            self.embeddings
            + self.attention
            + self.dense_ffns
            + self.routed_experts
            + self.shared_experts
            + self.routers
            + self.qk_norms
            + self.block_norms
            + self.final_norm
        )

    def as_dict(self) -> dict[str, int]:
        d = asdict(self)
        d["total"] = self.total
        return d


def _embedding_params(cfg: ArchitectureConfig) -> int:
    # Tied LM head: the input embedding matrix is reused as the output
    # projection, so it is counted once, not twice.
    return cfg.embeddings.vocab_size * cfg.embeddings.dim


def _attention_params_per_layer(cfg: ArchitectureConfig) -> int:
    a = cfg.attention
    q_dim = a.n_query_heads * a.head_dim
    kv_dim = a.n_kv_heads * a.head_dim
    bias = 0  # attention_bias is false for ja150m-v0.1; kept explicit below.
    if a.attention_bias:
        bias = q_dim + kv_dim + kv_dim + cfg.core.d_model
    q_proj = cfg.core.d_model * q_dim
    k_proj = cfg.core.d_model * kv_dim
    v_proj = cfg.core.d_model * kv_dim
    o_proj = q_dim * cfg.core.d_model
    return q_proj + k_proj + v_proj + o_proj + bias


def _attention_params(cfg: ArchitectureConfig) -> int:
    return _attention_params_per_layer(cfg) * cfg.core.n_layers


def _swiglu_ffn_params(d_model: int, hidden_dim: int, bias: bool) -> int:
    # gate_proj (d_model -> hidden), up_proj (d_model -> hidden),
    # down_proj (hidden -> d_model). No bias for ja150m-v0.1 FFNs.
    params = 3 * d_model * hidden_dim
    if bias:
        params += 2 * hidden_dim + d_model
    return params


def _dense_ffn_params(cfg: ArchitectureConfig) -> int:
    per_ffn = _swiglu_ffn_params(cfg.core.d_model, cfg.dense_ffn.dim, cfg.dense_ffn.bias)
    return per_ffn * len(cfg.core.dense_layers)


def _expert_params_per_expert(cfg: ArchitectureConfig) -> int:
    assert cfg.moe is not None
    # Routed and shared experts share the same SwiGLU FFN shape at
    # expert_ffn_dim, bias-free, per ja150m-v0.1.
    return _swiglu_ffn_params(cfg.core.d_model, cfg.moe.expert_ffn_dim, bias=False)


def _routed_expert_params(cfg: ArchitectureConfig) -> int:
    if cfg.moe is None:
        return 0
    n_moe_layers = len(cfg.core.moe_layers)
    return _expert_params_per_expert(cfg) * cfg.moe.n_routed_experts * n_moe_layers


def _shared_expert_params(cfg: ArchitectureConfig) -> int:
    if cfg.moe is None:
        return 0
    n_moe_layers = len(cfg.core.moe_layers)
    return _expert_params_per_expert(cfg) * cfg.moe.n_shared_experts * n_moe_layers


def _router_params(cfg: ArchitectureConfig) -> int:
    if cfg.moe is None:
        return 0
    n_moe_layers = len(cfg.core.moe_layers)
    per_layer = cfg.moe.router_input_dim * cfg.moe.router_output_dim
    if cfg.moe.router_bias:
        per_layer += cfg.moe.router_output_dim
    return per_layer * n_moe_layers


def _qk_norm_params(cfg: ArchitectureConfig) -> int:
    if not cfg.attention.qk_norm:
        return 0
    # Per-head RMSNorm with a scale vector shared across heads: one
    # head_dim-sized weight for Q, one head_dim-sized weight for K, per layer.
    per_layer = 2 * cfg.attention.head_dim
    return per_layer * cfg.core.n_layers


def _block_norm_params(cfg: ArchitectureConfig) -> int:
    # Pre-Norm RMSNorm before attention and before FFN, per layer.
    per_layer = 2 * cfg.core.d_model
    return per_layer * cfg.core.n_layers


def _final_norm_params(cfg: ArchitectureConfig) -> int:
    return cfg.core.d_model


def total_parameter_breakdown(cfg: ArchitectureConfig) -> ParameterBreakdown:
    """Total parameter count breakdown (every parameter that exists)."""
    return ParameterBreakdown(
        embeddings=_embedding_params(cfg),
        attention=_attention_params(cfg),
        dense_ffns=_dense_ffn_params(cfg),
        routed_experts=_routed_expert_params(cfg),
        shared_experts=_shared_expert_params(cfg),
        routers=_router_params(cfg),
        qk_norms=_qk_norm_params(cfg),
        block_norms=_block_norm_params(cfg),
        final_norm=_final_norm_params(cfg),
    )


def standard_active_parameter_breakdown(cfg: ArchitectureConfig) -> ParameterBreakdown:
    """Standard active parameter breakdown: embeddings, all attention, the
    dense-anchor FFNs, top_k routed experts per MoE layer (not all routed
    experts), the shared expert(s) (always active), routers, and all norms.

    This is a parameter-accounting convention -- see
    docs/research/project-charter.md, "Sparse-vs-dense requirement". It is
    not a measurement of realized FLOPs.
    """
    if cfg.moe is None:
        # Dense models have no routing distinction between "total" and
        # "active" -- every parameter that exists is used on every token.
        return total_parameter_breakdown(cfg)

    n_moe_layers = len(cfg.core.moe_layers)
    active_routed_experts = _expert_params_per_expert(cfg) * cfg.moe.top_k * n_moe_layers

    return ParameterBreakdown(
        embeddings=_embedding_params(cfg),
        attention=_attention_params(cfg),
        dense_ffns=_dense_ffn_params(cfg),
        routed_experts=active_routed_experts,
        shared_experts=_shared_expert_params(cfg),
        routers=_router_params(cfg),
        qk_norms=_qk_norm_params(cfg),
        block_norms=_block_norm_params(cfg),
        final_norm=_final_norm_params(cfg),
    )
