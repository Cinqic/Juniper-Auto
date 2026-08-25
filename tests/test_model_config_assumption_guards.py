"""Every architecture-config field the model code assumes is fixed (rather
than genuinely branching on) must be validated at construction time and
rejected loudly if a config asks for a variant that isn't implemented --
found during Phase 1 self-review Pass B (frozen architecture audit) as a
set of silently-unvalidated assumptions. These tests are the regression
coverage for that fix: each pins one config field to an unimplemented
value and asserts construction raises, rather than silently mishandling it."""

from __future__ import annotations

import pytest

from juniper_auto.model.attention import GroupedQueryAttention
from juniper_auto.model.block import DenseBlock
from juniper_auto.model.model import JuniperAutoModel
from juniper_auto.model.moe import MoELayer
from tests.model_fixtures import make_tiny_sparse_config


def test_residual_rezero_is_rejected():
    cfg = make_tiny_sparse_config()
    broken = cfg.model_copy(update={"residual": cfg.residual.model_copy(update={"rezero": True})})
    with pytest.raises(ValueError, match="ReZero"):
        DenseBlock(broken)


def test_residual_deepnorm_is_rejected():
    cfg = make_tiny_sparse_config()
    broken = cfg.model_copy(update={"residual": cfg.residual.model_copy(update={"deepnorm": True})})
    with pytest.raises(ValueError, match="ReZero"):
        DenseBlock(broken)


def test_residual_learned_gates_is_rejected():
    cfg = make_tiny_sparse_config()
    broken = cfg.model_copy(update={"residual": cfg.residual.model_copy(update={"learned_gates": True})})
    with pytest.raises(ValueError, match="ReZero"):
        DenseBlock(broken)


def test_residual_scale_is_actually_applied():
    cfg = make_tiny_sparse_config()
    scaled = cfg.model_copy(update={"residual": cfg.residual.model_copy(update={"scale": 0.5})})
    block = DenseBlock(scaled)
    assert block.residual_scale == 0.5


def test_normalization_post_norm_is_rejected():
    cfg = make_tiny_sparse_config()
    broken = cfg.model_copy(update={"normalization": cfg.normalization.model_copy(update={"placement": "post_norm"})})
    with pytest.raises(ValueError, match="placement"):
        DenseBlock(broken)


def test_normalization_attention_norm_disabled_is_rejected():
    cfg = make_tiny_sparse_config()
    broken = cfg.model_copy(
        update={"normalization": cfg.normalization.model_copy(update={"attention_norm": False})}
    )
    with pytest.raises(ValueError, match="attention_norm"):
        DenseBlock(broken)


def test_normalization_final_norm_disabled_is_rejected():
    cfg = make_tiny_sparse_config()
    broken = cfg.model_copy(update={"normalization": cfg.normalization.model_copy(update={"final_norm": False})})
    with pytest.raises(ValueError, match="final_norm"):
        JuniperAutoModel(broken)


def test_attention_non_causal_is_rejected():
    cfg = make_tiny_sparse_config()
    broken = cfg.model_copy(update={"attention": cfg.attention.model_copy(update={"causal": False})})
    with pytest.raises(ValueError, match="causal"):
        GroupedQueryAttention(broken)


def test_attention_sliding_window_is_rejected():
    cfg = make_tiny_sparse_config()
    broken = cfg.model_copy(update={"attention": cfg.attention.model_copy(update={"sliding_window": 128})})
    with pytest.raises(ValueError, match="sliding_window"):
        GroupedQueryAttention(broken)


def test_qk_norm_after_rope_placement_is_rejected():
    cfg = make_tiny_sparse_config()
    broken = cfg.model_copy(
        update={"attention": cfg.attention.model_copy(update={"qk_norm_placement": "after_rope"})}
    )
    with pytest.raises(ValueError, match="QK-Norm"):
        GroupedQueryAttention(broken)


def test_moe_expert_choice_routing_is_rejected():
    cfg = make_tiny_sparse_config()
    broken = cfg.model_copy(update={"moe": cfg.moe.model_copy(update={"routing_kind": "expert_choice"})})
    with pytest.raises(ValueError, match="routing_kind"):
        MoELayer(broken)


def test_moe_non_fp32_router_logits_dtype_is_rejected():
    cfg = make_tiny_sparse_config()
    broken = cfg.model_copy(update={"moe": cfg.moe.model_copy(update={"router_logits_dtype": "fp16"})})
    with pytest.raises(ValueError, match="router_logits_dtype"):
        MoELayer(broken)


def test_moe_non_fp32_router_softmax_dtype_is_rejected():
    cfg = make_tiny_sparse_config()
    broken = cfg.model_copy(update={"moe": cfg.moe.model_copy(update={"router_softmax_dtype": "bf16"})})
    with pytest.raises(ValueError, match="router_softmax_dtype"):
        MoELayer(broken)


def test_nonzero_dropout_is_rejected():
    cfg = make_tiny_sparse_config()
    broken = cfg.model_copy(update={"dropout": cfg.dropout.model_copy(update={"attention": 0.1})})
    with pytest.raises(ValueError, match="dropout"):
        JuniperAutoModel(broken)
