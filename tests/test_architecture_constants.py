import pytest

from juniper_auto.config import ArchitectureConfig, assert_frozen_v01, load_architecture_config
from juniper_auto.config.frozen import FrozenValueMismatch


def test_sparse_frozen_values_pass(sparse_config_path):
    cfg = load_architecture_config(sparse_config_path)
    assert_frozen_v01(cfg)  # must not raise


def test_dense_frozen_values_pass(dense_config_path):
    cfg = load_architecture_config(dense_config_path)
    assert_frozen_v01(cfg)  # must not raise


def test_sparse_frozen_d_model_drift_detected(sparse_config_path):
    cfg = load_architecture_config(sparse_config_path)
    tampered = cfg.model_copy(update={"core": cfg.core.model_copy(update={"d_model": 513})})
    with pytest.raises(FrozenValueMismatch, match="d_model"):
        assert_frozen_v01(tampered)


def test_sparse_frozen_vocab_size_drift_detected(sparse_config_path):
    cfg = load_architecture_config(sparse_config_path)
    tampered = cfg.model_copy(update={"embeddings": cfg.embeddings.model_copy(update={"vocab_size": 32000})})
    with pytest.raises(FrozenValueMismatch, match="vocab_size"):
        assert_frozen_v01(tampered)


def test_sparse_frozen_head_counts(sparse_config_path):
    cfg = load_architecture_config(sparse_config_path)
    assert cfg.attention.n_query_heads == 8
    assert cfg.attention.n_kv_heads == 2
    assert cfg.attention.head_dim == 64


def test_sparse_frozen_context_length(sparse_config_path):
    cfg = load_architecture_config(sparse_config_path)
    assert cfg.attention.context_length == 4096
    assert cfg.attention.future_context_target == 16384
    assert cfg.attention.future_context_advertised is False


def test_sparse_frozen_moe_shape(sparse_config_path):
    cfg = load_architecture_config(sparse_config_path)
    assert cfg.moe.n_routed_experts == 8
    assert cfg.moe.n_shared_experts == 1
    assert cfg.moe.top_k == 2
    assert cfg.moe.expert_ffn_dim == 512
    assert cfg.moe.dropless is True
    assert cfg.moe.token_dropping_allowed is False


def test_sparse_frozen_normalization_and_rope(sparse_config_path):
    cfg = load_architecture_config(sparse_config_path)
    assert cfg.normalization.kind == "rmsnorm"
    assert cfg.normalization.placement == "pre_norm"
    assert cfg.normalization.epsilon == 1e-5
    assert cfg.position_encoding.theta == 100000
    assert cfg.position_encoding.rotary_dim == 64


def test_sparse_frozen_initialization(sparse_config_path):
    cfg = load_architecture_config(sparse_config_path)
    assert cfg.initialization.base_std == 0.02
    assert cfg.initialization.router_std == 0.02
    assert cfg.initialization.embedding_std == 0.02
    assert cfg.initialization.residual_output_projection_std == pytest.approx(0.02 / (40**0.5))


def test_sparse_frozen_dropout_all_zero(sparse_config_path):
    cfg = load_architecture_config(sparse_config_path)
    assert cfg.dropout.embedding == 0.0
    assert cfg.dropout.attention == 0.0
    assert cfg.dropout.ffn == 0.0
    assert cfg.dropout.residual == 0.0


def test_dense_control_shares_frozen_attention_and_vocab_with_sparse(sparse_config_path, dense_config_path):
    sparse = load_architecture_config(sparse_config_path)
    dense = load_architecture_config(dense_config_path)
    assert sparse.attention.n_query_heads == dense.attention.n_query_heads
    assert sparse.attention.n_kv_heads == dense.attention.n_kv_heads
    assert sparse.attention.head_dim == dense.attention.head_dim
    assert sparse.attention.context_length == dense.attention.context_length
    assert sparse.embeddings.vocab_size == dense.embeddings.vocab_size
    assert sparse.core.d_model == dense.core.d_model
    assert sparse.core.n_layers == dense.core.n_layers
    assert sparse.dense_ffn.dim == dense.dense_ffn.dim
