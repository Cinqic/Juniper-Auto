"""Initialization: seed reproducibility, statistical sanity of sampled
weights, the smaller residual-output std, and norm weights at ones."""

from __future__ import annotations

import torch

from juniper_auto.config import load_architecture_config
from juniper_auto.model import build_model
from juniper_auto.model.attention import GroupedQueryAttention
from juniper_auto.model.block import DenseBlock, MoEBlock
from juniper_auto.model.model import JuniperAutoModel, initialize_weights
from tests.model_fixtures import make_tiny_sparse_config

SPARSE_CFG = load_architecture_config("configs/architecture/ja150m-v0.1.yaml")


def test_same_seed_reproduces_identical_initialization():
    cfg = make_tiny_sparse_config()
    model_a = build_model(cfg, seed=777)
    model_b = build_model(cfg, seed=777)
    for (name_a, p_a), (name_b, p_b) in zip(model_a.named_parameters(), model_b.named_parameters()):
        assert name_a == name_b
        torch.testing.assert_close(p_a, p_b)


def test_different_seed_changes_initialization():
    cfg = make_tiny_sparse_config()
    model_a = build_model(cfg, seed=1)
    model_b = build_model(cfg, seed=2)
    any_different = any(
        not torch.allclose(p_a, p_b) for p_a, p_b in zip(model_a.parameters(), model_b.parameters())
    )
    assert any_different


def test_explicit_seed_does_not_consume_ambient_cpu_rng_state():
    cfg = make_tiny_sparse_config()
    torch.manual_seed(12345)
    state_before = torch.get_rng_state().clone()
    build_model(cfg, seed=777)
    assert torch.equal(torch.get_rng_state(), state_before)


def test_general_projection_std_close_to_base_std_on_official_model():
    model = build_model(SPARSE_CFG, seed=0)
    # A large-N sample of "general" (non-residual-output, non-router,
    # non-embedding) projection weights: attention q_proj across all layers.
    samples = torch.cat([block.attention.q_proj.weight.reshape(-1) for block in model.layers])
    observed_std = samples.std().item()
    assert abs(observed_std - 0.02) < 0.001  # >1M samples -> tight statistical tolerance


def test_residual_output_projection_std_is_the_smaller_scaled_value():
    model = build_model(SPARSE_CFG, seed=0)
    o_proj_samples = torch.cat([block.attention.o_proj.weight.reshape(-1) for block in model.layers])
    observed_std = o_proj_samples.std().item()
    expected_std = 0.02 / (2 * 20) ** 0.5
    assert abs(observed_std - expected_std) < 0.0005
    # And it must be meaningfully smaller than the general std -- the whole
    # point of the scaled-residual-init rule.
    assert observed_std < 0.01


def test_router_std_matches_config():
    model = build_model(SPARSE_CFG, seed=0)
    from juniper_auto.model.block import MoEBlock

    router_samples = torch.cat(
        [block.moe.router.weight.reshape(-1) for block in model.layers if isinstance(block, MoEBlock)]
    )
    observed_std = router_samples.std().item()
    assert abs(observed_std - 0.02) < 0.002


def test_embedding_std_matches_config():
    model = build_model(SPARSE_CFG, seed=0)
    observed_std = model.embedding.weight.reshape(-1).std().item()
    assert abs(observed_std - 0.02) < 0.0001


def test_norm_weights_are_exactly_ones_at_construction():
    model = build_model(SPARSE_CFG, seed=0)
    for block in model.layers:
        assert torch.equal(block.attention_norm.weight, torch.ones_like(block.attention_norm.weight))
        assert torch.equal(block.ffn_norm.weight, torch.ones_like(block.ffn_norm.weight))
        if block.attention.qk_norm:
            assert torch.equal(block.attention.q_norm.weight, torch.ones_like(block.attention.q_norm.weight))
            assert torch.equal(block.attention.k_norm.weight, torch.ones_like(block.attention.k_norm.weight))
    assert torch.equal(model.final_norm.weight, torch.ones_like(model.final_norm.weight))


def test_no_bias_parameters_to_initialize():
    model = build_model(SPARSE_CFG, seed=0)
    assert not any(name.endswith(".bias") for name, _ in model.named_parameters())


def test_initialization_policy_reaches_every_projection_once(monkeypatch):
    cfg = make_tiny_sparse_config()
    model = JuniperAutoModel(cfg)
    calls = {}

    def record(weight, mean, std, generator):
        assert id(weight) not in calls, "a tied or traversed weight was initialized twice"
        calls[id(weight)] = (mean, std)

    monkeypatch.setattr("juniper_auto.model.model._init_normal", record)
    initialize_weights(model, cfg, generator=torch.Generator().manual_seed(1))

    base = (cfg.initialization.mean, cfg.initialization.base_std)
    residual = (cfg.initialization.mean, cfg.initialization.residual_output_projection_std)
    router = (cfg.initialization.mean, cfg.initialization.router_std)
    embedding = (cfg.initialization.mean, cfg.initialization.embedding_std)
    assert calls[id(model.embedding.weight)] == embedding
    assert model.embedding.weight is model.lm_head.weight

    for block in model.layers:
        attn = block.attention
        for projection in (attn.q_proj, attn.k_proj, attn.v_proj):
            assert calls[id(projection.weight)] == base
        assert calls[id(attn.o_proj.weight)] == residual
        if isinstance(block, DenseBlock):
            experts = [block.ffn]
        else:
            assert isinstance(block, MoEBlock)
            assert calls[id(block.moe.router.weight)] == router
            experts = [*block.moe.routed_experts, block.moe.shared_expert]
        for expert in experts:
            assert calls[id(expert.gate_proj.weight)] == base
            assert calls[id(expert.up_proj.weight)] == base
            assert calls[id(expert.down_proj.weight)] == residual
