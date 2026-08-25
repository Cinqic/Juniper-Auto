"""Full-model backward pass: gradients exist and are finite where expected,
for both sparse and dense tiny configs."""

from __future__ import annotations

import torch

from juniper_auto.model import build_model
from juniper_auto.model.block import MoEBlock
from tests.model_fixtures import make_tiny_dense_config, make_tiny_sparse_config


def test_dense_backward_all_params_receive_finite_gradients():
    cfg = make_tiny_dense_config()
    model = build_model(cfg, seed=0)
    input_ids = torch.randint(0, cfg.embeddings.vocab_size, (2, 6))
    labels = input_ids.clone()
    out = model(input_ids, labels=labels)
    out.loss.backward()

    for name, p in model.named_parameters():
        assert p.grad is not None, f"missing gradient: {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite gradient: {name}"


def test_sparse_backward_router_and_active_experts_receive_finite_gradients():
    # Use a batch large enough (relative to n_routed_experts) that every
    # expert is very likely selected at least once, so this test can
    # legitimately assert full coverage rather than "some subset".
    cfg = make_tiny_sparse_config(n_routed_experts=4, top_k=2)
    model = build_model(cfg, seed=0)
    torch.manual_seed(0)
    input_ids = torch.randint(0, cfg.embeddings.vocab_size, (4, 32))
    labels = input_ids.clone()
    out = model(input_ids, labels=labels, return_diagnostics=True)
    out.loss.backward()

    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite gradient: {name}"

    for block, diag in zip(model.layers, out.diagnostics):
        if not isinstance(block, MoEBlock):
            continue
        assert block.moe.router.weight.grad is not None
        for p in block.moe.shared_expert.parameters():
            assert p.grad is not None
        selected = set(diag.topk_idx.reshape(-1).tolist())
        assert len(selected) == cfg.moe.n_routed_experts, (
            "test batch too small to exercise every routed expert; increase batch/seq"
        )
        for expert in block.moe.routed_experts:
            assert all(p.grad is not None for p in expert.parameters())


def test_embedding_and_tied_lm_head_gradient_is_a_single_accumulated_tensor():
    cfg = make_tiny_sparse_config()
    model = build_model(cfg, seed=0)
    input_ids = torch.randint(0, cfg.embeddings.vocab_size, (1, 5))
    labels = input_ids.clone()
    out = model(input_ids, labels=labels)
    out.loss.backward()
    # Since embedding.weight and lm_head.weight are the same Parameter
    # object, there is exactly one .grad tensor, accumulated from both the
    # embedding-lookup backward and the LM-head-matmul backward.
    assert model.embedding.weight.grad is model.lm_head.weight.grad
    assert torch.isfinite(model.embedding.weight.grad).all()
