"""Structural verification of the actual official ja150m-v0.1 /
ja150m-v0.1-dense modules -- not the config, the instantiated PyTorch
model. A config can be correct while the code that builds modules from it
is wrong, so every check here inspects `model.modules()`/`model.parameters()`
directly. This is Phase 1's Method B parameter verification (Method A is
`juniper_auto.accounting`, already covered by Phase 0's tests)."""

from __future__ import annotations

import torch

from juniper_auto.accounting import standard_active_parameter_breakdown, total_parameter_breakdown
from juniper_auto.config import load_architecture_config
from juniper_auto.config.frozen import FROZEN_STANDARD_ACTIVE_PARAMETERS, FROZEN_TOTAL_PARAMETERS
from juniper_auto.model import build_model
from juniper_auto.model.inspection import (
    bias_audit,
    dropout_audit,
    layer_placement_report,
    pytorch_parameter_breakdown,
    qk_norm_parameter_count,
    total_parameters,
    verify_weight_tying,
)

SPARSE_CFG = load_architecture_config("configs/architecture/ja150m-v0.1.yaml")
DENSE_CFG = load_architecture_config("configs/architecture/ja150m-v0.1-dense.yaml")


def test_sparse_total_parameters_exact_match_pytorch_and_config():
    model = build_model(SPARSE_CFG, seed=0)
    method_b_total = total_parameters(model)
    method_a_total = total_parameter_breakdown(SPARSE_CFG).total
    assert method_b_total == 150_031_360 == FROZEN_TOTAL_PARAMETERS["ja150m-v0.1"]
    assert method_a_total == method_b_total


def test_sparse_standard_active_parameters_config_derived():
    method_a_active = standard_active_parameter_breakdown(SPARSE_CFG).total
    assert method_a_active == 79_252_480 == FROZEN_STANDARD_ACTIVE_PARAMETERS["ja150m-v0.1"]


def test_dense_total_parameters_exact_match_pytorch_and_config():
    model = build_model(DENSE_CFG, seed=0)
    method_b_total = total_parameters(model)
    method_a_total = total_parameter_breakdown(DENSE_CFG).total
    assert method_b_total == 79_191_040 == FROZEN_TOTAL_PARAMETERS["ja150m-v0.1-dense"]
    assert method_a_total == method_b_total


def test_sparse_pytorch_breakdown_matches_config_breakdown_field_by_field():
    model = build_model(SPARSE_CFG, seed=0)
    method_b = pytorch_parameter_breakdown(model)
    method_a = total_parameter_breakdown(SPARSE_CFG).as_dict()
    for key in method_a:
        assert method_b[key] == method_a[key], f"mismatch in {key}: pytorch={method_b[key]} config={method_a[key]}"


def test_dense_pytorch_breakdown_matches_config_breakdown_field_by_field():
    model = build_model(DENSE_CFG, seed=0)
    method_b = pytorch_parameter_breakdown(model)
    method_a = total_parameter_breakdown(DENSE_CFG).as_dict()
    for key in method_a:
        assert method_b[key] == method_a[key]


def test_qk_norm_parameter_count_is_exactly_2560():
    model = build_model(SPARSE_CFG, seed=0)
    assert qk_norm_parameter_count(model) == 2560


def test_sparse_layer_placement_matches_frozen_dense_moe_positions():
    model = build_model(SPARSE_CFG, seed=0)
    report = layer_placement_report(model)
    dense_positions = {entry["layer"] for entry in report if entry["kind"] == "dense"}
    moe_positions = {entry["layer"] for entry in report if entry["kind"] == "moe"}
    assert dense_positions == {1, 5, 10, 15, 20}
    assert moe_positions == set(range(1, 21)) - {1, 5, 10, 15, 20}
    for entry in report:
        if entry["kind"] == "moe":
            assert entry["n_routed_experts"] == 8
            assert entry["n_shared_experts"] == 1
            assert entry["top_k"] == 2


def test_dense_control_layer_placement_is_all_dense():
    model = build_model(DENSE_CFG, seed=0)
    report = layer_placement_report(model)
    assert all(entry["kind"] == "dense" for entry in report)
    assert len(report) == 20


def test_weight_tying_is_real_object_identity():
    model = build_model(SPARSE_CFG, seed=0)
    assert verify_weight_tying(model)
    assert model.embedding.weight.data_ptr() == model.lm_head.weight.data_ptr()


def test_no_bias_parameters_anywhere_in_sparse_model():
    model = build_model(SPARSE_CFG, seed=0)
    assert bias_audit(model) == []


def test_no_bias_parameters_anywhere_in_dense_model():
    model = build_model(DENSE_CFG, seed=0)
    assert bias_audit(model) == []


def test_no_dropout_modules_in_either_model():
    assert dropout_audit(build_model(SPARSE_CFG, seed=0)) == []
    assert dropout_audit(build_model(DENSE_CFG, seed=0)) == []


def test_sparse_forward_backward_smoke_on_full_official_architecture():
    model = build_model(SPARSE_CFG, seed=0)
    torch.manual_seed(0)
    input_ids = torch.randint(0, SPARSE_CFG.embeddings.vocab_size, (1, 8))
    labels = input_ids.clone()
    out = model(input_ids, labels=labels, return_diagnostics=True)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert any(p.grad is not None for p in model.parameters())

    moe_diags = [d for d in out.diagnostics if d is not None]
    assert len(moe_diags) == 15  # exactly the 15 MoE layers
    for diag in moe_diags:
        e0 = diag.topk_idx[:, 0]
        e1 = diag.topk_idx[:, 1]
        assert torch.all(e0 != e1)
        torch.testing.assert_close(diag.topk_weights.sum(dim=-1), torch.ones(8), atol=1e-5, rtol=1e-5)


def test_dense_forward_backward_smoke_on_full_official_architecture():
    model = build_model(DENSE_CFG, seed=0)
    torch.manual_seed(0)
    input_ids = torch.randint(0, DENSE_CFG.embeddings.vocab_size, (1, 8))
    labels = input_ids.clone()
    out = model(input_ids, labels=labels)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert any(p.grad is not None for p in model.parameters())
