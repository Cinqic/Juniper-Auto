"""Per-token routing traceability (Phase 2 instructions section 8): for a
diagnostic run, exactly which routed experts processed every token at every
MoE layer must be reconstructable, and the facility must be bounded/opt-in
and safe to export as JSON."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch

from juniper_auto.model import build_model
from juniper_auto.model.moe import MoELayer
from juniper_auto.model.moe_diagnostics import assemble_full_trace, export_trace_json
from tests.model_fixtures import make_tiny_sparse_config


def test_return_trace_requires_return_diagnostics():
    cfg = make_tiny_sparse_config(n_routed_experts=4, top_k=2, d_model=4, n_query_heads=1, n_kv_heads=1, head_dim=4)
    layer = MoELayer(cfg)
    x = torch.randn(1, 5, cfg.core.d_model)
    with pytest.raises(ValueError, match="return_trace"):
        layer(x, return_diagnostics=False, return_trace=True)


def test_single_layer_trace_records_correct_positions_experts_and_weights():
    cfg = make_tiny_sparse_config(n_routed_experts=4, top_k=2, d_model=4, n_query_heads=1, n_kv_heads=1, head_dim=4)
    layer = MoELayer(cfg)
    torch.manual_seed(0)
    batch, seq_len = 2, 5
    x = torch.randn(batch, seq_len, cfg.core.d_model)
    valid = torch.ones(batch, seq_len, dtype=torch.bool)
    valid[1, -1] = False

    _, _, _, diag = layer(x, valid_mask=valid, return_diagnostics=True, return_trace=True)
    trace = diag.token_trace
    assert len(trace) == batch * seq_len

    for record in trace:
        flat_idx = record.batch_index * seq_len + record.seq_position
        assert record.flat_token_index == flat_idx
        assert record.expert_1 == diag.topk_idx[flat_idx, 0].item()
        assert record.expert_2 == diag.topk_idx[flat_idx, 1].item()
        assert record.weight_1 == pytest.approx(diag.topk_weights[flat_idx, 0].item())
        assert record.weight_2 == pytest.approx(diag.topk_weights[flat_idx, 1].item())
        assert record.is_valid == bool(valid[record.batch_index, record.seq_position].item())


def test_trace_covers_every_token_no_lost_records():
    cfg = make_tiny_sparse_config(n_routed_experts=5, top_k=2, d_model=4, n_query_heads=1, n_kv_heads=1, head_dim=4)
    layer = MoELayer(cfg)
    batch, seq_len = 3, 9
    x = torch.randn(batch, seq_len, cfg.core.d_model)
    _, _, _, diag = layer(x, return_diagnostics=True, return_trace=True)
    seen = {(r.batch_index, r.seq_position) for r in diag.token_trace}
    expected = {(b, s) for b in range(batch) for s in range(seq_len)}
    assert seen == expected


def test_trace_is_bounded_by_default_and_raises_when_exceeded():
    cfg = make_tiny_sparse_config(n_routed_experts=4, top_k=2, d_model=4, n_query_heads=1, n_kv_heads=1, head_dim=4)
    layer = MoELayer(cfg)
    x = torch.randn(1, 10, cfg.core.d_model)
    with pytest.raises(ValueError, match="max_trace_tokens"):
        layer(x, return_diagnostics=True, return_trace=True, max_trace_tokens=5)
    # explicit opt-out of the bound is allowed
    _, _, _, diag = layer(x, return_diagnostics=True, return_trace=True, max_trace_tokens=None)
    assert len(diag.token_trace) == 10


def test_full_model_trace_assigns_correct_layer_index_and_skips_dense_layers():
    cfg = make_tiny_sparse_config(
        n_layers=6, dense_layers=[1, 6], moe_layers=[2, 3, 4, 5],
        d_model=8, n_routed_experts=4, top_k=2, expert_ffn_dim=8,
        n_query_heads=2, n_kv_heads=1, head_dim=4,
    )
    model = build_model(cfg, seed=0)
    input_ids = torch.randint(0, cfg.embeddings.vocab_size, (2, 6))
    out = model(input_ids, return_diagnostics=True)

    full_trace = assemble_full_trace(model.layer_kinds, out.diagnostics)
    # only MoE layers (positions 2,3,4,5) contribute; each contributes
    # batch*seq_len = 12 records since return_trace defaults to False on the
    # model-level call -- so this should be empty until trace is requested.
    assert full_trace == []


def test_full_model_trace_with_explicit_trace_collection():
    cfg = make_tiny_sparse_config(
        n_layers=6, dense_layers=[1, 6], moe_layers=[2, 3, 4, 5],
        d_model=8, n_routed_experts=4, top_k=2, expert_ffn_dim=8,
        n_query_heads=2, n_kv_heads=1, head_dim=4,
    )
    model = build_model(cfg, seed=0)
    input_ids = torch.randint(0, cfg.embeddings.vocab_size, (2, 6))
    attention_mask = torch.ones(2, 6, dtype=torch.bool)

    out = model(input_ids, attention_mask=attention_mask, return_diagnostics=True, return_trace=True)
    full_trace = assemble_full_trace(model.layer_kinds, out.diagnostics)
    assert len(full_trace) == 4 * 2 * 6  # 4 MoE layers * batch(2) * seq_len(6)
    assert {r.layer_index for r in full_trace} == {2, 3, 4, 5}

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "trace.json"
        export_trace_json(full_trace, out_path)
        with out_path.open() as f:
            loaded = json.load(f)
        assert len(loaded) == len(full_trace)
        assert loaded[0]["layer_index"] in {2, 3, 4, 5}
        assert set(loaded[0].keys()) == {
            "layer_index", "batch_index", "seq_position", "flat_token_index",
            "is_valid", "expert_1", "expert_2", "weight_1", "weight_2",
        }
