"""End-to-end shape tests across batch sizes, sequence lengths, and
padding patterns, for both sparse and dense tiny configs."""

from __future__ import annotations

import pytest
import torch

from juniper_auto.model import build_model
from tests.model_fixtures import make_tiny_dense_config, make_tiny_sparse_config


@pytest.mark.parametrize("kind", ["sparse", "dense"])
@pytest.mark.parametrize("batch_size,seq_len", [(1, 1), (1, 5), (3, 1), (2, 9), (4, 16)])
def test_forward_shapes_across_batch_and_seq_len(kind, batch_size, seq_len):
    cfg = make_tiny_sparse_config() if kind == "sparse" else make_tiny_dense_config()
    model = build_model(cfg, seed=0)
    input_ids = torch.randint(0, cfg.embeddings.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()

    out = model(input_ids, labels=labels)
    assert out.logits.shape == (batch_size, seq_len, cfg.embeddings.vocab_size)
    assert torch.isfinite(out.logits).all()
    assert out.loss.dim() == 0
    assert out.lm_loss.dim() == 0
    if seq_len > 1:
        # seq_len == 1 has no valid next-token target at all (nothing
        # follows the only position), so the shifted loss is legitimately
        # undefined (NaN) rather than a bug -- see
        # test_single_token_sequence_has_no_valid_loss_target below.
        assert torch.isfinite(out.loss)


@pytest.mark.parametrize("kind", ["sparse", "dense"])
def test_forward_with_various_padding_patterns(kind):
    cfg = make_tiny_sparse_config() if kind == "sparse" else make_tiny_dense_config()
    model = build_model(cfg, seed=0)
    batch, seq_len = 3, 8
    input_ids = torch.randint(0, cfg.embeddings.vocab_size, (batch, seq_len))
    labels = input_ids.clone()

    patterns = [
        [True] * seq_len,  # no padding
        [True] * 5 + [False] * 3,  # trailing padding
        [True, False, True, True, False, True, True, True],  # scattered
    ]
    attention_mask = torch.tensor(patterns)
    labels = torch.where(attention_mask, labels, torch.full_like(labels, -100))

    out = model(input_ids, attention_mask=attention_mask, labels=labels)
    assert out.logits.shape == (batch, seq_len, cfg.embeddings.vocab_size)
    assert torch.isfinite(out.logits).all()
    assert torch.isfinite(out.loss)


def test_diagnostics_omitted_by_default_and_present_on_request():
    cfg = make_tiny_sparse_config()
    model = build_model(cfg, seed=0)
    input_ids = torch.randint(0, cfg.embeddings.vocab_size, (1, 4))

    out_default = model(input_ids)
    assert out_default.diagnostics is None

    out_requested = model(input_ids, return_diagnostics=True)
    assert out_requested.diagnostics is not None
    assert len(out_requested.diagnostics) == cfg.core.n_layers
    moe_diag_present = [d is not None for d, kind in zip(out_requested.diagnostics, model.layer_kinds)]
    for present, kind in zip(moe_diag_present, model.layer_kinds):
        assert present == (kind == "moe")


def test_position_ids_uniform_shift_is_invariant_a_defining_rope_property():
    # RoPE encodes *relative* position via the query/key dot product, so
    # shifting every position by the same constant leaves all pairwise
    # relative offsets (and therefore attention scores and outputs)
    # unchanged. This is expected RoPE behavior, not a bug.
    cfg = make_tiny_sparse_config()
    model = build_model(cfg, seed=0)
    model.eval()
    input_ids = torch.randint(0, cfg.embeddings.vocab_size, (1, 5))

    with torch.no_grad():
        out_default = model(input_ids)
        shifted_positions = torch.arange(5).unsqueeze(0) + 100
        out_shifted = model(input_ids, position_ids=shifted_positions)
    torch.testing.assert_close(out_default.logits, out_shifted.logits, atol=1e-4, rtol=1e-4)


def test_position_ids_with_different_relative_spacing_changes_output():
    cfg = make_tiny_sparse_config()
    model = build_model(cfg, seed=0)
    model.eval()
    input_ids = torch.randint(0, cfg.embeddings.vocab_size, (1, 5))

    with torch.no_grad():
        out_default = model(input_ids)  # positions 0,1,2,3,4
        wide_spacing = torch.tensor([[0, 2, 4, 6, 8]])  # different relative gaps
        out_wide = model(input_ids, position_ids=wide_spacing)
    assert not torch.allclose(out_default.logits, out_wide.logits)


def test_single_token_sequence_has_no_valid_loss_target():
    # A length-1 sequence has no next-token label at all; the shifted loss
    # is legitimately NaN (0 supervised predictions), not a computation bug.
    cfg = make_tiny_sparse_config()
    model = build_model(cfg, seed=0)
    input_ids = torch.randint(0, cfg.embeddings.vocab_size, (1, 1))
    out = model(input_ids, labels=input_ids.clone())
    assert torch.isnan(out.loss)
    assert torch.isfinite(out.logits).all()
