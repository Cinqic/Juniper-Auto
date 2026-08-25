"""Model state_dict save/load: parameters match, weight tying survives
the round trip, and outputs match before and after restoration."""

from __future__ import annotations

import io

import torch

from juniper_auto.model import build_model
from juniper_auto.model.inspection import verify_weight_tying
from tests.model_fixtures import make_tiny_sparse_config


def test_state_dict_round_trip_preserves_parameters_and_outputs():
    cfg = make_tiny_sparse_config()
    model = build_model(cfg, seed=0)
    model.eval()

    torch.manual_seed(0)
    input_ids = torch.randint(0, cfg.embeddings.vocab_size, (2, 6))
    with torch.no_grad():
        out_before = model(input_ids)

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    loaded_state = torch.load(buffer, weights_only=True)

    restored = build_model(cfg, seed=999)  # different seed -> different weights before loading
    restored.load_state_dict(loaded_state)
    restored.eval()

    for (name_a, p_a), (name_b, p_b) in zip(model.named_parameters(), restored.named_parameters()):
        assert name_a == name_b
        torch.testing.assert_close(p_a, p_b)

    assert verify_weight_tying(restored)
    assert restored.embedding.weight.data_ptr() == restored.lm_head.weight.data_ptr()

    with torch.no_grad():
        out_after = restored(input_ids)
    torch.testing.assert_close(out_before.logits, out_after.logits, atol=1e-6, rtol=1e-6)


def test_state_dict_contains_both_tied_key_names_but_shares_storage():
    cfg = make_tiny_sparse_config()
    model = build_model(cfg, seed=0)
    state = model.state_dict()
    assert "embedding.weight" in state
    assert "lm_head.weight" in state
    assert state["embedding.weight"].data_ptr() == state["lm_head.weight"].data_ptr()


def test_mutating_loaded_model_does_not_affect_original():
    cfg = make_tiny_sparse_config()
    model = build_model(cfg, seed=0)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    loaded_state = torch.load(buffer, weights_only=True)

    restored = build_model(cfg, seed=0)
    restored.load_state_dict(loaded_state)

    with torch.no_grad():
        restored.embedding.weight.add_(1.0)

    assert not torch.allclose(model.embedding.weight, restored.embedding.weight)
