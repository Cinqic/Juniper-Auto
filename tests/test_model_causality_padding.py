"""Full-model causality and padding correctness -- not just at the
attention-module level, but through the whole stack (embeddings, all
blocks, MoE routing, final norm, LM head)."""

from __future__ import annotations

import torch

from juniper_auto.model import build_model
from tests.model_fixtures import make_tiny_dense_config, make_tiny_sparse_config


def test_full_model_causality_via_future_token_mutation():
    cfg = make_tiny_sparse_config()
    model = build_model(cfg, seed=0)
    model.eval()

    torch.manual_seed(0)
    input_ids = torch.randint(0, cfg.embeddings.vocab_size, (1, 7))
    mutated = input_ids.clone()
    mutated[0, 4:] = (mutated[0, 4:] + 1) % cfg.embeddings.vocab_size

    with torch.no_grad():
        out_a = model(input_ids)
        out_b = model(mutated)

    torch.testing.assert_close(out_a.logits[:, :4, :], out_b.logits[:, :4, :], atol=1e-4, rtol=1e-4)
    assert not torch.allclose(out_a.logits[:, 4:, :], out_b.logits[:, 4:, :])


def test_full_model_padding_key_does_not_affect_valid_logits():
    cfg = make_tiny_sparse_config()
    model = build_model(cfg, seed=0)
    model.eval()

    torch.manual_seed(1)
    input_ids = torch.randint(0, cfg.embeddings.vocab_size, (1, 6))
    attention_mask = torch.tensor([[True, True, True, True, False, False]])

    mutated_ids = input_ids.clone()
    mutated_ids[0, 4:] = (mutated_ids[0, 4:] + 7) % cfg.embeddings.vocab_size

    with torch.no_grad():
        out_a = model(input_ids, attention_mask=attention_mask)
        out_b = model(mutated_ids, attention_mask=attention_mask)

    torch.testing.assert_close(out_a.logits[:, :4, :], out_b.logits[:, :4, :], atol=1e-4, rtol=1e-4)


def test_padding_excluded_from_causal_loss():
    cfg = make_tiny_sparse_config()
    model = build_model(cfg, seed=0)

    torch.manual_seed(2)
    input_ids = torch.randint(0, cfg.embeddings.vocab_size, (1, 6))
    attention_mask = torch.tensor([[True, True, True, True, False, False]])
    labels = input_ids.clone()
    labels[~attention_mask] = -100

    out_a = model(input_ids, attention_mask=attention_mask, labels=labels)

    mutated_ids = input_ids.clone()
    mutated_ids[0, 4:] = (mutated_ids[0, 4:] + 3) % cfg.embeddings.vocab_size
    out_b = model(mutated_ids, attention_mask=attention_mask, labels=labels)

    torch.testing.assert_close(out_a.lm_loss, out_b.lm_loss, atol=1e-5, rtol=1e-5)


def test_padding_excluded_from_moe_auxiliary_losses():
    cfg = make_tiny_sparse_config()
    model = build_model(cfg, seed=0)

    torch.manual_seed(3)
    input_ids = torch.randint(0, cfg.embeddings.vocab_size, (1, 6))
    attention_mask = torch.tensor([[True, True, True, True, False, False]])

    out_a = model(input_ids, attention_mask=attention_mask)

    mutated_ids = input_ids.clone()
    mutated_ids[0, 4:] = (mutated_ids[0, 4:] + 5) % cfg.embeddings.vocab_size
    out_b = model(mutated_ids, attention_mask=attention_mask)

    torch.testing.assert_close(out_a.load_balance_loss_raw, out_b.load_balance_loss_raw, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(out_a.router_z_loss_raw, out_b.router_z_loss_raw, atol=1e-6, rtol=1e-6)


def test_no_nan_with_a_fully_padded_row_in_the_batch():
    cfg = make_tiny_sparse_config()
    model = build_model(cfg, seed=0)
    input_ids = torch.randint(0, cfg.embeddings.vocab_size, (2, 5))
    attention_mask = torch.tensor([[True, True, True, True, True], [False, False, False, False, False]])
    labels = input_ids.clone()
    labels[~attention_mask] = -100

    out = model(input_ids, attention_mask=attention_mask, labels=labels)
    assert torch.isfinite(out.logits).all()
    assert torch.isfinite(out.loss)


def test_dense_model_has_zero_moe_losses_cleanly():
    cfg = make_tiny_dense_config()
    model = build_model(cfg, seed=0)
    input_ids = torch.randint(0, cfg.embeddings.vocab_size, (1, 4))
    labels = input_ids.clone()
    out = model(input_ids, labels=labels)

    assert out.load_balance_loss.item() == 0.0
    assert out.router_z_loss.item() == 0.0
    torch.testing.assert_close(out.loss, out.lm_loss)
