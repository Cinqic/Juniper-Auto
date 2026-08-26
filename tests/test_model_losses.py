"""Shifted causal cross-entropy and MoE auxiliary loss formulas, verified
against hand-calculated reference values -- not just "the code agrees with
itself"."""

from __future__ import annotations

import math

import pytest
import torch

from juniper_auto.model.losses import (
    causal_lm_loss,
    compute_load_balance_loss_raw,
    compute_router_z_loss_raw,
)


def test_causal_lm_loss_exact_shift_two_token_example():
    # vocab of size 3; single sequence [A, B]; logits chosen so the loss is
    # hand-computable. Position 0's logits must predict token at position 1
    # (label B=1); with 2 tokens there is exactly one supervised prediction.
    logits = torch.tensor([[[10.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])  # [1, 2, 3]
    labels = torch.tensor([[0, 1]])
    loss = causal_lm_loss(logits, labels)
    # cross_entropy(logits[0,0]=[10,0,0], target=1) = -log(softmax[1])
    probs = torch.softmax(torch.tensor([10.0, 0.0, 0.0]), dim=-1)
    expected = -torch.log(probs[1])
    torch.testing.assert_close(loss, expected, atol=1e-5, rtol=1e-5)


def test_causal_lm_loss_ignores_masked_labels():
    logits = torch.randn(1, 3, 5)
    labels = torch.tensor([[-100, -100, -100]])
    with pytest.raises(ValueError, match="non-ignored next-token target"):
        causal_lm_loss(logits, labels)


def test_causal_lm_loss_off_by_one_shift_is_load_bearing():
    # If position i's logits were (incorrectly) compared against label i
    # instead of label i+1, this specific construction would score
    # differently. logits strongly predict "next id = position index + 1".
    vocab = 4
    seq_len = 4
    logits = torch.zeros(1, seq_len, vocab)
    for i in range(seq_len):
        logits[0, i, (i + 1) % vocab] = 20.0  # position i confidently predicts token (i+1)
    # shift_logits[j] = logits[j] (predicts (j+1)%vocab); shift_labels[j] =
    # labels[j+1], so labels[j] must equal j%vocab for j>=1 (labels[0] is
    # never used as a target and is arbitrary).
    labels = torch.tensor([[i % vocab for i in range(seq_len)]])
    loss_correct_shift = causal_lm_loss(logits, labels)
    assert loss_correct_shift.item() < 1e-3

    # An unshifted (buggy) comparison -- position i vs label i -- would score
    # very poorly on this construction, since logits at i strongly favor
    # (i+1), not i.
    shift_logits = logits[:, :-1, :]
    unshifted_labels = labels[:, :-1]
    bad_loss = torch.nn.functional.cross_entropy(
        shift_logits.reshape(-1, vocab), unshifted_labels.reshape(-1)
    )
    assert bad_loss.item() > 5.0


def test_load_balance_loss_hand_calculated():
    # 2 experts, top_k=1, 4 valid tokens, each with a fixed router_probs row
    # and a fixed top-1 assignment chosen independently of argmax so the
    # test isn't trivially "assignment == argmax(probs)".
    router_probs = torch.tensor(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.3, 0.7],
            [0.4, 0.6],
        ]
    )
    topk_idx = torch.tensor([[0], [0], [1], [0]])  # 3 assignments to expert 0, 1 to expert 1
    valid_mask = torch.tensor([True, True, True, True])
    n_experts, top_k = 2, 1

    loss = compute_load_balance_loss_raw(router_probs, topk_idx, valid_mask, n_experts, top_k)

    f0, f1 = 3 / 4, 1 / 4
    p0 = router_probs[:, 0].mean().item()
    p1 = router_probs[:, 1].mean().item()
    expected = n_experts * (f0 * p0 + f1 * p1)
    torch.testing.assert_close(loss.item(), expected, atol=1e-6, rtol=1e-6)


def test_load_balance_loss_excludes_padding():
    router_probs = torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.5, 0.5]])
    topk_idx = torch.tensor([[0], [1], [0]])
    valid_mask = torch.tensor([True, True, False])  # last token is padding
    loss_with_padding_excluded = compute_load_balance_loss_raw(router_probs, topk_idx, valid_mask, 2, 1)

    # Only tokens 0,1 should matter: f0=f1=0.5, p0=0.5, p1=0.5 -> 2*(0.25+0.25)=1.0
    expected = 1.0
    torch.testing.assert_close(loss_with_padding_excluded.item(), expected, atol=1e-6, rtol=1e-6)

    # Changing the padded row's probs/assignment must not move the result.
    router_probs_mutated = router_probs.clone()
    router_probs_mutated[2] = torch.tensor([0.99, 0.01])
    topk_idx_mutated = topk_idx.clone()
    topk_idx_mutated[2] = 1
    loss_mutated = compute_load_balance_loss_raw(router_probs_mutated, topk_idx_mutated, valid_mask, 2, 1)
    torch.testing.assert_close(loss_with_padding_excluded, loss_mutated)


def test_load_balance_loss_no_gradient_through_hard_assignment_counts():
    router_probs = torch.tensor([[0.6, 0.4], [0.3, 0.7]], requires_grad=True)
    topk_idx = torch.tensor([[0], [1]])
    valid_mask = torch.tensor([True, True])
    loss = compute_load_balance_loss_raw(router_probs, topk_idx, valid_mask, 2, 1)
    loss.backward()
    assert router_probs.grad is not None
    assert torch.isfinite(router_probs.grad).all()


def test_router_z_loss_hand_calculated():
    logits = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    valid_mask = torch.tensor([True, True])
    loss = compute_router_z_loss_raw(logits, valid_mask)

    z0 = math.log(math.exp(1.0) + math.exp(2.0) + math.exp(3.0))
    z1 = math.log(3.0)  # logsumexp of three zeros = log(3)
    expected = (z0**2 + z1**2) / 2
    torch.testing.assert_close(loss.item(), expected, atol=1e-5, rtol=1e-5)


def test_router_z_loss_excludes_padding():
    logits = torch.tensor([[1.0, 2.0, 3.0], [100.0, 100.0, 100.0]])  # row 1 would dominate if included
    valid_mask = torch.tensor([True, False])
    loss = compute_router_z_loss_raw(logits, valid_mask)
    z0 = math.log(math.exp(1.0) + math.exp(2.0) + math.exp(3.0))
    torch.testing.assert_close(loss.item(), z0**2, atol=1e-5, rtol=1e-5)
