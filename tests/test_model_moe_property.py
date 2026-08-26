"""Randomized property coverage for MoE dropless routing invariants (Phase 2
instructions section 18): across many seeds, batch sizes, sequence lengths,
padding layouts, expert counts, and activation magnitudes, prove the
dropless contract holds exactly -- not just on one hand-picked tensor.
No external property-testing dependency is used; this is a deterministic,
parametrized pytest suite, per the instructions' explicit allowance."""

from __future__ import annotations

import itertools
import math

import pytest
import torch

from juniper_auto.model.moe import MoELayer
from tests.model_fixtures import make_tiny_sparse_config


def _make_valid_mask(batch, seq_len, layout, generator):
    valid = torch.ones(batch, seq_len, dtype=torch.bool)
    if layout == "none":
        pass
    elif layout == "trailing":
        valid[:, seq_len // 2 :] = False
    elif layout == "leading":
        valid[:, : seq_len // 2] = False
    elif layout == "scattered":
        valid[torch.rand(batch, seq_len, generator=generator) < 0.5] = False
    elif layout == "per_row_different_lengths":
        for b in range(batch):
            length = 1 + (b % seq_len)
            valid[b, length:] = False
    elif layout == "almost_all_padding":
        valid[:] = False
        valid[0, 0] = True
    elif layout == "fully_padded_row":
        valid[0, :] = False
    return valid


CASES = list(
    itertools.product(
        range(6),  # seeds
        [1, 2, 3],  # batch
        [5, 7, 12],  # seq_len
        [4, 6, 8],  # n_routed_experts
        [1, 2, 3],  # top_k
        ["none", "trailing", "leading", "scattered", "per_row_different_lengths", "almost_all_padding", "fully_padded_row"],
    )
)
# Full cross product is large; subsample deterministically (index stride) to
# keep CI fast while still exercising every axis many times over.
CASES = CASES[::17]


@pytest.mark.parametrize("seed,batch,seq_len,n_experts,top_k,layout", CASES)
def test_dropless_invariants_hold_across_randomized_configurations(seed, batch, seq_len, n_experts, top_k, layout):
    if top_k > n_experts:
        pytest.skip("top_k cannot exceed n_routed_experts")
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)

    cfg = make_tiny_sparse_config(
        n_routed_experts=n_experts,
        top_k=top_k,
        d_model=8,
        expert_ffn_dim=8,
        n_query_heads=1,
        n_kv_heads=1,
        head_dim=8,
    )
    layer = MoELayer(cfg)
    x = torch.randn(batch, seq_len, cfg.core.d_model, generator=generator) * (1.0 + 5.0 * torch.rand((), generator=generator).item())
    valid_mask = _make_valid_mask(batch, seq_len, layout, generator)
    n_valid = int(valid_mask.sum().item())

    out, lb, z, diag = layer(x, valid_mask=valid_mask, return_diagnostics=True)

    # Core dropless contract.
    assert diag.assignment_counts_per_expert.sum().item() == n_valid * top_k

    # Every token (valid or padding) gets exactly top_k *unique* experts.
    flat_topk = diag.topk_idx
    for row in flat_topk.tolist():
        assert len(set(row)) == top_k

    # Renormalized weights sum to 1 for every token.
    torch.testing.assert_close(diag.topk_weights.sum(dim=-1), torch.ones(batch * seq_len), atol=1e-4, rtol=1e-4)

    # Shapes and finiteness.
    assert out.shape == (batch, seq_len, cfg.core.d_model)
    assert torch.isfinite(out).all()
    assert torch.isfinite(lb).all() and torch.isfinite(z).all()

    # Expert-pair coactivation: total pairs counted == n_valid * C(top_k, 2),
    # and it is never larger than the number of valid tokens times the
    # number of distinct pairs one token can contribute.
    if top_k >= 2:
        expected_pairs = n_valid * math.comb(top_k, 2)
        assert diag.expert_pair_coactivation.sum().item() == expected_pairs
        # strictly upper-triangular: no mass on/under the diagonal
        assert torch.triu(diag.expert_pair_coactivation, diagonal=1).sum() == diag.expert_pair_coactivation.sum()


@pytest.mark.parametrize("seed", range(10))
def test_no_duplicate_expert_within_one_tokens_topk(seed):
    torch.manual_seed(seed)
    cfg = make_tiny_sparse_config(
        n_routed_experts=8, top_k=3, d_model=8, expert_ffn_dim=8, n_query_heads=1, n_kv_heads=1, head_dim=8
    )
    layer = MoELayer(cfg)
    x = torch.randn(3, 17, cfg.core.d_model)
    _, _, _, diag = layer(x, return_diagnostics=True)
    idx = diag.topk_idx
    assert (idx[:, 0] != idx[:, 1]).all()
    assert (idx[:, 0] != idx[:, 2]).all()
    assert (idx[:, 1] != idx[:, 2]).all()


def test_padding_never_alters_valid_token_statistics_across_many_layouts():
    torch.manual_seed(21)
    cfg = make_tiny_sparse_config(
        n_routed_experts=6, top_k=2, d_model=8, expert_ffn_dim=8, n_query_heads=1, n_kv_heads=1, head_dim=8
    )
    layer = MoELayer(cfg)
    x = torch.randn(2, 10, cfg.core.d_model)
    valid = torch.ones(2, 10, dtype=torch.bool)
    valid[0, 6:] = False
    valid[1, :3] = False

    _, lb_a, z_a, diag_a = layer(x, valid_mask=valid, return_diagnostics=True)

    x_mutated_padding = x.clone()
    x_mutated_padding[~valid] = torch.randn_like(x_mutated_padding[~valid]) * 1000.0
    _, lb_b, z_b, diag_b = layer(x_mutated_padding, valid_mask=valid, return_diagnostics=True)

    torch.testing.assert_close(lb_a, lb_b)
    torch.testing.assert_close(z_a, z_b)
    assert torch.equal(diag_a.assignment_counts_per_expert, diag_b.assignment_counts_per_expert)
    assert torch.equal(diag_a.expert_pair_coactivation, diag_b.expert_pair_coactivation)
    torch.testing.assert_close(
        diag_a.entropy[valid.reshape(-1)].mean(), diag_b.entropy[valid.reshape(-1)].mean()
    )
