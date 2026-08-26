"""Grouped-query attention: head mapping, causal masking, padding masking,
and an independent manual-reference numerical check."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest

from juniper_auto.model.attention import GroupedQueryAttention, build_attention_mask, repeat_kv
from tests.model_fixtures import make_tiny_sparse_config


def test_repeat_kv_maps_each_kv_head_to_contiguous_query_heads():
    # 2 kv heads, n_rep=3 -> 6 query heads: kv head 0 serves query heads
    # [0,1,2], kv head 1 serves query heads [3,4,5].
    kv = torch.zeros(1, 2, 1, 4)
    kv[0, 0, 0, :] = 1.0
    kv[0, 1, 0, :] = 2.0
    out = repeat_kv(kv, n_rep=3)
    assert out.shape == (1, 6, 1, 4)
    assert torch.equal(out[0, 0], out[0, 1]) and torch.equal(out[0, 1], out[0, 2])
    assert torch.equal(out[0, 3], out[0, 4]) and torch.equal(out[0, 4], out[0, 5])
    assert (out[0, 0] == 1.0).all()
    assert (out[0, 3] == 2.0).all()


def test_repeat_kv_noop_when_n_rep_is_one():
    kv = torch.randn(2, 4, 3, 5)
    out = repeat_kv(kv, n_rep=1)
    assert out is kv


def test_gqa_module_shapes_match_config():
    cfg = make_tiny_sparse_config(n_query_heads=4, n_kv_heads=2, head_dim=8, d_model=32)
    attn = GroupedQueryAttention(cfg)
    assert attn.q_proj.weight.shape == (32, 32)  # 4*8 -> 32 output, 32 input
    assert attn.k_proj.weight.shape == (16, 32)  # 2*8 -> 16
    assert attn.v_proj.weight.shape == (16, 32)
    assert attn.o_proj.weight.shape == (32, 32)
    assert attn.n_rep == 2


def test_gqa_forward_shape_and_no_bias():
    cfg = make_tiny_sparse_config(n_query_heads=4, n_kv_heads=2, head_dim=8, d_model=32)
    attn = GroupedQueryAttention(cfg)
    x = torch.randn(2, 7, 32)
    pos = torch.arange(7).unsqueeze(0).expand(2, 7)
    out = attn(x, pos)
    assert out.shape == (2, 7, 32)
    for proj in (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj):
        assert proj.bias is None


@pytest.mark.parametrize(
    "position_shape,mask_shape,match",
    [((1, 4), (2, 5), "position_ids"), ((2, 5), (1, 5), "key_valid_mask")],
)
def test_gqa_rejects_invalid_position_or_mask_shapes(position_shape, mask_shape, match):
    cfg = make_tiny_sparse_config()
    attn = GroupedQueryAttention(cfg)
    x = torch.randn(2, 5, cfg.core.d_model)
    position_ids = torch.zeros(position_shape, dtype=torch.long)
    key_valid_mask = torch.ones(mask_shape, dtype=torch.bool)
    with pytest.raises(ValueError, match=match):
        attn(x, position_ids, key_valid_mask)


def test_causality_future_tokens_do_not_affect_earlier_logits():
    cfg = make_tiny_sparse_config()
    attn = GroupedQueryAttention(cfg)
    attn.eval()
    torch.manual_seed(0)
    x = torch.randn(1, 6, cfg.core.d_model)
    pos = torch.arange(6).unsqueeze(0)

    x_mutated = x.clone()
    x_mutated[0, 4:, :] = torch.randn_like(x_mutated[0, 4:, :])  # change only future positions

    with torch.no_grad():
        out_a = attn(x, pos)
        out_b = attn(x_mutated, pos)

    torch.testing.assert_close(out_a[:, :4, :], out_b[:, :4, :], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(out_a[:, 4:, :], out_b[:, 4:, :])


def test_padding_key_does_not_affect_valid_query_output():
    cfg = make_tiny_sparse_config()
    attn = GroupedQueryAttention(cfg)
    attn.eval()
    torch.manual_seed(1)
    x = torch.randn(1, 5, cfg.core.d_model)
    pos = torch.arange(5).unsqueeze(0)
    key_valid = torch.tensor([[True, True, True, False, False]])

    x_mutated_padding = x.clone()
    x_mutated_padding[0, 3:, :] = torch.randn_like(x_mutated_padding[0, 3:, :])

    with torch.no_grad():
        out_a = attn(x, pos, key_valid)
        out_b = attn(x_mutated_padding, pos, key_valid)

    # Valid positions (0,1,2) never attend to keys 3,4 (both future and
    # padded), so their output must be unaffected by what's written there.
    torch.testing.assert_close(out_a[:, :3, :], out_b[:, :3, :], atol=1e-5, rtol=1e-5)


def test_all_padded_row_does_not_produce_nan():
    seq_len = 4
    key_valid = torch.tensor([[False, False, False, False]])
    mask = build_attention_mask(seq_len, key_valid, device=torch.device("cpu"))
    # Every row must have at least one allowed key (self-fallback), else SDPA softmax NaNs.
    assert mask.any(dim=-1).all()

    cfg = make_tiny_sparse_config()
    attn = GroupedQueryAttention(cfg)
    x = torch.randn(1, seq_len, cfg.core.d_model)
    pos = torch.arange(seq_len).unsqueeze(0)
    out = attn(x, pos, key_valid)
    assert torch.isfinite(out).all()


def test_build_attention_mask_is_pure_causal_without_padding_arg():
    mask = build_attention_mask(4, None, device=torch.device("cpu"))
    expected = torch.tril(torch.ones(4, 4, dtype=torch.bool)).view(1, 1, 4, 4)
    assert torch.equal(mask, expected)


def test_gqa_matches_independent_manual_reference():
    # Fully manual attention computation for a tiny, hand-inspectable
    # config (no QK-Norm, no RoPE, so the only thing under test is the
    # GQA head-repetition + masked-softmax-attention arithmetic).
    cfg = make_tiny_sparse_config(
        n_query_heads=2, n_kv_heads=1, head_dim=4, d_model=8, qk_norm=False, attention_scale=0.5
    )
    attn = GroupedQueryAttention(cfg)
    torch.manual_seed(7)
    x = torch.randn(1, 3, 8)
    pos = torch.arange(3).unsqueeze(0)

    with torch.no_grad():
        out = attn(x, pos)

        q = attn.q_proj(x).view(1, 3, 2, 4).transpose(1, 2)  # [1,2,3,4]
        k = attn.k_proj(x).view(1, 3, 1, 4).transpose(1, 2)  # [1,1,3,4]
        v = attn.v_proj(x).view(1, 3, 1, 4).transpose(1, 2)
        cos, sin = attn.rope(pos)
        from juniper_auto.model.rope import apply_rotary_pos_emb

        q, k = apply_rotary_pos_emb(q, k, cos, sin, attn.rotary_dim)
        k_rep = k.expand(1, 2, 3, 4)  # n_rep=2, single kv head serves both query heads
        v_rep = v.expand(1, 2, 3, 4)

        causal = torch.tril(torch.ones(3, 3, dtype=torch.bool))
        manual_out = torch.zeros(1, 2, 3, 4)
        for h in range(2):
            scores = (q[0, h] @ k_rep[0, h].T) * 0.5
            scores = scores.masked_fill(~causal, float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            manual_out[0, h] = weights @ v_rep[0, h]
        manual_out = manual_out.transpose(1, 2).reshape(1, 3, 8)
        expected = attn.o_proj(manual_out)

    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)
