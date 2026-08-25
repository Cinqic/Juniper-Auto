"""RoPE correctness: theta, rotary dimension, position dependence,
determinism, norm preservation, and Q/K application."""

from __future__ import annotations

import math

import pytest
import torch

from juniper_auto.model.rope import RotaryEmbedding, apply_rotary_pos_emb, rotate_half


def test_inv_freq_uses_configured_theta_and_dim():
    dim = 8
    theta = 100000.0
    rope = RotaryEmbedding(dim=dim, theta=theta)
    expected = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    torch.testing.assert_close(rope.inv_freq, expected)
    assert rope.inv_freq.numel() == dim // 2


def test_different_theta_changes_rotation_for_same_positions():
    dim = 8
    pos = torch.tensor([[0, 1, 2, 3]])
    cos_a, sin_a = RotaryEmbedding(dim=dim, theta=10000.0)(pos)
    cos_b, sin_b = RotaryEmbedding(dim=dim, theta=100000.0)(pos)
    assert not torch.allclose(cos_a[:, 1:], cos_b[:, 1:])  # position 0 is theta-invariant (angle=0)


def test_position_zero_is_identity_rotation():
    dim = 8
    rope = RotaryEmbedding(dim=dim, theta=100000.0)
    cos, sin = rope(torch.tensor([[0]]))
    assert torch.allclose(cos, torch.ones_like(cos))
    assert torch.allclose(sin, torch.zeros_like(sin), atol=1e-6)


def test_rope_is_position_dependent():
    dim = 8
    rope = RotaryEmbedding(dim=dim, theta=100000.0)
    torch.manual_seed(0)
    q = torch.randn(1, 1, 1, dim)
    k = q.clone()
    cos0, sin0 = rope(torch.tensor([[0]]))
    cos5, sin5 = rope(torch.tensor([[5]]))
    q0, _ = apply_rotary_pos_emb(q, k, cos0, sin0, dim)
    q5, _ = apply_rotary_pos_emb(q, k, cos5, sin5, dim)
    assert not torch.allclose(q0, q5)


def test_rope_deterministic_same_seed_and_positions():
    dim = 8
    rope = RotaryEmbedding(dim=dim, theta=100000.0)
    pos = torch.tensor([[0, 1, 2, 3, 4]])
    cos_a, sin_a = rope(pos)
    cos_b, sin_b = rope(pos)
    torch.testing.assert_close(cos_a, cos_b)
    torch.testing.assert_close(sin_a, sin_b)


def test_rope_preserves_vector_norm():
    # Rotation is orthogonal per 2D pair, so it must not change the L2 norm
    # of the rotated slice.
    dim = 16
    rope = RotaryEmbedding(dim=dim, theta=100000.0)
    torch.manual_seed(2)
    q = torch.randn(2, 3, 7, dim)
    k = torch.randn(2, 3, 7, dim)
    cos, sin = rope(torch.arange(7).unsqueeze(0).expand(2, 7))
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin, dim)
    torch.testing.assert_close(q_rot.norm(dim=-1), q.norm(dim=-1), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(k_rot.norm(dim=-1), k.norm(dim=-1), atol=1e-4, rtol=1e-4)


def test_rope_applied_independently_to_q_and_k():
    dim = 8
    rope = RotaryEmbedding(dim=dim, theta=100000.0)
    torch.manual_seed(3)
    q = torch.randn(1, 1, 4, dim)
    k = torch.randn(1, 1, 4, dim) * 5.0  # different magnitude/values than q
    cos, sin = rope(torch.arange(4).unsqueeze(0))
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin, dim)
    # Rotating q must not depend on k's values and vice versa: verify by
    # rotating each alone with the identical (cos, sin) and comparing.
    q_rot_alone, _ = apply_rotary_pos_emb(q, q, cos, sin, dim)
    _, k_rot_alone = apply_rotary_pos_emb(k, k, cos, sin, dim)
    torch.testing.assert_close(q_rot, q_rot_alone)
    torch.testing.assert_close(k_rot, k_rot_alone)


def test_partial_rotary_dim_leaves_remaining_channels_untouched():
    head_dim = 16
    rotary_dim = 8
    rope = RotaryEmbedding(dim=rotary_dim, theta=100000.0)
    torch.manual_seed(4)
    q = torch.randn(1, 1, 3, head_dim)
    k = torch.randn(1, 1, 3, head_dim)
    cos, sin = rope(torch.arange(3).unsqueeze(0))
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin, rotary_dim)
    torch.testing.assert_close(q_rot[..., rotary_dim:], q[..., rotary_dim:])
    torch.testing.assert_close(k_rot[..., rotary_dim:], k[..., rotary_dim:])
    assert not torch.allclose(q_rot[..., :rotary_dim], q[..., :rotary_dim])


def test_initial_scaling_defaults_to_a_no_op():
    dim = 8
    unscaled = RotaryEmbedding(dim=dim, theta=100000.0, initial_scaling=1.0)
    default = RotaryEmbedding(dim=dim, theta=100000.0)
    pos = torch.tensor([[0, 3, 7]])
    cos_a, sin_a = unscaled(pos)
    cos_b, sin_b = default(pos)
    torch.testing.assert_close(cos_a, cos_b)
    torch.testing.assert_close(sin_a, sin_b)


def test_initial_scaling_divides_effective_position():
    dim = 8
    scale = 2.0
    scaled = RotaryEmbedding(dim=dim, theta=100000.0, initial_scaling=scale)
    unscaled = RotaryEmbedding(dim=dim, theta=100000.0, initial_scaling=1.0)
    # Position 8 under scaling=2.0 must match position 4 unscaled.
    cos_scaled, sin_scaled = scaled(torch.tensor([[8]]))
    cos_unscaled, sin_unscaled = unscaled(torch.tensor([[4]]))
    torch.testing.assert_close(cos_scaled, cos_unscaled)
    torch.testing.assert_close(sin_scaled, sin_unscaled)


def test_initial_scaling_rejects_non_positive_value():
    with pytest.raises(ValueError):
        RotaryEmbedding(dim=8, theta=100000.0, initial_scaling=0.0)


def test_rotate_half_is_the_standard_construction():
    x = torch.arange(8.0).view(1, 1, 1, 8)
    x1, x2 = x.chunk(2, dim=-1)
    expected = torch.cat([-x2, x1], dim=-1)
    torch.testing.assert_close(rotate_half(x), expected)
