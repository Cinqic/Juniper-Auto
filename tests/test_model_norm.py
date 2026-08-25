"""RMSNorm and per-head QK-Norm math correctness."""

from __future__ import annotations

import torch

from juniper_auto.model.norm import RMSNorm


def _manual_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x64 = x.double()
    variance = x64.pow(2).mean(dim=-1, keepdim=True)
    normed = x64 / torch.sqrt(variance + eps)
    return weight.double() * normed


def test_rmsnorm_matches_independent_reference():
    torch.manual_seed(0)
    norm = RMSNorm(16, eps=1e-5)
    norm.weight.data.uniform_(0.5, 1.5)
    x = torch.randn(3, 5, 16)
    out = norm(x)
    expected = _manual_rmsnorm(x, norm.weight, 1e-5)
    torch.testing.assert_close(out.double(), expected, atol=1e-5, rtol=1e-5)


def test_rmsnorm_reduction_axis_is_last_dim_only():
    norm = RMSNorm(8)
    x = torch.zeros(2, 3, 8)
    x[0, 0, :] = torch.arange(8.0)
    x[0, 1, :] = 1.0  # constant vector -> RMS = 1, normed output = constant / rms
    out = norm(x)
    # Row [0,1] has all-equal features, so normalization must leave it uniform.
    assert torch.allclose(out[0, 1, :], out[0, 1, 0].expand(8), atol=1e-5)
    # Row [0,0] and row [0,1] must differ in the output since their raw
    # magnitudes differ -- proves normalization is not a no-op.
    assert not torch.allclose(out[0, 0, :], out[0, 1, :])


def test_rmsnorm_weight_ones_at_init_is_identity_scale():
    norm = RMSNorm(12)
    assert torch.equal(norm.weight.data, torch.ones(12))


def test_rmsnorm_scale_is_learnable_and_no_bias():
    norm = RMSNorm(4)
    params = dict(norm.named_parameters())
    assert set(params.keys()) == {"weight"}
    assert params["weight"].requires_grad


def test_rmsnorm_fp32_reduction_under_fp16_input():
    # Reduction must happen in FP32 even when the input activation is FP16 --
    # this guards against an implementation that reduces directly in FP16
    # and silently loses precision on values that would overflow/underflow
    # FP16 mean-of-squares.
    norm = RMSNorm(8)
    x = (torch.randn(4, 8) * 300).to(torch.float16)  # large enough that FP16 squared-mean is lossy
    out = norm(x)
    assert out.dtype == torch.float16
    assert torch.isfinite(out).all()

    expected = _manual_rmsnorm(x, norm.weight, 1e-5)
    torch.testing.assert_close(out.double(), expected, atol=2e-2, rtol=2e-2)


def test_per_head_qk_norm_independent_per_head_shared_scale():
    head_dim = 8
    norm = RMSNorm(head_dim)
    norm.weight.data.uniform_(0.5, 1.5)
    torch.manual_seed(1)
    x = torch.randn(2, 5, 3, head_dim)  # [batch, seq, n_heads, head_dim]

    out = norm(x)

    # Each head normalized independently: compare against per-head slices
    # run through the same module one head at a time.
    for h in range(3):
        expected_h = norm(x[:, :, h, :])
        torch.testing.assert_close(out[:, :, h, :], expected_h)

    # Exactly `head_dim` learnable parameters, shared across all heads --
    # this is what keeps the frozen architecture's QK-Norm accounting at
    # head_dim (not n_heads * head_dim) parameters per projection.
    assert norm.weight.numel() == head_dim
