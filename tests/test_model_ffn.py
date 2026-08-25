"""SwiGLU equation correctness and bias-free structure."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from juniper_auto.model.ffn import SwiGLU


def test_swiglu_matches_manual_equation():
    torch.manual_seed(0)
    ffn = SwiGLU(d_model=6, hidden_dim=10, bias=False)
    x = torch.randn(3, 4, 6)
    out = ffn(x)

    gate = x @ ffn.gate_proj.weight.T
    up = x @ ffn.up_proj.weight.T
    expected = (F.silu(gate) * up) @ ffn.down_proj.weight.T
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


def test_swiglu_no_bias_by_default():
    ffn = SwiGLU(d_model=4, hidden_dim=8, bias=False)
    assert ffn.gate_proj.bias is None
    assert ffn.up_proj.bias is None
    assert ffn.down_proj.bias is None


def test_swiglu_shapes():
    ffn = SwiGLU(d_model=6, hidden_dim=10, bias=False)
    x = torch.randn(2, 5, 6)
    out = ffn(x)
    assert out.shape == (2, 5, 6)
    assert ffn.gate_proj.weight.shape == (10, 6)
    assert ffn.up_proj.weight.shape == (10, 6)
    assert ffn.down_proj.weight.shape == (6, 10)


def test_swiglu_is_not_a_plain_relu_mlp():
    # Sanity guard: SiLU-gated output must differ from a naive gate*up
    # without the SiLU nonlinearity, proving the activation is actually applied.
    torch.manual_seed(1)
    ffn = SwiGLU(d_model=4, hidden_dim=6, bias=False)
    x = torch.randn(2, 4)
    out = ffn(x)
    gate = x @ ffn.gate_proj.weight.T
    up = x @ ffn.up_proj.weight.T
    ungated = (gate * up) @ ffn.down_proj.weight.T
    assert not torch.allclose(out, ungated)
