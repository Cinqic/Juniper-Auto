# Juniper Auto v0.1 Reference Model Implementation

Status: Phase 1 executable reference implementation of the frozen
`ja150m-v0.1` / `ja150m-v0.1-dense` architectures. This document describes
what the code in `juniper_auto/model/` and `juniper_auto/training/`
actually does; it does not restate the frozen numeric values themselves
(those live in `configs/architecture/*.yaml` and
`juniper_auto/config/frozen.py`) and it does not claim any language
capability -- see `docs/phases/phase-1-architecture.md` for what has and
has not been demonstrated.

## Module map

| Module | Responsibility |
|---|---|
| `juniper_auto/model/norm.py` | `RMSNorm`: FP32-reduced RMS normalization, reused unmodified as per-head QK-Norm. |
| `juniper_auto/model/rope.py` | `RotaryEmbedding` + `apply_rotary_pos_emb`: rotary positional encoding. |
| `juniper_auto/model/attention.py` | `GroupedQueryAttention`, `build_attention_mask`, `repeat_kv`: causal GQA with a unified causal+padding mask. |
| `juniper_auto/model/ffn.py` | `SwiGLU`: shared bias-free gated FFN, used for the dense-anchor FFN and every MoE expert. |
| `juniper_auto/model/moe.py` | `MoELayer`, `MoEDiagnostics`: router + dropless top-2 dispatch + shared expert. |
| `juniper_auto/model/losses.py` | `causal_lm_loss`, `compute_load_balance_loss_raw`, `compute_router_z_loss_raw`. |
| `juniper_auto/model/block.py` | `DenseBlock`, `MoEBlock`: Pre-Norm transformer blocks. |
| `juniper_auto/model/model.py` | `JuniperAutoModel`, `ModelOutput`, `build_model`, `initialize_weights`: the full model and its public construction path. |
| `juniper_auto/model/inspection.py` | Independent (Method B) parameter accounting and structural audits over the instantiated module tree. |
| `juniper_auto/training/state.py` | RNG state capture/restore; `SyntheticSequenceStream` (deterministic, checkpointable synthetic training data). |
| `juniper_auto/training/checkpoint.py` | Versioned checkpoint build/save/load/validate. |
| `juniper_auto/training/tiny_overfit.py` | `TinyOverfitHarness`, `run_tiny_overfit`: end-to-end training loop used for the tiny-batch overfit gates and the checkpoint/resume experiment. |
| `juniper_auto/training/profiling.py` | FLOWBOX inference/training-step/checkpoint-I/O profiling utilities. |

## Construction path

```
ArchitectureConfig (juniper_auto.config)
        |
juniper_auto.model.build_model(cfg, device=..., dtype=..., seed=...)
        |
JuniperAutoModel(cfg)              -- constructs modules, always on CPU first
        |
initialize_weights(model, cfg, generator)  -- frozen init policy, deterministic given `seed`
        |
.to(device).to(dtype)              -- moved last, so init is reproducible independent of CUDA
```

`build_model` is the only supported entry point; `JuniperAutoModel`'s
`__init__` alone leaves weights at PyTorch's default (soon-to-be-overwritten)
init, not the frozen policy.

## Shared infrastructure vs. per-layer choice

One `JuniperAutoModel` class serves both architectures. Per layer `i` in
`1..core.n_layers`, the model looks up `i` in `core.dense_layers` /
`core.moe_layers` and instantiates a `DenseBlock` or `MoEBlock`
accordingly. For `ja150m-v0.1-dense`, `moe_layers` is empty, so every
block is a `DenseBlock` -- the dense control exercises exactly the same
embedding, attention, RoPE, QK-Norm, normalization, and initialization
code as the sparse model, differing only in which FFN each block uses.
`FoundationProbe` (Phase 0) is unrelated to this model and is not reused
or extended by it.

## RMSNorm / QK-Norm (`norm.py`)

```
input_dtype = x.dtype
x32 = x.to(float32)
variance = mean(x32 ** 2, dim=-1, keepdim=True)
x_normed = x32 * rsqrt(variance + eps)
output = (weight * x_normed.to(input_dtype)).to(input_dtype)
```

The reduction (`variance`, `rsqrt`) always runs in FP32 regardless of the
input activation dtype; the result is explicitly re-cast to the input
dtype both before *and* after the learnable-scale multiply, so the
module's output dtype always matches its input dtype exactly (this is
stricter than some reference RMSNorm implementations, which let an FP32
weight silently widen an FP16 activation via dtype promotion).

QK-Norm reuses this exact module at `head_dim` (64 for the frozen
architectures) applied to a `[..., n_heads, head_dim]` tensor: the
reduction is over the last axis only, so each head is normalized
independently, while the `[head_dim]`-shaped scale parameter broadcasts
across the head axis and is therefore shared by all heads of that
projection -- this is what keeps QK-Norm at `2 * head_dim` parameters per
layer (2,560 total for the sparse/dense architectures) rather than
`2 * n_heads * head_dim`.

## RoPE (`rope.py`)

Standard rotate-half construction: `inv_freq` is a non-persistent buffer
(moves with `.to(device)`, never checkpointed, since it is a deterministic
function of `dim`/`theta`); `cos`/`sin` are recomputed from `position_ids`
on every forward call (no cache keyed by sequence length, so there is no
stale-cache-after-`.to()` failure mode). Supports a `rotary_dim < head_dim`
partial-rotation mode for future test configs, even though the frozen
architectures use `rotary_dim == head_dim` (full rotation).

## Grouped-query attention (`attention.py`)

Projections: `q_proj: d_model -> n_query_heads*head_dim`,
`k_proj`/`v_proj: d_model -> n_kv_heads*head_dim`, `o_proj:
n_query_heads*head_dim -> d_model`, all bias-free. QK-Norm (if enabled)
is applied immediately after projection/reshape, before RoPE. KV heads
are repeated contiguously (`repeat_kv`) so KV head `h` serves query heads
`[h*n_rep, (h+1)*n_rep)`.

Masking is always an explicit boolean `[batch_or_1, 1, seq, seq]` tensor
built by `build_attention_mask`, combining strict causal masking with any
key-side padding mask, passed to
`torch.nn.functional.scaled_dot_product_attention` with `is_causal=False`
-- never SDPA's internal causal-mask shortcut, so causal and padding
masking share one directly testable code path. A query row that ends up
with zero allowed keys (a padded position whose own key is also masked)
gets a forced self-attend fallback so softmax cannot divide by zero; the
output at that row is discarded downstream (no loss, no cross-token
effect) so only its finiteness matters.

## SwiGLU (`ffn.py`)

`down_proj(SiLU(gate_proj(x)) * up_proj(x))`, all three projections
bias-free. One class serves the dense-anchor FFN (`dense_ffn.dim`) and
every MoE expert -- routed and shared alike (`moe.expert_ffn_dim`).

## MoE (`moe.py`)

Router: `Linear(router_input_dim, router_output_dim, bias=router_bias)`.
Router logits and softmax are computed inside a
`torch.autocast(..., enabled=False)` block with explicit `.to(float32)`
casts on both the input and the weight -- this forces FP32 regardless of
an enclosing autocast context, verified directly (not just documented)
under both FP16 CUDA autocast and CPU bfloat16 autocast in
`tests/test_model_moe.py`.

Dispatch is a correctness-first reference loop: for each of the
`n_routed_experts` experts and each of the `top_k` selection slots, the
subset of tokens whose slot selected that expert is gathered, run through
that expert, weighted by its (renormalized) selection probability, and
scattered back via `Tensor.index_add` (differentiable, out-of-place) into
an accumulator that starts as the always-active shared expert's output.
No token is ever dropped, there is no capacity factor, and there is no
division by `top_k + 1` or any other averaging factor -- the combination
is exactly `shared(x) + sum_i(weight_i * routed_i(x))` with
`sum_i(weight_i) == 1`. `MoEDiagnostics` (opt-in via
`return_diagnostics=True`) exposes router logits/probs, selected expert
indices/weights, and per-expert valid-token assignment counts for testing
and for future routing analysis; it is never populated on an ordinary
forward pass, so it costs nothing when unused.

A combination-weight dtype subtlety: under autocast, an expert's Linear
layers may execute in a lower-precision dtype than the input activation
itself was in going in (autocast decides per-op). The combination weight
is therefore cast to each expert's *actual output* dtype at the point of
multiplication, not pre-cast once to the input's dtype -- an earlier draft
of this code pre-cast once and failed under CPU bfloat16 autocast with a
dtype mismatch in `index_add_`; see `tests/test_model_moe.py`'s autocast
tests, which exist specifically because this class of bug is silent until
exercised under a real mixed-precision context.

Padding tokens are still routed and produce a finite output (MoE has no
cross-token mixing, so this cannot corrupt other tokens' outputs), but are
excluded from both auxiliary-loss statistics and diagnostic assignment
counts via `valid_mask`.

## Losses (`losses.py`, `docs/adr/0008`)

`causal_lm_loss` shifts logits/labels by one position and computes FP32
cross-entropy with `ignore_index=-100`. The load-balance and router
Z-loss formulas are defined in ADR-0008; both are computed over valid
tokens only and averaged across the 15 MoE layers before the frozen
coefficients (0.01, 0.001) are applied, in `JuniperAutoModel.forward`.

## Blocks and the full model (`block.py`, `model.py`)

Pre-Norm residual blocks: `x = x + Attn(Norm(x)); x = x + FFN_or_MoE(Norm(x))`,
residual scale 1.0, no ReZero/DeepNorm/learned gates. `JuniperAutoModel`
ties `lm_head.weight` to `embedding.weight` by assigning the same
`nn.Parameter` object (verified by `is` identity in
`tests/test_model_official_architecture.py`, and confirmed to survive a
`state_dict` save/load round trip in `tests/test_model_serialization.py`).
`ModelOutput` always returns `load_balance_loss`/`router_z_loss` (weighted)
and their raw (unweighted, layer-averaged) counterparts; for the dense
control these are always exactly `0.0` (the model's per-instance
coefficients are `0.0` when `cfg.moe is None`), so a training loop can
treat both architectures' outputs uniformly without a dense/sparse branch.

Optional per-block activation (gradient) checkpointing
(`model.set_gradient_checkpointing(True)`) is off by default, only takes
effect in `model.training` mode, and is verified to reproduce identical
forward output and gradients to the non-checkpointed path
(`tests/test_model_activation_checkpointing.py`). It is incompatible with
`return_diagnostics=True` (raises `ValueError`) since per-layer MoE
diagnostics would otherwise be silently recomputed and discarded.

## Initialization (`model.py::initialize_weights`)

Applied after construction, in a fixed module-traversal order, so a given
`torch.Generator` seed always produces the same final weights regardless
of what PyTorch's default `nn.Linear`/`nn.Embedding` init happened to
sample first (that default init is entirely overwritten, never blended
in):

- Q/K/V projections, FFN/expert gate & up projections: `N(0, base_std^2)`.
- Attention `o_proj`, every FFN/expert `down_proj` (dense, routed, shared):
  `N(0, residual_output_projection_std^2)` -- the smaller scaled-residual
  init.
- Router: `N(0, router_std^2)`.
- Embedding (and, via tying, the LM head): `N(0, embedding_std^2)`.
- RMSNorm/QK-Norm scale vectors: left at the ones set by `RMSNorm.__init__`
  (not touched by `initialize_weights`).
- No bias parameters exist anywhere for the frozen v0.1 configs, so there
  is no bias-init branch.

## Training support

`SyntheticSequenceStream` (`juniper_auto/training/state.py`) is Phase 1's
only training data source: a fixed pool of deterministic random-token-id
sequences, served in shuffled batches from an independently checkpointable
generator. It is always labeled `synthetic Phase 1 engineering data`
wherever it appears (checkpoints, experiment registry) -- there is no
tokenizer or real corpus yet.

The checkpoint format (`juniper_auto/training/checkpoint.py`) bundles
model/optimizer/scheduler/GradScaler state, every RNG stream (Python,
NumPy, torch CPU, all CUDA devices), the sampler state, step counters,
architecture identity + full config, git commit, and dataset/tokenizer
identity, written atomically (temp file + `os.replace`) and validated
(missing field / wrong format version / wrong architecture id) before any
state is applied. See `docs/phases/phase-1-architecture.md` for the
executed checkpoint/resume equivalence result.
