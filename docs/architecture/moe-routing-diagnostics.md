# Juniper Auto Phase 2 MoE Routing, Diagnostics, and Ablation Infrastructure

Status: Phase 2 executable infrastructure built on top of the frozen
`ja150m-v0.1` MoE architecture and the Phase 1 reference `MoELayer`. This
document describes what the code in `juniper_auto/model/moe*.py` and
`juniper_auto/analysis/` actually does. It does not claim any trained
model behavior -- see `docs/phases/phase-2-moe.md` for what has and has
not been demonstrated, and section "What this is not" below.

## Module map

| Module | Responsibility |
|---|---|
| `juniper_auto/model/routing.py` | Pure router math (FP32 logits/softmax, top-k selection, renormalization), shared by every dispatch backend. |
| `juniper_auto/model/moe_dispatch.py` | `reference_dispatch` (Phase 1, correctness-first, preserved bit-exact for the default no-ablation call) and `optimized_dispatch` (Phase 2, sort-and-group pure PyTorch). |
| `juniper_auto/model/moe_ablations.py` | `MoEAblationConfig` and the evaluation-only override resolvers consumed by `MoELayer.forward`. |
| `juniper_auto/model/moe_diagnostics.py` | `MoEDiagnostics`, per-token trace records, `RoutingWindowAccumulator`, and the routing-health detectors. |
| `juniper_auto/model/moe.py` | `MoELayer`: orchestrates the above, owns router/expert parameters and frozen-config validation. |
| `juniper_auto/analysis/context_sensitivity.py` | Router-decision comparison metrics and the `ProbeCase`/`run_probe_case` model-level harness. |

## Forward-call data flow

```
MoELayer.forward(x, valid_mask, return_diagnostics, backend, ablation, return_trace, max_trace_tokens)
        |
routing.compute_router_logits_and_probs   -- FP32 forced regardless of ambient autocast
        |
ablation.is_router_override? --yes--> ablation.resolve_router_override (uniform/random)
        |no
routing.select_topk                        -- top-k by probability, optional renormalization
        |
shared_expert(flat_x) --(ablation.disable_shared_expert? zero it)-->
        |
moe_dispatch.DISPATCH_BACKENDS[backend](..., initial_output=shared_out, **ablation.resolve_dispatch_kwargs)
        |
output = dispatch result (shared + weighted routed contributions)
        |
losses.compute_load_balance_loss_raw / compute_router_z_loss_raw   -- valid tokens only
        |
return_diagnostics? --yes--> moe_diagnostics.build_moe_diagnostics (entropy, margins, pair
        |                    co-activation, contribution norms, logit-magnitude stats, optional
        |                    bounded per-token trace)
        v
(output, load_balance_loss_raw, router_z_loss_raw, diagnostics_or_None)
```

`MoEBlock.forward` and `JuniperAutoModel.forward` thread `return_trace`,
`max_trace_tokens`, and `ablation` through unchanged; a single `ablation`
config, when given at the model level, applies identically to every MoE
layer in that forward call. `juniper_auto.model.moe_diagnostics.assemble_full_trace(model.layer_kinds, out.diagnostics)`
reassigns each layer's trace records to the model's 1-indexed layer
position and concatenates them into one full-model trace.

## Reference vs. optimized dispatch

Both backends compute identical routing (they call the same
`routing.py` functions) and differ only in *how* the weighted routed
contributions are accumulated into the output:

- `reference_dispatch`: a Python loop over `(expert, slot)` pairs
  (`n_routed_experts * top_k` iterations), each doing one boolean `==`
  compare over the full token dimension, one gather, one matmul, one
  `index_add`. Preserved bit-for-bit identical to the approved Phase 1
  `moe.py` for the default no-ablation call -- proven directly against a
  copy of `juniper_auto/model/moe.py` loaded from the `phase-1-architecture`
  git tag in `tests/test_model_moe_dispatch.py`.
- `optimized_dispatch`: one global `argsort` of the flattened
  `(token, slot)` assignments by expert id, then one gather + matmul +
  `index_add` per *expert* (not per `(expert, slot)` pair).

Numerical equivalence (not bit-exact -- summation order differs) is
proven within 1e-5 absolute/relative tolerance on CPU (forward and
backward) and within a documented looser tolerance under real CUDA FP16
autocast, in `tests/test_model_moe_dispatch.py` and experiment `exp-0015`.
FLOWBOX RTX 2060 throughput measurements (`exp-0021`) found `optimized`
1.5x-2.6x faster at every profiled shape with negligible VRAM difference.
Per `docs/adr/0009-moe-dispatch-backend-selection.md`, `MoELayer`'s
default remains `backend="reference"`; `optimized` is fully available via
`backend="optimized"`.

## Diagnostics

`MoEDiagnostics` adds, computed only when `return_diagnostics=True` and always
restricted to valid (non-padding) tokens for aggregate statistics:

- `entropy` / `normalized_entropy` -- full router-distribution entropy per
  token, and `entropy / log(n_experts)`.
- `top1_top2_prob_margin` / `top1_top2_logit_margin` -- per token.
- `router_logit_abs_mean` / `_rms` / `_abs_max` -- scalar, valid tokens only.
- `expert_pair_coactivation` -- `[n_experts, n_experts]`, strictly
  upper-triangular (no double counting).
- `shared_contribution_norm_mean` / `_rms`, `routed_contribution_norm_mean`
  / `_rms`, `routed_shared_norm_ratio`.
- `load_balance_loss_raw`, `router_z_loss_raw`, plus the separately named
  coefficient-weighted values -- carried through without conflation.
- `ablation_mode` -- `None` on the production path, otherwise the explicit
  evaluation-only override label.
- `token_trace` -- only when `return_trace=True`; a bounded (default
  `max_trace_tokens=4096`, override or pass `None` to disable the bound)
  list of `TokenRoutingTraceRecord` (layer, batch/seq/flat/reconstruction
  position, validity, token ID when supplied, selected and actually executed
  experts, weights and their sum/normalization result, routed assignment
  count, and shared-expert activation). Padding records explicitly show no
  executed experts or shared activation.

Expert gradients do not exist during forward diagnostic construction.
After `backward()`, `collect_layer_expert_gradient_norms` and
`collect_model_expert_gradient_norms` expose router, shared-expert, and each
routed expert's L2 norm per MoE layer. `None` distinguishes a module that
received no gradient from a participating module whose norm is exactly zero.

`RoutingWindowAccumulator` aggregates many `MoEDiagnostics` instances
(multiple batches) into window-level load shares and mean entropy/margin/
logit statistics -- the unit the routing-health detectors are meant to be
applied to, not a single batch. Detector thresholds
(`DEAD_EXPERT_LOAD_SHARE_THRESHOLD`, `STARVED_EXPERT_LOAD_SHARE_RATIO`,
`DOMINANT_EXPERT_LOAD_SHARE_RATIO`, `COLLAPSE_NORMALIZED_ENTROPY_THRESHOLD`,
`COLLAPSE_TOP_EXPERT_LOAD_SHARE_THRESHOLD`,
`SATURATION_LOGIT_ABS_MEAN_THRESHOLD`, `SATURATION_TOP1_MARGIN_THRESHOLD`,
`OSCILLATION_TOP1_CHANGE_RATE_THRESHOLD`) live in `moe_diagnostics.py`,
not in the frozen architecture config, since they are analysis/evaluation
configuration rather than an architectural property. Callers override them
through validated `RoutingHealthThresholds`; module constants are retained
as documented default aliases, not immutable research truths. Each has a
synthetic-case validation test in `tests/test_model_moe_diagnostics.py`
and a canonical validation run in experiment `exp-0018`.

## Ablations

Evaluation-only, applied via `MoELayer.forward(..., ablation=MoEAblationConfig(...))`
or `JuniperAutoModel.forward(..., ablation=...)` (applies identically to
every MoE layer). `ablation=None` (the default) takes the pre-Phase-2
dispatch path exactly -- no ablation state can leak into a normal call, and
passing an ablation while the model/layer is in training mode raises rather
than silently altering training. Unknown modes, irrelevant fields, duplicate,
negative, or out-of-range expert IDs also fail before dispatch. This is tested
directly (`tests/test_model_moe_ablations.py::test_ablation_state_does_not_persist_across_calls`,
experiment `exp-0019`). Exact per-mode semantics (frozen, documented in
full in `juniper_auto/model/moe_ablations.py`'s module docstring):
`disable_routed_expert`, `zero_expert_output`, `disable_shared_expert`,
`replace_routed_expert`, `uniform_router` (deterministic round-robin, no
`torch.topk` tie-breaking involved), `random_router` (seeded, isolated
`torch.Generator`, reproducible independent of ambient RNG state and
device).

## Context-sensitivity probe harness

`juniper_auto/analysis/context_sensitivity.py` measures *whether a
routing decision changed between two hidden-state contexts* -- it cannot
and does not measure learned semantic specialization, since Juniper Auto
has no trained checkpoint, no frozen tokenizer, and no real corpus.
`compare_routing_across_variants` computes `top1_change_rate`,
`exact_topk_change_rate` (aliased as `pair_change_rate`, matching the
Phase 2 instructions' wording -- "pair" here means a pair of *context
variants*, not a pair of co-activated experts), `mean_entropy_difference`,
and `mean_js_divergence` across every pair of variants. `ProbeCase` /
`run_probe_case` batch several full-sequence variants of one probe token
through a real `JuniperAutoModel` forward pass and compare routing at the
marked position, per MoE layer -- the same interface a later phase can
feed real tokenizer-backed semantic probe sets into unchanged.
`run_untrained_official_model_probe` is a labeled convenience wrapper
around the official untrained architecture, used for `exp-0017`; its
docstring and every result it produces are explicit that this is an
engineering/proxy smoke test, not specialization evidence.

The tokenizer-independent `CANONICAL_CONTEXT_PROBE_TEMPLATES` catalog freezes
the required later tokenizer-backed methodology before results exist:
semantic ambiguity, same syntax across domains, code/prose lexical overlap,
mathematical-symbol reuse, syntax reuse across prose/code/JSON/math, and
positional control. `validate_context_probe_templates` ensures every category
retains at least two contexts containing the exact same probe text.

## Padding execution contract

Router probabilities may be computed for all flattened positions so a deep
trace can show padding explicitly, but expert execution is compacted to the
valid mask. Each valid token executes exactly two routed experts plus the
shared expert. Padding executes neither branch, contributes zero expert load
and zero MoE output, and is scattered back as a zero contribution at its
original position. Unpadded calls retain the Phase 1 operation order exactly;
compacted padded calls may differ from the old valid-token result only at
last-bit FP32 GEMM accumulation (`1e-6` regression bound).

## What this is not

- Not evidence of learned expert specialization, semantic routing, or
  context-aware routing -- no training has occurred.
- Not a claim that any routing-health detector has fired meaningfully on
  a real model -- `exp-0018` validates the detectors against synthetic
  cases only.
- Not a tokenizer, evaluation suite, or pretraining component -- see
  `docs/phases/phase-2-moe.md`'s accepted limitations for the full list.
- Not a claim of bitwise-identical CUDA determinism beyond what Phase 1
  already established; Phase 2 does not broaden that claim.
