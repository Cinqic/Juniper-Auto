# Phase 2 GPT-5.6 Sol Independent Review

## Final decision

**APPROVED.** Phase 2 satisfies the sparse-routing, expert-behavior, MoE
kernel, diagnostic, reproducibility, repository, and consumer-hardware
requirements needed to proceed to Phase 3. Canonical approval is the
annotated `phase-2-moe` tag, created only after the approval-metadata
commit itself passes all three required remote workflows.

This approval does not claim trained expert specialization, superiority
over the dense control, or optimality of either auxiliary-loss coefficient.

## Identity

- Project: Juniper Auto 150M v0.1
- Phase: 2 — Sparse Routing, Expert Behavior, and MoE Kernel Validation
- Architecture: `ja150m-v0.1`
- Review date: 2026-08-28
- Reviewer / approval authority: GPT-5.6 Sol
- Original candidate: `20d08b35e8c0199511f37fa62739bc15e078962d`
- Substantive repair commit: `2d5a34f85c996bf0beededcb47629b567685b907`
- Final approved commit: resolved by annotated tag `phase-2-moe` to avoid
  impossible commit self-reference
- Approved Phase 1 base: annotated tag `phase-1-architecture`, commit
  `073acf46e04241ed35d00bc4b4c29ac463ee744d`

## Environment

| Field | Independent-review environment |
|---|---|
| OS | Linux Mint / Linux `7.0.0-30-generic`, x86_64 |
| Python | 3.12.3 |
| PyTorch | 2.13.0+cu130 |
| CUDA runtime | 13.0; available |
| GPU | NVIDIA GeForce RTX 2060, 6144 MiB |
| CPU target | Ryzen 7 5700G baseline |
| Architecture config | `configs/architecture/ja150m-v0.1.yaml` |
| Tokenizer | not yet created; Phase 3 scope |

The review started from a clean `main` tracking `origin/main`. No
`phase-2-moe` tag existed. Existing remote Phase 0/1/2 workflows were green
on the candidate, but that was treated only as a baseline claim to audit.

## Independent method

I read the model, routing, dispatch, loss, diagnostic, ablation, context
probe, configuration, training, checkpoint, experiment, manifest, CI,
recovery, time-accounting, and report paths; inspected commit/tag/remote
state; compared the reference path with the immutable Phase 1 tag; ran the
candidate validator; constructed adversarial padding/ablation/trace/
gradient cases; repaired discovered gaps with regression tests; generated
clean-commit evidence; reran the complete validators on FLOWBOX; exercised
a fresh clone; inspected the final diff; pushed the final state; and
verified remote CI before tagging approval.

## Defects found and repaired

| Finding | Severity | Repair and evidence |
|---|---|---|
| CI workflows used shallow checkout while the Phase 1 golden test silently skipped when its tag was absent | Approval-blocking | All Phase 0/1/2 workflows now fetch full history/tags; missing or moved `phase-1-architecture` is a hard failure in both the validator and golden test. The first repaired push exposed the same inherited checkout flaw in the Phase 0/1 workflows, which was then fixed and regression-tested. |
| “Evaluation-only” ablations executed while modules were in training mode | Approval-blocking | `JuniperAutoModel` and `MoELayer` now fail closed unless `eval()` is active; regression tests cover the boundary. |
| Invalid/unknown ablation modes and expert IDs could be ignored or fail later with an opaque index error | High | Strict mode-specific dataclass validation, range checks at the concrete layer, uniqueness checks, and tests. |
| Padding positions still executed shared and routed experts and produced expert contributions | High | Dispatch now compacts valid tokens, executes only those tokens, scatters back by original position, and returns zero MoE contribution for padding. Valid-token Phase 1 behavior remains equivalent; unpadded calls remain bit-exact. |
| The required diagnostic trace omitted weight-sum confirmation, shared activation, reconstruction position, token ID, routed assignment count, and selected-vs-executed identity | High | Trace schema and JSON export now include all fields; padding and ablation execution are explicit; the canonical demonstration covers all 15 MoE layers. |
| Expert gradient norms were claimed via ad-hoc `p.grad` inspection but no per-layer diagnostic collector existed | High | Added post-backward router/shared/per-routed-expert L2 telemetry and model-wide collection, with selected/unselected tests and canonical output. |
| Per-layer diagnostic objects exposed raw auxiliary losses but not their weighted contributions | Moderate | Diagnostics now expose raw and coefficient-weighted balance/Z values separately; tests prove each coefficient is applied once. |
| Context-sensitivity infrastructure lacked the required frozen lexical probe categories | Moderate | Added tokenizer-independent templates for ambiguity, domain/syntax shift, code/prose overlap, mathematical reuse, format syntax reuse, and position control, plus catalog validation. |
| Routing-health thresholds were hard-coded rather than caller-configurable | Moderate | Added validated `RoutingHealthThresholds` configuration while retaining documented defaults; oscillation now has an explicit classifier. |
| The reference/optimized artifact omitted mean error, routing-weight error, gradients, diverse shapes/padding, and combined performance/memory context | Moderate / evidence integrity | Exp-0022 records every required field and a strict rationale-backed `1e-5` tolerance. |
| The historical trace summary pointed to an ephemeral staging path and could not directly show the input | Moderate / artifact integrity | The new canonical demonstration records input IDs/mask, a repository-relative trace path, and representative valid/padding records. |

## Routing validation

- Frozen layout: 20 layers; dense at 1/5/10/15/20; 15 MoE layers elsewhere.
- Each MoE layer: 8 routed experts, one ungated always-active shared expert,
  top-2 unique token-choice routing, bias-free 512→8 router, 512-wide
  bias-free SwiGLU experts.
- Router projection and softmax remain FP32 under CPU BF16 and real RTX 2060
  FP16 autocast.
- Selected routed weights renormalize to one. There is no shared gate,
  hidden averaging, capacity factor, overflow path, or token dropping.
- Every valid token has exactly two executed routed assignments and one
  shared-expert activation; total valid assignment count is exactly
  `2 * valid_token_count` per layer.
- Padding receives zero expert assignments, no shared activation, and zero
  MoE contribution. Compact dispatch scatters valid outputs back to their
  exact original positions.
- The unpadded reference path remains bit-exact with the approved Phase 1
  implementation. Padded valid-token outputs use a `1e-6` regression bound
  because compacting changes GEMM batch shape and only last-bit accumulation.

## Reference vs optimized

Canonical exp-0022 uses 20 seeds; shapes `(1,1)`, `(1,17)`, `(2,32)`,
`(3,11)`, `(2,64)`; and no-padding, trailing, scattered, almost-all-padding,
and all-padding layouts on CPU FP32.

- Routing assignment agreement: `1.0`
- Maximum routing-weight difference: `0.0`
- Maximum output difference: `1.4901161193847656e-08`
- Mean absolute output difference: `1.779114286163173e-10`
- Maximum input/router gradient difference: `2.3283064365386963e-10`
- Accepted output/gradient tolerance: `1e-5` absolute and relative
- Raw balance and Z losses: identical between backends

The prior FLOWBOX profile remains valid because dispatch kernel math did
not change: optimized was 1.51×–2.64× faster across five measured shapes;
peak-VRAM differences were 81,408–606,720 bytes, under 1 MiB. The readable
reference path remains the default correctness oracle under ADR-0009.

## Auxiliary losses

The independently reviewed formulas are:

```text
f_e = valid top-2 assignments to e / (N * 2)
p_e = mean valid full-softmax probability for e
L_balance_layer = 8 * sum_e(f_e * p_e)
L_z_layer = mean_valid(logsumexp(router_logits)^2)
```

Layer values are averaged, then weighted once by `0.01` and `0.001`.
Padding is excluded. Hand-computed balanced/imbalanced, masking, finiteness,
and gradient tests pass. Raw and weighted values are separately visible.

## Diagnostics, pathology detection, and ablations

Per-layer diagnostics expose full probabilities, selected IDs/weights,
valid expert counts/load, full-distribution entropy, probability/logit
top-1/top-2 margins, 28-pair co-activation representation, shared/routed
contribution norms, logit magnitude, raw/weighted auxiliary losses, and
post-backward per-expert gradient norms. Summary diagnostics are opt-in;
bounded token-level trace is deeper opt-in.

Exp-0023 deterministically records 60 layer/token records for four input
positions over all 15 MoE layers: 45 valid records have two executed unique
experts, normalized weights, shared activation, and identity reconstruction;
15 padding records have zero executed experts/shared activation. Synthetic
dead, dominant, strongly skewed, low-entropy, large-logit, collapsed,
saturated, and oscillating cases all trigger the intended configurable
detectors while a uniform healthy case triggers none.

All six ablations are explicit, per-call, evaluation-only, range-validated,
diagnostically labeled, nonpersistent, and tested: routed disable, shared
disable, selected-expert replacement, deterministic uniform routing,
seeded random routing, and output zeroing. Traces distinguish selected from
actually executed/replaced experts.

## Context sensitivity

The low-level hidden-state tests distinguish identical, partially perturbed,
and strongly perturbed routing. The model-level harness measures the same
token ID at marked sequence positions per MoE layer. The frozen text-method
catalog covers semantic ambiguity (`bank`), same syntax across domains
(`class`), code/prose overlap (`return`), mathematical symbol reuse (`*`),
syntax reuse across prose/Python/JSON/math (`:`), and position control.

No tokenizer or trained checkpoint exists. These are methodology and router
input-sensitivity tests, not evidence of learned semantic specialization.

## Regression and validation

- Exact sparse parameters: **150,031,360**
- Exact sparse standard-active convention: **79,252,480**
- Exact dense control parameters: **79,191,040**
- Full local suite after repair: **654 passed** on FLOWBOX, including live
  CUDA tests; one known PyTorch warning states memory-efficient attention
  backward is nondeterministic.
- Phase 0, Phase 1, and Phase 2 canonical validators: pass.
- Serialization, checkpoint/resume, forward/backward, optimizer, tiny
  overfit, architecture, padding/causality, and parameter-count regressions:
  pass.
- Fresh-clone/fresh-venv recovery and final remote CI identities are recorded
  in `docs/recovery/phase-2.md` and the annotated `phase-2-moe` tag.

## Negative results and accepted limitations

- CUDA backward bitwise determinism is not claimed; the warning is preserved.
- The optimized dispatch is faster but remains opt-in/default-reference by
  ADR-0009.
- Context probes and detector thresholds are synthetic engineering baselines,
  not trained-model calibration.
- No tokenizer, real corpus, checkpoint, expert specialization, or semantic
  routing evidence exists yet.
- The original candidate's green CI did not execute its advertised golden
  comparison; this negative finding is preserved above rather than hidden.

## Time and Git/CI closure

Primary implementer time remains as recorded by Sonnet. Repair engineering,
independent review, CPU validation, and GPU validation are separate Phase 2
rows in `docs/time/phase-hours.csv`; no historical time was invented.

The approval report cannot contain the hash of the commit that contains
itself or CI run IDs created after that commit. Project precedent resolves
this honestly: the annotated `phase-2-moe` tag points to the exact final
metadata commit only after Phase 0, Phase 1, and Phase 2 workflows all pass;
its tag message records the final commit and exact workflow run IDs/results.

## Approval statement

Phase 2 is independently approved. Juniper Auto may proceed to Phase 3:
Unified Tokenizer and Control Vocabulary.
