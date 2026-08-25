# Phase 0 Authoritative Requirements Traceability

Reviewer: GPT-5.6 Sol Medium

Review baseline: `ca49784e5da913c639d387cc37d672b947c91718`

Classification vocabulary: satisfied, partially satisfied, missing,
incorrectly implemented, incorrectly documented, not applicable yet,
accepted limitation.

This matrix compares semantics, not headings. “Final” describes the reviewed
Phase 0 candidate; executable evidence is identified in the last column.

| Requirement | Baseline | Final | Evidence / disposition |
|---|---|---|---|
| Primary research question | satisfied | satisfied | `project-charter.md` |
| All 14 secondary research questions | missing (8 rewritten questions) | satisfied | Exact ordered count and semantic regression tests |
| One-model philosophy | satisfied | satisfied | Charter; governance rule 1 |
| Sparse-vs-dense research requirement | satisfied | satisfied | Charter; ADR-0002; parameter tests |
| Consumer-hardware constraint | satisfied | satisfied | Charter; environment specification |
| Local-first objective | satisfied | satisfied | Charter; governance rule 30 |
| Open/modifiable and customization objectives | satisfied | satisfied | Charter; governance rules 31 and 33 |
| 6B effective-token target | satisfied | satisfied | Charter; frozen-artifact manifest |
| 100-300M post-training envelope | satisfied | satisfied | Charter; frozen-artifact manifest |
| Full v0.1 model capability scope | partially satisfied | satisfied | Charter enumerates all 17 targets; structural test |
| Full v0.1 runtime/system scope | partially satisfied | satisfied | Charter enumerates all 14 targets and separates scope from implementation |
| All 11 explicit non-goals | satisfied with weakened checkpoint wording | satisfied | Exact semantic non-goal test |
| All 40 permanent governance rules | missing (39) | satisfied | Ordered heading test; official-configuration rule restored as rule 33 |
| Phase references match authoritative 0-15 roadmap | incorrectly documented | satisfied | Versioned roadmap and stale-mapping regression test |
| Environment specification | satisfied | satisfied | `docs/architecture/environment-specification.md` |
| Hash-pinned dependency lock and declaration consistency | partially tested | satisfied | Canonical dependency-consistency validator gate; clean install |
| Structured strict configuration | partially satisfied | satisfied | Pydantic extra-forbid schema; cross-field and frozen-value tests |
| Architecture ID/version enforcement | partially satisfied | satisfied | ID pattern, known-ID kind rules, frozen load enforcement |
| Deterministic explicit seeds | satisfied | satisfied | Python/NumPy/Torch/CUDA seeding; CPU probe tests |
| Structured logging | partially satisfied | satisfied | Run/config/experiment-capable context; probe emits start/complete JSON |
| Experiment registry mandatory fields | satisfied | satisfied | Registry schema tests; honest not-yet-created identities |
| Time accounting | partially satisfied pending review | satisfied when review row is finalized | CSV schema/semantics gate; no fabricated Blessom time |
| ADR system and six foundation decisions | satisfied | satisfied | ADR-0001 through ADR-0006 |
| Router-jitter interpretation decision | incorrectly implemented | satisfied | ADR-0007; experiment-only policy, null magnitude, eval/inference false |
| Complete precision policy | partially satisfied | satisfied | Frozen versioned precision-policy document and regression test |
| Phase-report template | satisfied | satisfied | Report-structure tests |
| Recovery documentation | satisfied | satisfied | `docs/recovery/README.md` |
| Frozen artifact registry categories | satisfied | satisfied | Manifest test and strengthened validator gate |
| Architecture artifact hashes | satisfied before drift | satisfied after regeneration | SHA-256 code/manifest exact-set tests; no recursive self-hash |
| Repository validator | partially satisfied | satisfied | 11 fail-closed gates plus full pytest suite |
| Remote CI | satisfied for handoff only | pending exact reviewed commit | Workflow runs canonical validator in locked fresh venv |
| Fresh-clone recovery | satisfied for handoff only | pending exact reviewed commit | Final remote clone/venv exercise required before tagging |
| Review documentation and bookkeeping | incorrectly documented | satisfied | Independent-review report; phase report distinguishes all commits/roles |
| Final approval state and hashes | not applicable yet | pending exact CI/recovery | Canonical annotated tag resolves approved commit without self-reference |

## Frozen architecture checklist

| Group | Final classification | Independently checked invariants |
|---|---|---|
| Core/layers | satisfied | d=512; 20 layers; dense anchors 1/5/10/15/20; 15 complementary MoE layers |
| Attention | satisfied | causal full 4096 GQA; 8Q/2KV; head 64; per-head Q/K RMSNorm before RoPE; scale .125; no bias/window |
| Dense FFN | satisfied | bias-free SwiGLU/SiLU, 1536, expansion 3.0 |
| MoE | satisfied | 8 routed + 1 shared; top-2; always-active ungated shared path; bias-free 512 SwiGLU experts; FP32 router; token-choice dropless; renormalized routed weights; summed outputs; .01/.001 losses |
| Router jitter | satisfied | training experiment-only and unset; evaluation/inference disabled |
| Normalization | satisfied | pre-RMSNorm, epsilon 1e-5, FP32 reductions, attention/FFN/final norms, no LayerNorm bias |
| Position | satisfied | RoPE theta 100000, no scaling, 100%/64 dims, 4096 current; 16384 future only |
| Residual | satisfied | additive scale 1; no ReZero, DeepNorm, or learned gates |
| Embedding/output | satisfied | learned 36864x512; tied head; no bias; scale 1; no softcap |
| Dropout | satisfied | all paths zero |
| Initialization | satisfied | zero-mean normal .02; router/embedding .02; residual output projections .02/sqrt(40) |
| Dense control | satisfied | all 20 layers dense; shared attention/normalization/position/init/precision policy |
| Precision | satisfied | complete training/inference baseline in versioned policy |

## Completion-gate disposition

Local tests, validator, hashes, integrity, and independent arithmetic must all
pass before the reviewed commit is pushed. Exact-commit remote CI and a final
fresh remote-clone recovery must then pass before the approval tag can be
created. A tag is forbidden while either item remains pending.
