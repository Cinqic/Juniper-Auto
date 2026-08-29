# Phase 2 Report: Sparse Routing, Expert Behavior, and MoE Kernel Validation

## Phase

Phase 2: Sparse Routing, Expert Behavior, and MoE Kernel Validation.

## Objective

Prove that Juniper Auto's sparse MoE machinery is mathematically correct,
dropless, reproducible, inspectable, numerically stable, fully traceable
per token, appropriately instrumented, ablatable, context-sensitivity
measurable, testable against deliberate failures, and ready for later
training phases -- without pretraining, tokenizer engineering, or premature
expert-specialization claims.

## Starting commit

`073acf46e04241ed35d00bc4b4c29ac463ee744d` (approved Phase 1 candidate,
tag `phase-1-architecture`).

## Final commit

The Sonnet substantive candidate is `63b05c5b9ec1a3eec21bf129c99b4c48d0fd0407`,
pushed to `origin/main` and confirmed present remotely
(`git ls-remote origin main` matched exactly). All three required GitHub
Actions workflows ran on this exact commit and succeeded:

| Workflow | Run ID | Conclusion | Duration |
|---|---|---|---|
| Phase 0 Validation | [33010601469](https://github.com/Cinqic/Juniper-Auto/actions/runs/33010601469) | success | 2m4s |
| Phase 1 Validation | [33010601512](https://github.com/Cinqic/Juniper-Auto/actions/runs/33010601512) | success | 2m45s |
| Phase 2 Validation | [33010601464](https://github.com/Cinqic/Juniper-Auto/actions/runs/33010601464) | success | 4m9s |

This report's own metadata-closure commit (recording the table above) is,
by construction, a commit whose hash could not have been known when it was
written -- resolved by the annotated `phase-2-moe` tag once GPT-5.6 Sol
performs independent review and approves, per the Phase 2 instructions
section 40/41 (Sonnet does not create that tag).

GPT-5.6 Sol independently reviewed this candidate on 2026-08-28, found and
repaired approval-blocking CI, padding, ablation, traceability, gradient-
telemetry, context-methodology, and evidence gaps. The substantive repair
commit is `2d5a34f85c996bf0beededcb47629b567685b907`; the exact final approved
metadata commit and final CI identities are resolved by the annotated
`phase-2-moe` tag. Full findings and evidence are in
[`phase-2-sol-independent-review.md`](phase-2-sol-independent-review.md).

## Implementation summary

- [`juniper_auto/model/routing.py`](../../juniper_auto/model/routing.py) --
  pure router math (FP32 logits/softmax, top-k selection, renormalization),
  extracted unmodified from the Phase 1 `MoELayer.forward` body and shared
  by every dispatch backend.
- [`juniper_auto/model/moe_dispatch.py`](../../juniper_auto/model/moe_dispatch.py) --
  `reference_dispatch` (Phase 1's correctness-first loop, preserved
  bit-for-bit for the default no-ablation call) and `optimized_dispatch`
  (Phase 2's pure-PyTorch sort-and-group backend).
- [`juniper_auto/model/moe_ablations.py`](../../juniper_auto/model/moe_ablations.py) --
  `MoEAblationConfig` and the six evaluation-only override modes, with
  frozen, exactly-decided semantics documented in the module docstring.
- [`juniper_auto/model/moe_diagnostics.py`](../../juniper_auto/model/moe_diagnostics.py) --
  extended `MoEDiagnostics`, per-token trace records + JSON export,
  `RoutingWindowAccumulator`, and the routing-health detectors.
- [`juniper_auto/model/moe.py`](../../juniper_auto/model/moe.py) --
  refactored into the orchestrator that wires the above together; existing
  Phase 1 tests (`tests/test_model_moe.py`, `tests/test_model_fault_injection.py`)
  pass with explicit padding-contract regression updates.
- [`juniper_auto/model/block.py`](../../juniper_auto/model/block.py),
  [`juniper_auto/model/model.py`](../../juniper_auto/model/model.py) --
  `return_trace`, `max_trace_tokens`, and `ablation` threaded through
  `MoEBlock`/`JuniperAutoModel.forward`.
- [`juniper_auto/analysis/context_sensitivity.py`](../../juniper_auto/analysis/context_sensitivity.py) --
  router-decision comparison metrics and the `ProbeCase`/`run_probe_case`
  model-level harness, plus a labeled untrained-model proxy baseline.
- [`scripts/run_phase2_experiment.py`](../../scripts/run_phase2_experiment.py),
  [`scripts/validate_phase2.py`](../../scripts/validate_phase2.py),
  [`scripts/generate_phase2_test_manifest.py`](../../scripts/generate_phase2_test_manifest.py) --
  canonical Phase 2 experiment runner, validator, and test manifest
  generator, mirroring Phase 1's conventions.
- Tests: `tests/test_model_moe_dispatch.py`, `tests/test_model_moe_property.py`,
  `tests/test_model_moe_diagnostics.py`, `tests/test_model_moe_ablations.py`,
  `tests/test_model_moe_trace.py`, `tests/test_context_sensitivity.py`, and
  20 new fault-injection tests appended to `tests/test_model_fault_injection.py`.
- Docs: this report, `docs/phases/phase-2-requirements-traceability.md`,
  `docs/phases/phase-2-sonnet-self-review.md`, `docs/recovery/phase-2.md`,
  `docs/architecture/moe-routing-diagnostics.md`,
  `docs/adr/0009-moe-dispatch-backend-selection.md`.

## Architecture / configuration IDs

`ja150m-v0.1` (sparse) and `ja150m-v0.1-dense` (dense control) --
**unchanged**. Phase 2 adds no new architecture config and changes no
frozen numeric value; verified by `gate_frozen_architecture_unchanged` in
`scripts/validate_phase2.py` and the parameter counts below.

## Environment

Same as Phase 1 (`docs/architecture/environment-specification.md`):
Python 3.12, `torch==2.13.0+cu130`, FLOWBOX RTX 2060. No new dependency was
added; `triton==3.7.1` (already present via the torch lock, per ADR-0004)
was considered and explicitly not used -- `optimized_dispatch` is plain
PyTorch, per Phase 2 instructions section 5's preference.

## Artifacts

All new/changed source, docs, and scripts listed above are `status: frozen`
in `manifests/frozen-artifacts.yaml` under the new
`moe_routing_and_diagnostics_infrastructure` entry, hash-tracked in
`manifests/phase-2-artifact-hashes.yaml`. The frozen `ja150m-v0.1` /
`ja150m-v0.1-dense` architecture entries are unchanged.

## Hashes

`manifests/phase-2-artifact-hashes.yaml` (generated by
`scripts/hash_manifest.py --phase 2`) and
`manifests/phase-2-test-manifest.yaml` (generated by
`scripts/generate_phase2_test_manifest.py`). `manifests/phase-1-artifact-hashes.yaml`
was regenerated at the end of this phase's engineering to reflect the
current content of the few files it tracks that legitimately continued to
evolve (`docs/time/phase-hours.csv`, `experiments/registry.yaml`) or that
Phase 2 explicitly refactored under a proven-equivalent contract
(`juniper_auto/model/moe.py`) -- see
`docs/adr/0009-moe-dispatch-backend-selection.md` and the note at the top
of `juniper_auto/util/hashing.py`'s `PHASE_2_HASHED_ARTIFACTS` list for why
this does not weaken Phase 1's actual evidence (the immutable
`phase-1-architecture` git tag, not this mutable hash manifest, is what the
golden bit-exact comparison test reads from).

## Tests

Full suite: `python -m pytest tests/ -q` -- all tests passing, including
the real-CUDA-gated tests (this machine has an RTX 2060, so those ran for
real, not skipped). New Phase 2 test files:

| File | Covers |
|---|---|
| `tests/test_model_moe_dispatch.py` | Golden bit-exact comparison against the approved Phase 1 commit's `moe.py`; reference-vs-optimized numerical/gradient equivalence, CPU and real CUDA FP16 autocast. |
| `tests/test_model_moe_property.py` | Randomized dropless invariants across seeds/batch/seq/padding/expert-count/top_k. |
| `tests/test_model_moe_diagnostics.py` | Entropy/margin/pair-coactivation/contribution-norm hand-computed correctness; window aggregation; every routing-health detector on synthetic healthy and pathological cases. |
| `tests/test_model_moe_ablations.py` | Every ablation mode's exact math with deterministic stub experts; seeded-random reproducibility and RNG isolation; no state leakage into normal inference. |
| `tests/test_model_moe_trace.py` | Per-token trace correctness/coverage, bounded-by-default behavior, full-model assembly, JSON export. |
| `tests/test_context_sensitivity.py` | Synthetic context-independent/partial/strong regimes; model-level probe harness; explicit proxy-test labeling. |
| `tests/test_model_fault_injection.py` (extended) | 20 new fault-injection tests -- see requirements traceability for the full list. |

## Evaluations

`not-applicable` -- no evaluation suite exists yet (Phase 5 scope); nothing
in this phase evaluates language capability.

## Ablations (where relevant)

Six evaluation-only ablation modes implemented and validated: see
`tests/test_model_moe_ablations.py` and experiment `exp-0019`
(official-architecture ablation validation: every mode changes output,
reproduces exactly, and does not leak into a subsequent normal call).

## CI workflow / run

The original candidate's runs are recorded above. Independent review found
that its workflows used shallow checkout, so the advertised Phase 1 golden
comparison skipped when the tag was unavailable. All repaired workflows fetch
complete history and tags, and a missing or moved golden tag now fails closed.
Exact final Phase 0/1/2 workflow identities are recorded in the
annotated `phase-2-moe` tag after they pass on the final metadata commit.

## Recovery status

See `docs/recovery/phase-2.md` for the executed fresh-clone recovery
exercise and its result.

## Engineering hours

`docs/time/phase-hours.csv`, `phase-2` rows (engineering row recorded;
self-review and independent-review rows recorded as those passes complete).

## Independent review hours

Recorded separately in `docs/time/phase-hours.csv`; GPT-5.6 Sol's independent
review includes repository audit, adversarial tests, repairs, canonical
experiments, fresh-clone recovery, and CI verification.

## GPU hours

`docs/time/phase-hours.csv`, Phase 2 rows. The independent pass includes real
RTX 2060 CUDA validation in addition to the original profiling evidence.

## CPU / data-processing hours

`docs/time/phase-hours.csv`, Phase 2 rows; the independent entry includes full
validators, test suites, experiment generation, and fresh-environment work.

## Project elapsed days

4 calendar days (initial commit `a991db7`, 2026-08-25, through independent
approval work on 2026-08-28).

## Known failures

None outstanding. Independent review found and repaired the CI/golden-tag,
padding execution, ablation boundary/validation, trace, gradient telemetry,
context-methodology, configurable-health-threshold, and experiment-evidence
gaps documented in `phase-2-sol-independent-review.md`. The one known test
warning is PyTorch's truthful notice that memory-efficient attention backward
is nondeterministic on CUDA; Phase 2 does not broaden its determinism claim.

The `run_phase2_experiment.py`'s `_write` provenance
guard initially rejected writing consecutive canonical experiment
artifacts directly into `docs/experiments/results/` in one shell loop
(each new artifact file made the tree "dirty" for the next `_write` call)
-- this is the same clean-tree discipline `scripts/run_phase1_experiment.py`
already enforces, not a Phase 2 regression; resolved by generating all
seven artifacts to a scratch location first and copying them into the
repository together before committing, matching how Phase 1's script would
need to be used for a multi-experiment batch.

## Negative results

The original candidate's green Phase 2 CI did not execute the golden
comparison because the approved Phase 1 tag was absent in its shallow clone;
that negative result is preserved and fixed. The original padding path also
executed experts for masked positions, now repaired with compact valid-token
dispatch. `exp-0021` showed optimized dispatch faster at every measured shape,
but the optimized path remains opt-in pending a deliberate backend decision.

## Accepted limitations

- **No pretrained checkpoint exists.** Nothing in this phase demonstrates
  learned expert specialization, semantic routing, or context-aware
  routing -- the context-sensitivity harness is measurement
  infrastructure, validated only on synthetic hidden states and a
  clearly labeled untrained-model proxy baseline (`exp-0017`).
- **The optimized dispatch backend is not the default.** Per
  `docs/adr/0009-moe-dispatch-backend-selection.md`, `MoELayer` defaults to
  `backend="reference"` despite `exp-0021` showing `optimized` is
  consistently faster -- this is a deliberate decoupling of the
  correctness-oracle default from a performance decision, not an
  unresolved question.
- **Routing-health detector thresholds are engineering defaults**, not
  empirically tuned against any real training run (none has occurred);
  validated only against synthetic cases (`exp-0018`).
- **No expert is named or claimed to specialize in anything.** Experts
  remain numeric identities throughout this phase's code, docs, and
  experiment results.
- **CUDA determinism claims are not broadened** beyond what Phase 1 already
  established; backward-pass CUDA nondeterminism is not claimed to be
  resolved.
- **Context length validated only through what Phase 1 already validated**
  (4,096 tokens in profiling); Phase 2 did not extend that.
- **No tokenizer, real corpus, or pretraining/runtime/evaluation-suite work
  was performed** -- out of scope per Phase 2 instructions section 23.

## Reproducibility procedure

From a fresh clone at this phase's final commit:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
pip install -e . --no-deps
python scripts/validate_repo.py --all
python scripts/validate_phase1.py --all
python scripts/validate_phase2.py --all
```

See `docs/recovery/phase-2.md` for the actual executed run and its output.
To reproduce a specific Phase 2 experiment:
`python scripts/run_phase2_experiment.py <equivalence|routing-trace|context-sensitivity-baseline|detector-validation|ablation-validation|reproducibility|flowbox-moe-profile|independent-review-demonstration> --output <path>`.

## Reviewer identity

Self-review (implementer): Sonnet (Claude Sonnet 5), see
`docs/phases/phase-2-sonnet-self-review.md`. Independent reviewer and sole
approval authority: GPT-5.6 Sol, see
`docs/phases/phase-2-sol-independent-review.md`.

## Approval status

`APPROVED`
