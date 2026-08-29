# Juniper Auto

**Status: Phase 0 approved; Phase 1 approved; Phase 2 independently
approved.** See the annotated tags `phase-0-foundation`,
`phase-1-architecture`, and `phase-2-moe`, the GPT-5.6 Sol
[independent Phase 1 review](docs/phases/phase-1-sol-independent-review.md),
the [independent Phase 2 review](docs/phases/phase-2-sol-independent-review.md),
and the consolidated Phase 2 report at
[docs/phases/phase-2-moe.md](docs/phases/phase-2-moe.md). **No
model has been trained on real data, no base or instruction-tuned
checkpoint exists, no tokenizer or real pretraining corpus exists, and no
expert specialization has been demonstrated.**

Juniper Auto is a research project asking: how capable, efficient,
autonomous, persistent, adaptable, reliable, and meaningfully
self-improving can one ~150-million-parameter open sparse
Mixture-of-Experts language model become while remaining practical on
consumer hardware and genuinely customizable?

It is **one cognitive language model**. Internal specialization happens
through learned MoE routing inside that one model -- not through separate
math/code/research/creativity models. See the full
[research charter](docs/research/project-charter.md) for the complete
research questions and scope.

## Current architecture (implemented as software, not yet trained)

| | Sparse (`ja150m-v0.1`) | Dense control (`ja150m-v0.1-dense`) |
|---|---|---|
| Total parameters | 150,031,360 | 79,191,040 |
| Standard active parameters | 79,252,480 | 79,191,040 (no sparsity) |
| Context (current) | 4,096 tokens | 4,096 tokens |
| Context (future target, not advertised) | 16,384 tokens | 16,384 tokens |
| Layers | 20 (5 dense-anchor + 15 MoE) | 20 (all dense) |
| Vocabulary (target) | 36,864 | 36,864 |

These counts are independently derived by code, not transcribed --
`juniper_auto/accounting/` computes them from
`configs/architecture/*.yaml`, and `tests/test_parameter_accounting.py`
asserts the exact values on every run. The dense control is a mandatory
research requirement (not an optional ablation) for measuring whether the
sparse architecture's complexity is earning its keep -- see
[ADR-0002](docs/adr/0002-dense-control-requirement.md).

**Consumer-hardware target:** the initial engineering baseline is
"FLOWBOX" -- a Ryzen 7 5700G, RTX 2060 (6 GB VRAM), 16 GB RAM machine.
Practical operation on hardware like this is a standing design constraint,
not a one-time benchmark.

**License:** this repository ships an MIT `LICENSE`. Whether that license
is the final intended license for trained model weights/datasets is an
open governance item, not yet decided -- see the charter's License status
section.

## What exists right now (approved Phases 0, 1, and 2)

### Phase 2 (independently approved)

Phase 2 validates the sparse MoE routing/dispatch machinery itself:
correctness, droplessness, reproducibility, per-token inspectability,
numerical stability, instrumentation, ablatability, and context-sensitivity
measurement infrastructure -- see
[docs/phases/phase-2-moe.md](docs/phases/phase-2-moe.md) for the full
report and
[docs/architecture/moe-routing-diagnostics.md](docs/architecture/moe-routing-diagnostics.md)
for the implementation details. What was actually verified, with executed
evidence:

- The Phase 1 correctness-first reference dispatch is preserved bit-for-
  bit for its default call, proven directly against a copy of `moe.py`
  loaded from the immutable `phase-1-architecture` git tag.
- A new pure-PyTorch optimized dispatch backend is numerically equivalent
  to the reference backend (routing identical; output within 1e-5
  tolerance on CPU, forward and backward) and measured 1.5x-2.6x faster
  on FLOWBOX's RTX 2060 across five representative shapes with negligible
  VRAM difference -- the reference backend remains `MoELayer`'s default
  (see [ADR-0009](docs/adr/0009-moe-dispatch-backend-selection.md)).
- Dropless routing invariants (every valid token gets exactly two unique
  routed experts, weights sum to one, no token dropped) hold across 212
  randomized (seed, shape, expert count, top_k, padding layout) cases.
- Full per-token, per-MoE-layer routing traces work end to end on the
  official architecture (480/480 expected records produced).
- Extended router instrumentation (entropy, margins, expert-pair
  co-activation, contribution norms, router-logit magnitude, window
  aggregation) and six routing-health detectors, each validated against
  synthetic healthy and pathological cases.
- Six evaluation-only ablation modes (disable/replace/zero expert,
  uniform/seeded-random router override), each proven exactly against
  hand-computed expected outputs, fail-closed during training, range-
  validated, and proven not to leak into normal inference.
- A context-sensitivity probe harness, validated on synthetic hidden
  states with known context-independent/partial/strong regimes, plus a
  tokenizer-independent controlled catalog covering all required lexical,
  domain, syntax, mathematical, and positional probe categories.
- 23 deliberate fault-injection tests proving the test suite is
  load-bearing (would fail on the specific broken behavior each targets).
- Independent repairs now exclude padding from physical expert execution,
  make every required route/reconstruction/shared-activation field explicit
  in traces, expose per-layer expert gradient norms and raw/weighted router
  losses, and ensure remote CI cannot silently skip the Phase 1 golden
  comparison.

**What Phase 2 explicitly does not claim:** no expert specialization,
semantic routing, or context-aware routing has been demonstrated (no
training has occurred); routing-health detector thresholds are engineering
defaults validated only against synthetic cases, not tuned against any
real training run; the optimized dispatch backend, though measured
faster, is not the production default; CUDA determinism claims are not
broadened beyond what Phase 1 already established. See
[docs/phases/phase-2-moe.md](docs/phases/phase-2-moe.md)'s accepted
limitations for the full list.

### Phase 1 (independently approved)

Phase 1 implements the full `ja150m-v0.1` (sparse MoE) and
`ja150m-v0.1-dense` architectures as real, executable PyTorch code on one
shared stack (`juniper_auto/model/`), plus the training-support plumbing
needed to exercise it (`juniper_auto/training/`). This is architecture and
training-plumbing verification, not pretraining -- see
[docs/phases/phase-1-architecture.md](docs/phases/phase-1-architecture.md)
for the full report and
[docs/architecture/reference-model-implementation.md](docs/architecture/reference-model-implementation.md)
for the implementation details. What was actually verified, with executed
evidence (not estimates):

- Both models instantiate to their exact frozen parameter counts
  (150,031,360 sparse / 79,191,040 dense), verified two independent ways
  (config-derived and directly from the instantiated PyTorch modules).
- Forward propagation, causal LM loss, backward propagation, and AdamW
  optimizer updates all work, including under real FP16 mixed precision
  with dynamic gradient scaling on the actual FLOWBOX RTX 2060.
- Both full official models memorize a tiny deterministic synthetic
  training set (100% token accuracy, near-zero loss, zero NaN/Inf events)
  -- proof the training plumbing is wired correctly, not evidence of any
  general language capability.
- Checkpoint save/restore, including every RNG stream and the training
  data-stream position, was validated with an exact (bit-for-bit)
  interrupted-vs-resumed training comparison on the full 150M-parameter
  model.
- FLOWBOX hardware profiling (inference, training step, checkpoint I/O,
  and a full 4,096-token batch-1 reference inference pass) was measured
  for both architectures and both comfortably fit the 6 GB VRAM budget.

**What Phase 1 explicitly does not claim:** no tokenizer exists yet, no
production pretraining corpus exists, no 6B-token pretraining has
occurred, no base or instruction-tuned checkpoint exists, no autonomy
runtime exists, no expert specialization has been demonstrated (the tiny
synthetic overfit target is far too small and repetitive to show
specialization), and 16K context remains unvalidated (only the frozen
4,096-token `context_length` was exercised). A model that memorizes four
synthetic sequences is training-plumbing correctness, not intelligence.

### Phase 0 (approved foundation)

Phase 0 is foundation engineering, not model training. It exists to prove
the project can survive the loss of its current local development machine.
What Phase 0 actually builds:

- The [research charter](docs/research/project-charter.md) and
  [governance rules](docs/research/project-governance.md) in operational
  (not just aspirational) form.
- The frozen `ja150m-v0.1` and `ja150m-v0.1-dense` architecture
  configurations, schema-validated
  (`juniper_auto/config/`) and independently parameter-counted
  (`juniper_auto/accounting/`).
- A deterministic seed framework, structured JSON logging, and a
  **FoundationProbe** -- a minimal tensor-pipeline smoke test that is
  explicitly *not* the Juniper Auto model, used only to prove imports,
  config loading, seeding, and reproducibility work end to end
  (`juniper_auto/foundation/`).
- An [ADR system](docs/adr/), an
  [experiment registry](docs/experiments/README.md), and
  [time accounting](docs/time/phase-hours.csv).
- A [frozen-artifact manifest](manifests/frozen-artifacts.yaml) and a
  [SHA-256 hash manifest](manifests/phase-0-artifact-hashes.yaml) for the
  artifacts that exist so far.
- A single validation entrypoint (`scripts/validate_repo.py --all`) and
  matching GitHub Actions CI
  (`.github/workflows/phase-0-validation.yml`) that runs on a clean,
  CPU-only Linux runner -- not FLOWBOX.
- A tested [recovery procedure](docs/recovery/README.md), actually
  exercised end to end on a fresh clone and fresh virtual environment (see
  `docs/phases/phase-0-foundation.md`).

**What does not exist yet:** a tokenizer, any real training data, any
trained checkpoint (base or instruction-tuned), a runtime, tools, memory,
or autonomy of any kind. Juniper Auto has **not** been trained on any real
tokens, does not have 16K validated context, and does not have measured
expert specialization. Phase 1 implements the model architecture and
training plumbing as executable software; Phase 2 (above) validates the
sparse routing subsystem specifically; production pretraining is a later
phase.

## Repository layout

```
configs/architecture/   frozen ja150m-v0.1 and ja150m-v0.1-dense configs
juniper_auto/            config, accounting, foundation probe, util (Phase 0); model, training (Phase 1); analysis (Phase 2)
juniper_auto/model/       reference model implementation (RMSNorm, RoPE, GQA, blocks, losses); MoE routing/dispatch/ablations/diagnostics (Phase 2)
juniper_auto/training/    checkpointing, tiny-overfit harness, FLOWBOX profiling
juniper_auto/analysis/    context-sensitivity probe harness (Phase 2)
docs/research/          charter, governance
docs/adr/                architecture decision records
docs/architecture/       environment specification, precision policy, reference-model implementation, MoE routing/diagnostics (Phase 2)
docs/experiments/        experiment registry documentation + results (docs/experiments/results/)
docs/time/                engineering time accounting
docs/phases/              phase report template + phase reports
docs/recovery/            clean-machine recovery procedure
manifests/                frozen-artifact registry + artifact hashes (Phase 0, Phase 1, Phase 2)
scripts/                  validate_repo.py / validate_phase1.py / validate_phase2.py (validators), hash_manifest.py, run_phase1_experiment.py, run_phase2_experiment.py
tests/                    pytest suite for everything above
.github/workflows/        CI (Phase 0 Validation, Phase 1 Validation, Phase 2 Validation)
```

`runtime/`, `tools/`, `data/`, `evals/` exist as reserved top-level
locations for later phases and are intentionally close to empty right now.

## Getting started (development environment)

```bash
git clone https://github.com/Cinqic/Juniper-Auto.git
cd Juniper-Auto
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-lock.txt
pip install -e . --no-deps
python scripts/validate_repo.py --all    # Phase 0 baseline
python scripts/validate_phase1.py --all  # Phase 1 (includes the Phase 0 baseline)
```

See [docs/recovery/README.md](docs/recovery/README.md) for the full,
troubleshooting-included procedure, and
[docs/architecture/environment-specification.md](docs/architecture/environment-specification.md)
for the exact environment this was developed and validated against.

## Contributing / project rules

Read [docs/research/project-governance.md](docs/research/project-governance.md)
before proposing architecture, data, or runtime changes -- it states the
project's permanent rules (one model, dense-control requirement,
reproducibility, evaluation before expansion, etc.) in operational form,
including what each rule requires in code or tests, not just in prose.
