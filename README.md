# Juniper Auto

**Status: Phase 1 candidate - pending independent review.** Phase 0
(research foundation) is independently reviewed and approved -- see tag
`phase-0-foundation`. Phase 1 (the executable reference model and dense
control architecture) is implemented and self-reviewed by Claude Sonnet 5
but **not yet independently approved**; GPT-5.6 Sol has not reviewed this
candidate. **No model has been trained on real data, and no base or
instruction-tuned checkpoint exists.**

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

## What exists right now (Phase 0 + Phase 1 candidate)

### Phase 1 (candidate, pending independent review)

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
expert specialization. Phase 1 (above) implements the model architecture
and training plumbing as executable software; production pretraining is a
later phase.

## Repository layout

```
configs/architecture/   frozen ja150m-v0.1 and ja150m-v0.1-dense configs
juniper_auto/            config, accounting, foundation probe, util (Phase 0); model, training (Phase 1)
juniper_auto/model/       reference model implementation (RMSNorm, RoPE, GQA, MoE, blocks, losses)
juniper_auto/training/    checkpointing, tiny-overfit harness, FLOWBOX profiling
docs/research/          charter, governance
docs/adr/                architecture decision records
docs/architecture/       environment specification, precision policy, reference-model implementation
docs/experiments/        experiment registry documentation + results (docs/experiments/results/)
docs/time/                engineering time accounting
docs/phases/              phase report template + phase reports
docs/recovery/            clean-machine recovery procedure
manifests/                frozen-artifact registry + artifact hashes (Phase 0 and Phase 1)
scripts/                  validate_repo.py / validate_phase1.py (validators), hash_manifest.py, run_phase1_experiment.py
tests/                    pytest suite for everything above
.github/workflows/        CI (Phase 0 Validation, Phase 1 Validation)
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
