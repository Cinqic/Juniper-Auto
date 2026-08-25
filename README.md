# Juniper Auto

**Status: Phase 0 research foundation independently reviewed and approved
with documented accepted limitations. The canonical approval commit is the
target of tag `phase-0-foundation`. No model has been trained.**

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

## Current architecture (target, not yet trained)

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

## What exists right now (Phase 0 only)

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

**What does not exist yet:** a tokenizer, any training data, any trained
checkpoint (base or instruction-tuned), a runtime, tools, memory, or
autonomy of any kind. Juniper Auto does **not** currently perform
inference, does not have 16K validated context, does not have measured
expert specialization, and has not been trained on any tokens. Model
implementation begins in Phase 1.

## Repository layout

```
configs/architecture/   frozen ja150m-v0.1 and ja150m-v0.1-dense configs
juniper_auto/           Phase 0 Python package: config, accounting, foundation probe, util
docs/research/          charter, governance
docs/adr/                architecture decision records
docs/architecture/       environment specification
docs/experiments/        experiment registry documentation
docs/time/                engineering time accounting
docs/phases/              phase report template + phase reports
docs/recovery/            clean-machine recovery procedure
manifests/                frozen-artifact registry + artifact hashes
scripts/                  validate_repo.py (canonical validator), hash_manifest.py
tests/                    pytest suite for everything above
.github/workflows/        CI
```

`model/`, `runtime/`, `training/`, `tools/`, `data/`, `evals/` exist as
reserved top-level locations for later phases and are intentionally close
to empty right now.

## Getting started (development environment)

```bash
git clone https://github.com/Cinqic/Juniper-Auto.git
cd Juniper-Auto
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-lock.txt
pip install -e . --no-deps
python scripts/validate_repo.py --all
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
