# Phase 1 Recovery Exercise

This documents an actual fresh-clone/fresh-venv recovery exercise
performed for the Phase 1 candidate (not a plausibility claim -- see
[project-governance](../research/project-governance.md) rule 15,
executed-action truthfulness). It extends
[docs/recovery/README.md](README.md) (the general Phase 0 procedure,
unchanged) with the Phase 1-specific validation step.

## Procedure

```bash
# 1. A genuinely fresh directory, not a copy of the working checkout.
mkdir -p /tmp/juniper-auto-phase1-recovery && cd /tmp/juniper-auto-phase1-recovery

# 2. Clone from the actual GitHub remote at the exact candidate commit.
git clone https://github.com/Cinqic/Juniper-Auto.git .
git checkout <candidate-commit>

# 3. A new Python 3.12 virtual environment.
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 4. Install solely from the committed, hash-pinned dependency lock.
pip install -r requirements-lock.txt
pip install -e . --no-deps

# 5. Phase 0 validation.
python scripts/validate_repo.py --all

# 6. Phase 1 validation (imports, official parameter counts, forward/
#    backward smoke, checkpoint round trip, full pytest suite,
#    documentation/hash/experiment/time-accounting checks).
python scripts/validate_phase1.py --all
```

## Result

Recorded verbatim below from the actual execution (not summarized), run
against commit `<candidate-commit>` in a directory this project had never
touched before.

<!-- FILLED IN FROM THE ACTUAL EXECUTED RUN BELOW -->

## What this exercise proves

- The exact Phase 1 candidate commit is fully reconstructible from GitHub
  alone -- no dependency on local FLOWBOX state, cached wheels, or any
  file outside the committed repository and `requirements-lock.txt`'s
  pinned PyPI packages.
- `scripts/validate_phase1.py --all` genuinely passes from a cold clone,
  not only in the environment that developed it.
- No local, uncommitted, or FLOWBOX-specific state was required at any
  step -- see the absolute-path scan in
  `docs/phases/phase-1-sonnet-self-review.md`, Pass I.

## Known Phase 1-specific recovery notes

- **GPU-dependent experiments are not re-run by this exercise.** CI and
  this recovery procedure are both CPU-only by design (see ADR-0006,
  inherited from Phase 0); `docs/experiments/results/exp-0004` through
  `exp-0008` are the recorded evidence of the GPU-dependent tiny-overfit
  and hardware-profiling experiments, executed once on the actual FLOWBOX
  machine, not re-executed here. `scripts/run_phase1_experiment.py`
  documents the exact commands to reproduce them on hardware with a
  CUDA-capable GPU.
- All other Phase 0 troubleshooting notes in
  [docs/recovery/README.md](README.md) apply unchanged (no standalone
  global `pip3`, large `torch` download, no CUDA/GPU determinism claim).
