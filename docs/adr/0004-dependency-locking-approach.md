# ADR-0004: Dependency locking approach

Status: accepted
Date: 2026-08-25

## Context

Phase 0 needs a reproducible dependency mechanism that a clean clone and a
GitHub Actions CI runner can both use without depending on the developer's
global Python packages, an existing virtual environment, or manually
maintained wheels. FLOWBOX (the development machine) does not have `uv`
installed.

## Decision

- **Dependency declaration:** `pyproject.toml` (PEP 621 `[project]` table),
  with a small, deliberately minimal Phase 0 dependency set
  (`pydantic`, `PyYAML`, `torch`, `numpy`; dev extra: `pytest`,
  `pip-tools`).
- **Lock mechanism:** `requirements-lock.txt`, generated with
  `pip-compile --generate-hashes --allow-unsafe` (pip-tools) from
  `pyproject.toml`. This hash-pins every dependency (direct and transitive),
  which is what makes `pip install -r requirements-lock.txt` deterministic
  -- not just `pip freeze`, which pins versions but not content hashes.
- **Bootstrap:** `python3 -m venv .venv` (stdlib, no extra tool needed) plus
  `pip install pip-tools` inside that venv to regenerate the lock when
  `pyproject.toml` changes. No global tool installation is required on the
  host machine.

## Alternatives considered

- **`uv` + `uv.lock`.** Rejected for Phase 0 specifically because `uv` was
  not present on FLOWBOX and installing a new global tool to bootstrap the
  project's own dependency bootstrapping added an extra layer of "trust me"
  that pip-tools (installable via `pip`, already trusted) avoids. This
  decision can be revisited by a superseding ADR if `uv` is adopted later
  -- nothing here depends on pip-tools specifically at the *code* level,
  only at the lock-generation-tooling level.
- **`pip freeze` as the lock file.** Rejected: only pins versions from
  whatever happens to be installed locally, with no hashes, and easily
  drifts from `pyproject.toml`'s declared direct dependencies if someone
  installs something ad hoc into the venv.
- **Poetry.** Rejected: heavier tool surface than needed for Phase 0's small
  dependency set; would also require a global install not currently present.
- **No lock file, floating versions only.** Rejected outright by the Phase 0
  requirements -- floating dependencies are explicitly disallowed as the
  sole reproducibility mechanism.

## Consequences

- `requirements-lock.txt` must be regenerated (`pip-compile --extra dev
  --output-file=requirements-lock.txt --generate-hashes --strip-extras
  --allow-unsafe pyproject.toml`) whenever `pyproject.toml`'s dependencies
  change, and the regenerated file committed in the same change.
- CI and the recovery procedure both install from `requirements-lock.txt`
  with `pip install -r requirements-lock.txt`, never from
  `pyproject.toml` directly, so CI and local development use identical
  pinned + hash-verified dependency versions.
- `torch` is resolved without CUDA-vs-CPU environment markers, so the same
  lock file installs the full CUDA-capable wheel set on both FLOWBOX (where
  CUDA is used) and the CPU-only GitHub Actions runner (where the CUDA
  libraries are installed but simply unused). This trades a larger CI
  install (multiple GB) for one lock file that is identical across
  environments; revisit with a superseding ADR if CI install time becomes a
  real problem.
