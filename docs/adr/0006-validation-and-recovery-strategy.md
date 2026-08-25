# ADR-0006: Validation and recovery strategy

Status: accepted
Date: 2026-08-25

## Context

Phase 0's stated purpose is to prove the project survives loss of the
current local machine. We need a concrete, repeatable way to check that
claim rather than trusting documentation to be accurate.

## Decision

- **One canonical local validation entrypoint:** `scripts/validate_repo.py
  --all`, which orchestrates environment sanity, imports, config
  validation, parameter accounting, artifact-hash verification, repository
  integrity checks, the deterministic foundation probe, and the pytest
  suite, in that order, failing fast with a clear gate name on the first
  failure.
- **Remote CI mirrors the same gates** on a clean, CPU-only, non-FLOWBOX
  Linux runner (GitHub Actions `ubuntu-latest`), so passing CI is real
  independent evidence, not a restatement of local results.
- **A genuine clean-clone recovery exercise** is performed as part of
  Phase 0 engineering, not merely described: a fresh `git clone` into an
  isolated directory, a fresh venv, dependency installation strictly from
  the committed lock file, and a full `scripts/validate_repo.py --all` run,
  with the actual commands and results recorded in the Phase 0 report.

## Alternatives considered

- **Multiple separate validation scripts (one per gate), run manually in
  sequence.** Rejected: easy for a future change to silently skip a gate;
  a single orchestrator with `--all` makes "did everything pass" a single
  yes/no answer.
- **Trust the recovery documentation without executing it.** Rejected
  outright by the Phase 0 requirements, and specifically flagged as
  insufficient ("do not merely read the recovery instructions and declare
  them plausible").
- **Treat GitHub Actions CI as sufficient on its own, skip the local
  fresh-clone exercise.** Rejected: CI proves a clean Linux environment
  works, but doesn't exercise the human-facing recovery instructions in
  `docs/recovery/` the way an actual person following them would.

## Consequences

- Every new Phase 0 validation gate (future config domains, future tests)
  must be wired into `scripts/validate_repo.py`, or it isn't really part of
  "the validation suite" for the project's own purposes.
- The Phase 0 report must record the recovery exercise as an actual
  executed test with real output, not a plausibility claim, per governance
  rule 15 (executed-action truthfulness).
