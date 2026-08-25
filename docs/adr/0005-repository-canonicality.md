# ADR-0005: Repository canonicality

Status: accepted
Date: 2026-08-25

## Context

The project's core Phase 0 premise is that local storage (including
FLOWBOX) is disposable. We need one unambiguous statement of what "the
project" actually refers to when local and remote state could in principle
diverge.

## Decision

`github.com/Cinqic/Juniper-Auto` (branch `main`) is the single canonical,
authoritative copy of Juniper Auto. A change is not "done" until it is
committed and pushed to this remote and passes GitHub Actions CI on that
exact commit hash. Local passing tests, local validation, or a local-only
commit are necessary but not sufficient.

## Alternatives considered

- **Treat the local FLOWBOX checkout as canonical, with GitHub as a
  backup.** Rejected: directly contradicts the "local storage is
  disposable" principle that motivates Phase 0's existence in the first
  place -- if FLOWBOX were canonical, losing it would be catastrophic
  rather than recoverable.
- **No single canonical remote; treat all clones/forks as equally
  authoritative.** Rejected: makes "is this done" undecidable without a
  single reference point, and complicates CI (which run counts?).

## Consequences

- Every phase report must record the exact candidate commit hash and the
  CI run identity for that commit (governance rules 2 and 3).
- Recovery documentation and the clean-clone recovery exercise
  (`docs/recovery/`) exist specifically to prove this remote, on its own,
  is sufficient to reconstruct a working project.
