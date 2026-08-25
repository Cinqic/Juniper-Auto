# Architecture Decision Records (ADRs)

## Naming convention

`docs/adr/NNNN-short-kebab-title.md`, where `NNNN` is a zero-padded,
immutable, monotonically increasing 4-digit id assigned at creation time and
never reused or renumbered.

## Format

Each ADR has:

- **Status**: `proposed`, `accepted`, `superseded by ADR-NNNN`, or
  `rejected`.
- **Date**: the date the decision was made (`YYYY-MM-DD`).
- **Context**: what problem or question forced this decision.
- **Decision**: what was decided, stated plainly.
- **Alternatives considered**: what else was on the table and why it lost.
- **Consequences**: what this decision makes easier, harder, or requires
  going forward.

## Rules

- ADRs are not created for trivial formatting decisions.
- Accepted ADR history is never rewritten. A later change supersedes an
  earlier ADR with a new ADR that explicitly references the old one
  (`Status: superseded by ADR-0007`) -- the old file's content is left as it
  was written.
- Not every Phase 0 decision needs an ADR -- only decisions with real
  alternatives and real consequences for later phases.

## Index

| ID | Title | Status |
|---|---|---|
| [0001](0001-one-model-moe-architecture.md) | One-model MoE architecture | accepted |
| [0002](0002-dense-control-requirement.md) | Dense-control requirement | accepted |
| [0003](0003-configuration-format-and-validation.md) | Configuration format and validation | accepted |
| [0004](0004-dependency-locking-approach.md) | Dependency locking approach | accepted |
| [0005](0005-repository-canonicality.md) | Repository canonicality | accepted |
| [0006](0006-validation-and-recovery-strategy.md) | Validation and recovery strategy | accepted |
