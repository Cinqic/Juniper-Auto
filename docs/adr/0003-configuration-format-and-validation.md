# ADR-0003: Configuration format and validation

Status: accepted
Date: 2026-08-25

## Context

Architecture, training, runtime, tools, memory, and evaluation
configuration all need one coherent, machine-readable, human-readable,
schema-validated, deterministic, versionable configuration system that will
still make sense once the repository has grown well past Phase 0.

## Decision

- **Serialization format: YAML** (`configs/architecture/*.yaml`). Human
  editable and diffable, standard for ML configuration, and comments are
  possible directly in the frozen files to explain non-obvious values (e.g.
  the residual-output-projection std derivation).
- **Schema/validation: Pydantic v2** (`juniper_auto/config/schema.py`),
  using `extra="forbid"` strict models so unknown fields are rejected rather
  than silently ignored, plus `model_validator` cross-field checks for
  things a plain JSON Schema can't easily express (layer partition
  correctness, head-count divisibility, dropless-routing enforcement).
- **Frozen-value assertions live separately** from structural schema
  validation, in `juniper_auto/config/frozen.py`. The schema validates that
  *any* architecture config (this version or a future one) is internally
  consistent; the frozen-value module asserts that a *specific known*
  architecture id (`ja150m-v0.1`, `ja150m-v0.1-dense`) matches its exact
  frozen numeric spec. This separation means schema code doesn't need
  editing every time a new architecture version is added, while frozen
  values still fail loudly on drift.

## Alternatives considered

- **JSON instead of YAML.** Rejected: no comments, worse ergonomics for
  hand-maintained frozen config files that need to explain derived values.
- **Plain JSON Schema instead of Pydantic.** Rejected: JSON Schema alone
  can't naturally express the cross-field invariants this config needs
  (layer partition, head divisibility, top_k vs. expert count) without
  external custom-keyword tooling; Pydantic gives typed Python objects for
  free, which `juniper_auto/accounting/` depends on directly.
- **Dataclasses + hand-written validation instead of Pydantic.** Rejected:
  would re-implement a worse version of what Pydantic already provides
  (coercion, strict mode, clear error messages), for one extra dependency
  that is broadly standard in the Python ecosystem.
- **Hard-coding frozen values inside the same schema class used for general
  validation.** Rejected: would force every future architecture version to
  either violate the frozen v0.1 values or require editing shared schema
  code, coupling "is this config well-formed" to "is this config exactly
  v0.1" in a way that doesn't scale.

## Consequences

- New configuration domains (training, runtime, tools, memory, evaluation)
  should follow the same YAML + Pydantic strict-model pattern for
  consistency, unless a specific domain has a concrete reason to differ
  (documented in its own ADR).
- Adding a new architecture version means adding a new YAML file plus, if
  its values should be frozen, a new entry in
  `juniper_auto/config/frozen.py` -- not editing the existing frozen entries.
