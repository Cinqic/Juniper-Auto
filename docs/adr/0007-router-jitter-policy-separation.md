# ADR-0007: Separate Router-Jitter Permission From Experiment Enablement

Status: accepted
Date: 2026-08-25

## Context

The initial Phase 0 sparse configuration encoded
`training_router_jitter: true`. The authoritative specification permits
training-only jitter only as a documented configuration, while Phase 8 must
test whether it helps or merely adds noise. A boolean `true` incorrectly
froze an unperformed empirical decision into the architecture.

## Decision

The v0.1 architecture records `training_router_jitter_policy:
experiment_only` and leaves `training_router_jitter_magnitude` null. It fixes
evaluation and inference jitter to false. A later Phase 8 experiment
configuration may enable and size training jitter without changing the
architecture identity.

## Alternatives

- Keep the boolean enabled: rejected because it claims a Phase 8 result in
  Phase 0.
- Disable and prohibit jitter: rejected because the specification explicitly
  permits a documented training experiment.
- Invent a default magnitude: rejected because no evidence supports one.

## Consequences

Architecture capability and experiment choice are unambiguous. Phase 8 must
record any enabled jitter and magnitude in its run configuration, and
evaluation/inference remain jitter-free.
