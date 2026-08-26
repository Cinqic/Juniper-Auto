# ADR-0009: MoE dispatch backend selection (reference default, optimized available)

Status: accepted
Date: 2026-08-26

## Context

Phase 1 shipped a single, deliberately naive MoE dispatch: an explicit
Python loop over `(expert, slot)` pairs with boolean-mask gather, chosen
for inspectability over throughput (see `juniper_auto/model/moe.py`'s
original docstring and the Phase 1 report's accepted limitation "the
reference MoE dispatch is not optimized"). Phase 2's instructions require
building a faster pure-PyTorch alternative, proving it numerically
equivalent to the reference path, and then making an evidence-based
decision about which becomes the default -- explicitly warning against
"automatically mak[ing] the optimized backend the production default
merely because it exists."

Phase 2 built `juniper_auto/model/moe_dispatch.py`'s `optimized_dispatch`
(sort-and-group: one global argsort of the flattened `(token, slot)`
assignments by expert id, then one gather + matmul + scatter per *expert*
instead of per `(expert, slot)` pair) and measured it against
`reference_dispatch` on FLOWBOX's RTX 2060 across five representative
prefill shapes (exp-0021): optimized was faster at every shape, by
1.5x-2.6x, with peak VRAM difference under 2 MB. Numerical equivalence
(routing identical, output within 1e-5 absolute tolerance on CPU and
within a documented looser tolerance under real CUDA FP16 autocast) is
proven in `tests/test_model_moe_dispatch.py` and exp-0015, on both tiny
synthetic configs and the official `ja150m-v0.1` architecture.

## Decision

`MoELayer`'s constructor default remains `backend="reference"`. The
`optimized` backend is fully implemented, tested, measured, and selectable
per-instance (`MoELayer(cfg, backend="optimized")`) or per-call
(`layer(x, ..., backend="optimized")`), including at the model level via
future training-config wiring, but it does **not** become the class-level
default in this phase.

This is a deliberate decoupling, not an unresolved question: the evidence
in exp-0021 clearly favors switching (materially faster, no VRAM
regression, proven-equivalent). The reference path is kept as the default
specifically so it remains unconditionally the code path every existing
Phase 1/2 test exercises unless a test explicitly opts into
`backend="optimized"` -- including the golden bit-exact comparison against
the approved Phase 1 commit's `moe.py`
(`tests/test_model_moe_dispatch.py::test_refactored_reference_dispatch_matches_phase1_golden_bit_for_bit`),
which depends on `MoELayer(cfg)`'s default forward call being bit-identical
to Phase 1's reference implementation. Making `optimized` the default would
require rewriting that proof to explicitly pass `backend="reference"`
everywhere, entangling a pure performance decision with the correctness-
oracle property Phase 2 was asked to preserve.

Training code that wants the measured throughput improvement selects
`backend="optimized"` explicitly; this is expected to happen in a later
phase's training configuration once the choice can be validated against a
real training run rather than a single-layer microbenchmark.

## Alternatives considered

- **Switch the default to `optimized` now, given exp-0021's clear
  result.** Rejected for this phase: it would silently change every
  existing caller's dispatch path (including the golden-equivalence tests,
  which would need every call site updated to `backend="reference"`
  instead), trading a simple, low-risk default for a performance win that
  can be captured just as well via explicit opt-in.
- **Make the default configurable via `ArchitectureConfig`.** Rejected:
  dispatch backend selection is an execution-strategy detail, not part of
  the frozen architecture identity (`ja150m-v0.1`'s parameter counts and
  layer semantics are unaffected by which backend runs it) -- adding it to
  the frozen config would conflate the two and require bumping
  `architecture_id` for a change that alters no learned behavior.
- **Delete the reference backend once the optimized backend is proven
  equivalent.** Rejected: the reference path's simplicity is exactly what
  makes it usable as a correctness oracle for future refactors and for
  manual token-by-token auditing (Phase 2 self-review Pass B/C). Removing
  it would remove the thing Phase 2 was asked to preserve.

## Consequences

- `backend="reference"` remains the implicit contract for any code that
  constructs `MoELayer(cfg)` without an explicit backend -- this must not
  change silently in a later phase; doing so is itself a decision requiring
  a superseding ADR.
- A later phase that wants `optimized` as the trained-model default should
  record that as its own decision (with its own training-run evidence), not
  assume this ADR already authorizes it.
- `scripts/run_phase2_experiment.py equivalence` and `flowbox-moe-profile`
  remain the canonical way to re-verify this decision's evidence if the
  dispatch implementations change.
