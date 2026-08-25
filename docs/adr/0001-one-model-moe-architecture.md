# ADR-0001: One-model MoE architecture

Status: accepted
Date: 2026-08-25

## Context

Prior related work (and a natural default for a project targeting broad
capability) is to build separate specialist models -- one for math, one for
code, one for creative writing, etc. Juniper Auto instead targets a single
~150M-parameter checkpoint. We need to decide, before any model code is
written, whether internal specialization happens via architecture (separate
models) or via learned routing inside one model.

## Decision

Juniper Auto is **one cognitive language model** (`ja150m-v0.1`). Internal
specialization occurs through learned Mixture-of-Experts routing (8 routed
experts + 1 shared expert per MoE layer, top-2 routing) inside that one
model. There is no separate math/code/research/productivity/creativity
language model. Former specialist-model projects may contribute research
methodology, data, tools, and evaluations as inputs to this one model, but
not as separate permanent model components. Multiple temporary runtime
*instances* of the same checkpoint may be studied without violating this.

## Alternatives considered

- **Separate specialist models per domain.** Rejected: multiplies the
  parameter/compute/maintenance budget on FLOWBOX-class hardware, and
  directly contradicts the project's primary research question (how far
  learned internal MoE specialization can go in one model).
- **A large single dense model instead of MoE.** Rejected as the primary
  architecture: forgoes the core research question about sparse
  specialization, though it is preserved as the mandatory dense control
  (see ADR-0002) specifically so this comparison stays possible.
- **Hierarchical or nested MoE (models routing to other models).**
  Rejected: explicit v0.1 non-goal (see
  docs/research/project-charter.md); adds architectural complexity before
  evidence justifies it, contradicting governance rule 39.

## Consequences

- `configs/architecture/` may only ever contain the sparse `ja150m-v0.1` and
  its dense control; a third trainable architecture requires a superseding
  ADR.
- Expert specialization claims must be measured (governance rule 11), not
  assumed from a naming scheme.
- Runtime work may still spin up multiple *instances* of the one checkpoint
  (different configuration/context/tools) without this being a violation.
