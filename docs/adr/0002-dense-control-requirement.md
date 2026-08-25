# ADR-0002: Dense-control requirement

Status: accepted
Date: 2026-08-25

## Context

A sparse MoE model's headline efficiency claims (active vs. total
parameters) are easy to overstate without a comparison point. We need to
decide whether a dense baseline is a mandatory research artifact or an
optional later ablation.

## Decision

`ja150m-v0.1-dense` (79,191,040 parameters) is a **mandatory, first-class**
research requirement, frozen alongside the sparse architecture in Phase 0,
not deferred to whenever someone gets around to an ablation. It shares
vocabulary, embeddings, layer count, attention/GQA/QK-Norm/RoPE, context,
normalization, and initialization family with the sparse model, differing
only in using the dense 1536-dim SwiGLU FFN in all 20 layers instead of
MoE routing.

## Alternatives considered

- **No dense control; compare only against external published models.**
  Rejected: external models differ in data, tokenizer, and training
  compute, confounding any efficiency claim about *this* project's MoE
  design specifically.
- **Build the dense control later, once the sparse model works.** Rejected:
  risks the dense config quietly drifting to make a future comparison
  easier to win (explicitly prohibited, see
  docs/research/project-charter.md), and risks the comparison never
  actually happening under time pressure.
- **Match dense and sparse by total parameters instead of active
  parameters.** Rejected: total-parameter matching would make the dense
  model far more expensive to run, defeating the point of asking whether
  MoE routing earns its complexity at a comparable serving cost.

## Consequences

- Any sparse-vs-dense capability or efficiency claim must run both
  architectures through the same evaluation harness (governance rule 8).
- The dense config is frozen at the same time as the sparse config and
  covered by the same parameter-accounting tests, so the two configs cannot
  silently diverge from their intended "matched except for FFN" design.
