# ADR-0008: MoE auxiliary-loss (load-balancing and router Z-loss) semantics

Status: accepted
Date: 2026-08-25

## Context

The frozen `ja150m-v0.1` configuration fixes the auxiliary-loss
coefficients (`moe.load_balance_loss_coefficient: 0.01`,
`moe.router_z_loss_coefficient: 0.001`), but not the exact
reduction/normalization formula each loss computes before that
coefficient is applied. Different published MoE architectures use
different conventions here (some load-balance formulas use a hard
top-1-only assignment count, some scale by `n_experts`, some average
per-token before per-layer, some per-layer before per-token). Leaving this
undocumented would make the auxiliary losses an unreviewed implementation
accident rather than a specified part of the architecture, and would make
it impossible to write a manual-reference test against a formula nobody
had committed to.

## Decision

Juniper Auto v0.1 defines, for each MoE layer, over `N` valid (non-padding)
tokens, `E = 8` routed experts, and `K = 2` selected experts per token:

**Load-balancing loss** (per layer):

```
f_e = (# top-K assignments to expert e among the N valid tokens) / (N * K)
p_e = mean over the N valid tokens of the full router softmax probability assigned to expert e
L_balance_layer = E * sum_e(f_e * p_e)
```

`f_e` is computed from the hard top-K *indices* (an integer count, via
`scatter_add` of a constant), so it carries no gradient. Only `p_e` (the
real softmax probability) is differentiable -- this is what lets the
router receive a training signal from this term without creating a false
gradient path through the discrete top-k selection.

**Router Z-loss** (per layer):

```
z_t = logsumexp(router_logits_t)          for each valid token t
L_z_layer = mean over the N valid tokens of z_t^2
```

**Aggregation across layers and coefficient application:**

```
L_balance_raw = mean over the 15 MoE layers of L_balance_layer
L_z_raw = mean over the 15 MoE layers of L_z_layer
L_balance_weighted = 0.01 * L_balance_raw
L_z_weighted = 0.001 * L_z_raw
total_loss = lm_loss + L_balance_weighted + L_z_weighted   (sparse only; dense total_loss = lm_loss)
```

Both losses are computed only over valid tokens; padding contributes to
neither. Both the raw and weighted values are returned separately by the
model's forward output for diagnostics (see
`juniper_auto.model.model.ModelOutput`), so training code can log either.

Implemented in `juniper_auto/model/losses.py`
(`compute_load_balance_loss_raw`, `compute_router_z_loss_raw`) and called
per-layer from `juniper_auto/model/moe.py`; averaged across layers in
`juniper_auto/model/model.py`. Verified against hand-calculated reference
tensors in `tests/test_model_losses.py`.

## Alternatives considered

- **Claim numerical identity with another published model family's
  routing-loss formula** (e.g. a specific well-known MoE paper's exact
  reduction). Rejected: no independent verification of that equivalence
  was performed for this project, and doing so would misrepresent
  provenance. This ADR defines Juniper Auto's own semantics instead.
- **Apply the coefficient per layer before averaging** (`mean_layers(coef *
  L_layer)` instead of `coef * mean_layers(L_layer)`). Mathematically
  equivalent for a constant coefficient (scalar multiplication commutes
  with averaging), so this is not a live alternative -- noted here only so
  a future coefficient schedule that varies per layer knows this ADR's
  formula assumes a single global coefficient.
- **Use per-token hard-argmax frequency (top-1 style) for `f_e` even though
  `top_k=2`.** Rejected: undercounts real dispatch load when `top_k > 1`
  and would not reflect the actual dropless top-2 assignment this
  architecture performs.

## Consequences

- Any future change to this formula is a new architecture-adjacent
  decision requiring a superseding ADR, per governance rule 4.
- `tests/test_model_losses.py` pins this exact formula with hand-computed
  examples; a change to the formula must update both this ADR and those
  tests together, not one without the other.
- Padding-exclusion is part of the spec, not an optional robustness
  addition -- `tests/test_model_causality_padding.py` and
  `tests/test_model_moe.py` verify padded tokens do not move either loss.
