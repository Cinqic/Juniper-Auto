# Phase 2 Sonnet Self-Review

## Candidate state reviewed

All Phase 2 work committed on top of the approved Phase 1 candidate
`073acf46e04241ed35d00bc4b4c29ac463ee744d` (tag `phase-1-architecture`),
through this self-review's fix cycle. This review assumes the
implementation is wrong until each specific check below demonstrates
otherwise -- it is a re-inspection of the repository as delivered, not a
restatement of the engineering summary.

## Review start / end

Self-review began after the core implementation (routing/dispatch/
ablations/diagnostics/context-sensitivity), the extended fault-injection
tests, the seven executed Phase 2 experiments, and the validator/CI/
manifest/documentation scaffolding were in place, and proceeded as a
distinct pass over the actual repository state. See `docs/time/phase-hours.csv`
for the self-review time row.

## Review passes and findings

### Pass A -- Baseline and scope

Confirmed `phase-1-architecture` resolves (via `git rev-parse
phase-1-architecture^{commit}`) to exactly `073acf46e04241ed35d00bc4b4c29ac463ee744d`,
matching the tag's annotated message and the expected commit stated at the
top of this engagement's instructions. Confirmed `configs/architecture/*.yaml`
are byte-for-byte unmodified from Phase 1 (`git diff phase-1-architecture --
configs/`) and that `gate_frozen_architecture_unchanged` in
`scripts/validate_phase2.py` independently re-derives all three frozen
parameter counts from the instantiated model, not from a cached constant.
Searched the diff against `phase-1-architecture` for any Phase 3+ scope
creep (tokenizer code, runtime protocol, evaluation harness, dataset
acquisition): none found -- `juniper_auto/analysis/` is new but is
explicitly probe/measurement infrastructure over the existing model, not a
tokenizer or corpus.

### Pass B -- Reference routing audit

Manually re-derived top-2 selection, renormalization, shared contribution,
routed contribution, reconstruction, and padding treatment for a 3-token,
4-expert, top-2 example by hand, then compared against
`juniper_auto/model/routing.py` and `moe_dispatch.py::reference_dispatch`
line by line -- matches. Independently, this is proven executably (not
just asserted here) by
`tests/test_model_moe_dispatch.py::test_refactored_reference_dispatch_matches_phase1_golden_bit_for_bit`,
which loads `juniper_auto/model/moe.py` directly from the
`phase-1-architecture` git tag (not from this working tree) via `git show`
and asserts `torch.equal` on output, both raw losses, and every diagnostic
field, across 25 parametrized (seed, shape, padding) cases plus a
gradient-equivalence case. This is the actual audit evidence; the by-hand
trace above is a sanity check on top of it, not a substitute.

### Pass C -- Optimized-equivalence audit

Explicitly checked for the risk named in the Phase 2 instructions: since
`routing.py` (router logits/softmax/top-k) is shared code between
`reference_dispatch` and `optimized_dispatch`, a bug in `routing.py` would
affect both backends identically and would **not** be caught by
`tests/test_model_moe_dispatch.py::test_reference_and_optimized_forward_agree`
(both backends would still "agree" with each other while both being
wrong). This is a real structural blind spot in reference-vs-optimized
comparison alone. Mitigation already in place: the golden-vs-Phase-1
comparison (Pass B) reads its router math from the immutable
`phase-1-architecture` git tag's `moe.py`, which has its *own*, textually
independent copy of the router formula (not a call into this working
tree's `routing.py`) -- so a `routing.py` regression that the
reference-vs-optimized test would miss is still caught by the golden
comparison, because the golden comparison's router math cannot share a
bug with `routing.py` by construction. Also checked: the equivalence
tests compare `out_ref`/`out_opt` directly (real output tensors), not
just derived diagnostics; and backward equivalence
(`test_reference_and_optimized_backward_agree`) is present, not just
forward. Tolerances (`atol=rtol=1e-5` CPU, `atol=rtol=2e-2` CUDA FP16) were
checked against `tests/test_model_fault_injection.py::test_reference_optimized_mismatch_is_not_hidden_by_an_overly_broad_tolerance`,
which proves a real induced bug is NOT hidden by the chosen tolerance
(only an artificially broad one would hide it).

### Pass D -- Dropless audit

Constructed and ran (via `tests/test_model_moe_property.py`) adversarial
padding layouts: none, trailing, leading, scattered, per-row-different-
lengths, almost-all-padding, and fully-padded-row, crossed with 6 seeds x
3 batch sizes x 3 sequence lengths x 3 expert counts x 3 top_k values
(212 cases after deterministic subsampling). Every case proves
`sum(assignment_counts) == n_valid * top_k` exactly, every token's top-k
set has no duplicate expert, weights sum to 1, and expert-pair
co-activation totals `n_valid * C(top_k, 2)` exactly. No case revealed a
violation.

### Pass E -- Diagnostics audit

Manually rederived entropy (uniform 4-way distribution -> `ln(4)`,
matches `test_entropy_matches_hand_computed_value_for_uniform_and_peaked_distributions`),
margins (0.7/0.2/0.1 -> prob margin 0.5, logits 5/2/-1 -> logit margin 3.0,
both matching their tests), pair co-activation (3 tokens, top-2, one
padding -> exactly 2 counted at [0,1], matching
`test_expert_pair_coactivation_matches_hand_computed_matrix`), and
contribution norms (constant-output stub experts with known L2 norms,
matching `test_contribution_norms_match_hand_computed_values_with_stub_experts`)
by hand against the actual test assertions -- all matched. **Finding
(fixed during this review):** no test exercised diagnostics under
deliberately large-magnitude router inputs/logits, as Phase 2 instructions
section 17 explicitly requires ("Deliberately test large-magnitude router
inputs/logits. Diagnostics intended to detect saturation should remain
numerically stable."). Added
`tests/test_model_moe_diagnostics.py::test_diagnostics_remain_finite_and_stable_under_large_magnitude_router_inputs`
(router weights at std=50, hidden states scaled x1000 -- confirms every
diagnostic field stays finite and that the saturation detector is
genuinely reachable from a real forward pass, not just from hand-fed
scalars) and `..._under_large_magnitude_inputs_with_padding`. Both pass.

### Pass F -- Context-sensitivity audit

Verified `compare_routing_across_variants` genuinely distinguishes
context-independent (identical hidden states -> all metrics exactly 0),
partially context-dependent (small perturbation -> JS divergence strictly
between the identical and strongly-perturbed cases), and strongly
context-dependent (large orthogonal perturbations -> top1_change_rate >
0.5) regimes, via
`tests/test_context_sensitivity.py::test_identical_hidden_states_are_context_independent`,
`test_small_perturbations_are_partially_context_dependent_between_the_two_extremes`,
`test_large_orthogonal_perturbations_are_strongly_context_dependent`. Read
every docstring in `context_sensitivity.py` and every line of
`docs/architecture/moe-routing-diagnostics.md`'s context-sensitivity
section and "What this is not" section: none claims learned/semantic
routing; `run_untrained_official_model_probe`'s docstring opens with
"ENGINEERING/PROXY TEST -- NOT SEMANTIC SPECIALIZATION EVIDENCE" and that
exact string is asserted present by
`test_untrained_official_model_probe_runs_and_is_explicitly_labeled`, so a
future edit that silently weakens the label would fail a test, not just a
manual doc review.

### Pass G -- Ablation audit

Traced all six ablation modes against constant-output stub experts with
hand-computed expected output tensors (not just "changed something"):
`disable_routed_expert` (weight preserved, contribution zeroed, no
renormalization), `zero_expert_output` (same mechanism, expert set),
`disable_shared_expert` (only the shared term drops), `replace_routed_expert`
(replacement expert's constant value used with the original weight;
separately confirmed the replaced expert's own parameters never receive a
gradient), `uniform_router` (exact round-robin indices verified per
token, `[t % E, (t+1) % E]`), `random_router` (unique experts per token,
equal weight, reproducible under a fixed seed and under two different
ambient global-RNG states). Confirmed no leakage into normal inference at
both the `MoELayer` level
(`test_ablation_state_does_not_persist_across_calls`, six modes run in
sequence then a plain call compared byte-for-byte to a pre-ablation
baseline) and the `JuniperAutoModel` level
(`validate_phase2.py::gate_diagnostics_and_ablations_smoke`).

### Pass H -- Precision and gradient audit

Re-confirmed FP32 router logits/probs hold under FP16 module weights, CPU
bf16 autocast, and real CUDA FP16 autocast (`tests/test_model_moe.py`,
unmodified from Phase 1, still passing; `test_model_moe_dispatch.py`'s
real-CUDA equivalence test). Confirmed finite gradients reach router,
shared expert, and every selected routed expert
(`tests/test_model_moe.py::test_gradients_reach_router_shared_expert_and_selected_experts_only`,
unmodified), and that a fault-injection test
(`test_missing_expert_gradient_statistics_is_detected`) demonstrates a
router-only gradient check would miss a broken shared-expert backward
path -- i.e. the gradient-audit checklist genuinely needs to (and does)
cover all three parameter groups, not just the router. Large-magnitude
input finiteness: see Pass E's fix.

### Pass I -- Performance audit

Reviewed `scripts/run_phase2_experiment.py::cmd_flowbox_moe_profile`:
untimed warmup (2 iterations) before every timed block, `torch.cuda.synchronize()`
before and after each timed region, `torch.cuda.reset_peak_memory_stats()`
before each backend's timed block, `torch.cuda.empty_cache()` between
backends within a shape. Five shapes profiled, none cherry-picked --
`exp-0021`'s result field reports all five, and the optimized backend won
at every one (1.5x-2.6x), so there was no incentive to hide an unfavorable
shape and none was hidden. Labels checked: the result is explicitly
"prefill-only latency/throughput (single forward pass per iteration, no
autoregressive decode loop exists in this reference model)" -- matching
the actual measurement, not overclaiming a decode-throughput number that
was never measured.

### Pass J -- Test-the-tests audit

Grepped every new test file for `except Exception`/bare `except:` (none
found -- no broad exception swallowing), for `pytest.mark.skip(` (none
found; the only skip mechanism used is the pre-existing, clearly-reasoned
`@pytest.mark.skipif(not torch.cuda.is_available(), reason=...)` pattern,
and on this machine CUDA is available so those tests ran for real rather
than skipping). Manually reviewed each of the 23 fault-injection tests
(20 new + Pass H's large-magnitude tests are separate) for whether the
"broken" variant is actually broken relative to the real implementation
(not a broken assertion that would pass either way) -- each constructs an
explicit wrong value or wrong code path and asserts the difference from
the correct one, per the module's established mutation-testing
convention.

### Pass K -- Reproducibility and provenance audit

Confirmed `scripts/run_phase2_experiment.py::_write` refuses a dirty
working tree unless `--allow-dirty` is passed (mirrors Phase 1's
`run_phase1_experiment.py` exactly) -- exp-0015 through exp-0021 were all
generated against the exact clean commit `12e40ece3b89bd7a4877e301d380079be4793f94`
(verified: every artifact's `git_commit`/`git_worktree_clean` field
matches). Confirmed `scripts/validate_phase2.py::gate_reproducibility` and
`exp-0020` both independently prove same-seed/same-config routing is
exactly reproducible on the official architecture. Confirmed
`random_router`'s seeded reproducibility is isolated from ambient global
RNG state (Pass G). Fresh-clone recovery: see `docs/recovery/phase-2.md`.

### Pass L -- Claims, docs, manifests, Git, and CI

Grepped the full Phase 2 diff for `specializ`, `intelligent`, `autonomous`,
`trained`, `pretrained`, `deterministic`, `optimized`, `faster`,
`production`: every occurrence checked in context (see the grep output
retained in this review's working notes) -- all uses are either negated
("not evidence of... specialization"), scoped to a proven claim
(`optimized`/`faster` always paired with the exp-0021 measurement and
ADR-0009's honest default-selection reasoning, never asserted as the
production default), or refer to the unmodified-forward-call meaning of
"production" (as opposed to an ablation/diagnostic path), not a trained/
deployed product. `manifests/frozen-artifacts.yaml`'s new
`moe_routing_and_diagnostics_infrastructure` entry explicitly states
"Pending GPT-5.6 Sol independent review; no phase-2-moe approval tag
exists yet." README status update (see below) states "Phase 2 candidate
complete, pending independent review," not approved.

## Fix cycle summary

| Finding | Pass | Severity | Fix |
|---|---|---|---|
| No test exercised diagnostics under deliberately large-magnitude router inputs/logits | E | Moderate (explicit instruction requirement, not previously covered) | Added two tests confirming finiteness/stability of every diagnostic field and that the saturation detector is reachable from a real forward pass |
| `routing.py` is shared code between reference and optimized dispatch, creating a blind spot for reference-vs-optimized-only comparison | C | Structural risk, not a bug | Confirmed already mitigated by the golden-vs-Phase-1-tag comparison (independent router math); documented explicitly here and in `moe_dispatch.py`'s module docstring |

No other defects were found. All 636 tests pass after the Pass E fix
(`python -m pytest tests/ -q`).

## Reruns performed after fixes

Re-ran the full pytest suite after the Pass E additions (636 passed, up
from 634). Did not need to re-run exp-0015 through exp-0021, since the fix
added new tests rather than changing any production code path those
experiments exercise; re-verified this by re-diffing `juniper_auto/model/`
and `juniper_auto/analysis/` against the commit those experiments were run
against (`12e40ece3b89bd7a4877e301d380079be4793f94`) and confirming zero
changes outside `tests/`.

## Remaining limitations

See `docs/phases/phase-2-moe.md`'s "Accepted limitations" section --
repeated here only by reference, per governance rule against duplicating
the same claim in two places that could drift apart.

## Remote CI

Pushed candidate `63b05c5b9ec1a3eec21bf129c99b4c48d0fd0407` to
`origin/main` (confirmed present remotely via `git ls-remote`). All three
required workflows succeeded on this exact commit: Phase 0 Validation (run
33010601469, success, 2m4s), Phase 1 Validation (run 33010601512, success,
2m45s), Phase 2 Validation (run 33010601464, success, 4m9s).

## Recovery result

See `docs/recovery/phase-2.md`: two fresh-clone recovery attempts, the
first of which caught and led to fixing a real stale-manifest gap, the
second of which passed cleanly.

## Final Sonnet candidate identity

Substantive candidate: `63b05c5b9ec1a3eec21bf129c99b4c48d0fd0407`. This
self-review document's own metadata-closure commit necessarily has a
different, not-yet-knowable hash at the time this sentence was written --
resolved by the annotated `phase-2-moe` tag once GPT-5.6 Sol's independent
review approves it (Sonnet does not create that tag).

---

**SELF-REVIEW PASSED (CANDIDATE) -- INDEPENDENT REVIEW REQUIRED**
