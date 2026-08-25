# Phase 1 Sonnet Self-Review

## Candidate state reviewed

All Phase 1 work committed on top of Phase 0's approved
`f6f8b5397b41d31081f55cdd60cb6363ec052be4` (tag `phase-0-foundation`),
through the commit immediately preceding this report's own commit (see
the final handoff message for the exact hash). This review assumes the
implementation is wrong until each specific check below demonstrates
otherwise -- it is a re-inspection of the repository as delivered, not a
restatement of the engineering summary.

## Review start / end

Self-review began after the core implementation, training support,
executed experiments, and initial documentation were in place, and
proceeded as a distinct pass over the actual repository state (not the
implementer's memory of writing it). See `docs/time/phase-hours.csv` for
the self-review time row.

## Review passes and findings

### Pass A -- Requirements traceability

Cross-checked every row of
`docs/phases/phase-1-requirements-traceability.md` against the actual
files/tests/experiments it cites (not just that the citation exists, but
that the cited file actually contains what the row claims). No row was
found to be prose-only or to cite a check that doesn't exist. No Phase 2+
work (optimized MoE kernels, expert-parallelism, routing research) was
found to have crept into the implementation.

### Pass B -- Frozen architecture audit

Compared the executable model, field by field, against
`configs/architecture/ja150m-v0.1.yaml` / `-dense.yaml`, independently of
parameter-count matching (a wrong architecture can coincidentally share a
parameter count with a correct one, so this pass deliberately did not
lean on Method A/B agreement as sufficient evidence). This found a real,
repeatable class of defect and fixed it:

**Finding (MAJOR): several `ArchitectureConfig` fields were read by
`load_architecture_config`/schema validation but never actually consulted
by the model code** -- the model code hard-coded the behavior the frozen
v0.1 values happen to imply, without validating that the loaded config
actually asked for that behavior. For the frozen v0.1 configs this
produced *correct* behavior (since the hard-coded assumption matches the
frozen values), but it meant a hypothetical future config that changed
one of these fields would be silently mishandled instead of rejected --
exactly the "config can load correctly and the executable model can still
be wrong" risk this audit exists to catch. Specific fields found unread
by the code before this fix:

- `residual.scale` (never multiplied into the residual add; always
  behaved as if `scale=1.0` regardless of config)
- `residual.rezero` / `residual.deepnorm` / `residual.learned_gates`
  (never checked)
- `normalization.placement`, `normalization.attention_norm`,
  `normalization.ffn_or_moe_norm`, `normalization.final_norm` (Pre-Norm
  and both norms were unconditionally applied)
- `attention.causal` (always built a causal mask regardless of the flag)
- `attention.sliding_window` (silently ignored -- always full attention)
- `position_encoding.kind` (RoPE always used, kind never checked)
- `position_encoding.initial_scaling` (defined in schema, frozen at 1.0,
  but never read by `RotaryEmbedding` at all -- a real dead config field)
- `moe.routing_kind` (`token_choice` always assumed; `expert_choice` is a
  reachable schema value the code would have silently mishandled)
- `moe.router_logits_dtype` / `moe.router_softmax_dtype` (FP32 always
  forced regardless of what these fields said)
- `dropout.*` (no `nn.Dropout` modules exist at all, so any nonzero
  dropout config would have been silently ignored)

**Fix**: `juniper_auto/model/block.py::_validate_block_assumptions`,
`juniper_auto/model/attention.py::GroupedQueryAttention.__init__`,
`juniper_auto/model/moe.py::MoELayer.__init__`, and
`juniper_auto/model/model.py::JuniperAutoModel.__init__` now explicitly
validate every one of these fields and raise `ValueError` if a config
asks for a variant that isn't implemented, instead of silently doing
something else. `residual.scale` and `position_encoding.initial_scaling`
are now actually applied (not just validated), since both have a
well-defined, correct behavior to implement (residual scaling; RoPE
position-interpolation-style scaling) rather than only a reject-if-wrong
check. Regression coverage:
`tests/test_model_config_assumption_guards.py` (13 tests, one per
rejected variant plus two proving `residual.scale` and
`initial_scaling` are genuinely applied, not just accepted) and three new
tests in `tests/test_model_rope.py` for `initial_scaling` specifically.
Verified the frozen v0.1 configs still pass every check (no regression):
full suite rerun, 258/258 passing, official parameter counts unchanged
(150,031,360 / 79,191,040).

**Finding (MINOR): dead code.** `MoELayer.__init__` set
`self.load_balance_coefficient` / `self.router_z_coefficient` from the
config, but nothing ever read them -- coefficient weighting actually
happens once, at the model level
(`JuniperAutoModel.load_balance_coefficient` /
`.router_z_coefficient`), which is the only place `ModelOutput` needs
them. Removed the unused per-layer copies.

No other field-by-field mismatch was found: `d_model`, `n_layers`,
`dense_layers`/`moe_layers` partition, GQA head counts/`head_dim`,
`attention_scale`, `qk_norm`/`qk_norm_placement`/`qk_norm_kind`, RoPE
`theta`/`rotary_dim`, FFN dims, expert counts/`top_k`, shared-expert
gating, router bias, `expert_output_combination`, aux-loss coefficients,
embedding `vocab_size`/`dim`/`tie_lm_head`/`output_bias`/
`embedding_scale`/`logit_softcap`, and initialization stds all matched
the frozen spec on direct comparison, independent of the parameter-count
match.

### Pass C -- Mathematical primitive audit

Manually re-derived, on paper and against small hand-computable tensors
(not just re-running the existing test suite), the causal cross-entropy
shift, the load-balance and router Z-loss formulas (ADR-0008), the RoPE
rotate-half construction, and the GQA head-repetition mapping. All
matched. Specifically checked for the failure modes the instructions call
out: normalization axis (confirmed last-dim only, both for RMSNorm and
per-head QK-Norm, via `tests/test_model_norm.py::test_per_head_qk_norm_independent_per_head_shared_scale`),
accidental FP16 reduction (confirmed FP32 throughout via
`test_rmsnorm_fp32_reduction_under_fp16_input` and the MoE router FP32
tests, including under real CUDA FP16 autocast during exp-0005), Q/K
reshape and KV-repetition order (confirmed via
`tests/test_model_attention.py::test_repeat_kv_maps_each_kv_head_to_contiguous_query_heads`
and the independent manual-reference attention test), and the
causal-loss shift direction (confirmed via a construction specifically
designed to fail loudly under an unshifted comparison,
`test_causal_lm_loss_off_by_one_shift_is_load_bearing`).

A real bug was found and fixed here during initial implementation (not
during this later review pass, but recorded for completeness since it is
exactly the class of defect this pass exists to catch): the MoE
combination weight was originally pre-cast once to the input activation's
dtype before the expert loop; under CPU bfloat16 autocast, an expert's
Linear layers executed in bfloat16 while the pre-cast weight remained the
input's original dtype, causing a dtype mismatch in `index_add_`. Fixed
by casting the weight to each expert output's *actual* dtype at the point
of multiplication (see `juniper_auto/model/moe.py`); verified directly
under both CPU bfloat16 autocast and real CUDA FP16 autocast (the actual
training precision path), not merely documented as fixed.

### Pass D -- MoE routing audit

Traced individual tokens through a diagnostic-enabled MoE forward pass
(`MoEDiagnostics`) confirming, for each valid token: exactly 2 unique
routed-expert indices, renormalized weights summing to 1, the shared
expert's unconditional contribution, and correct reconstruction to the
token's original position (`tests/test_model_moe.py::test_token_order_is_preserved_in_reconstruction`,
verified by permuting the input and checking the output permutes
identically). Verified padding tokens are excluded from
aux-loss/diagnostic statistics but still produce finite output (no
cross-token corruption is possible since MoE has no cross-token mixing).
Verified no hidden averaging via a controlled-stub test
(`test_shared_plus_topk_routed_combination_with_no_hidden_averaging`)
where any division-by-count bug would have failed the exact expected
value.

### Pass E -- Gradient and training audit

Re-ran both official tiny-overfit experiments (exp-0004, exp-0005) after
the Pass B fixes above, since those fixes touch code paths (`residual.py`
scale application, MoE validation) that the original experiment runs
predate. Results after rerun: dense LM loss 10.60 -> 0.0012 (100% token
accuracy), sparse LM loss 10.61 -> 0.0003 (100% token accuracy), both
zero non-finite events -- unchanged in substance from the pre-fix runs
(as expected, since `residual.scale=1.0` for the frozen configs, so the
fix is a no-op numerically here; the fix only *matters* for a
hypothetical non-1.0-scale config). Confirmed router/shared-expert/
selected-routed-expert gradients are present and finite
(`tests/test_model_gradients.py`), and that an expert never selected in a
given batch is correctly not required to have a gradient (not a bug, a
property of sparse dispatch).

### Pass F -- Checkpoint and resume audit

Deliberately tested checkpoint rejection of a payload missing
`rng_state`, missing `sampler_state`, declaring an unsupported format
version, and declaring the wrong `architecture_id`
(`tests/test_training_checkpoint.py`) -- all four are rejected at
validation time with a specific error, not a generic crash. Re-ran the
interrupted/resumed equivalence experiment (exp-0006) after the Pass B
fixes; still bit-exact (loss, parameters, AdamW state) on the full
150M-parameter sparse model. Confirmed the negative controls in
`tests/test_training_resume_equivalence.py` still correctly diverge (a
same-seed-but-no-checkpoint-load harness, and a harness that skips
sampler-state restoration), proving the equivalence result is load-bearing
and not a vacuous same-seed coincidence.

### Pass G -- Hardware and consumer-practicality audit

**Finding (MAJOR, already fixed during initial engineering, re-verified
here): a CUDA OOM in the FLOWBOX sparse hardware profile.** Root-caused to
`scripts/run_phase1_experiment.py` not releasing the training-step
model/optimizer's GPU memory before constructing a second full model for
the checkpoint-I/O probe -- a profiling-script bug, not an architecture or
hardware limitation. This review re-ran both profiles from a clean
process to confirm the fix holds (exp-0007, exp-0008 both completed
cleanly; dense 4,096-token batch-1 probe dropped from an earlier
contaminated reading of 3.66 GB peak VRAM to a clean 1.09 GB once the
memory leak between stages was fixed, confirming the contamination
diagnosis). Checked CUDA synchronization placement in
`juniper_auto/training/profiling.py`: synchronized immediately before
starting and immediately after stopping every timed section, with
untimed warmup iterations before any measurement, so steady-state timing
is not contaminated by one-time CUDA context/kernel-compilation cost. No
undocumented system CUDA toolkit dependency was introduced (only
`torch`'s bundled CUDA runtime is used, consistent with Phase 0's
recovery documentation). The reference MoE dispatch's near-zero FP16
inference speedup (exp-0008) was investigated and attributed to
kernel-launch/Python-loop overhead dominating at batch=1 -- not
re-profiled with a larger batch to "fix" the number, since the honest
finding for the reference implementation at this batch size is more
valuable than a cherry-picked configuration.

### Pass H -- Test the tests (fault injection)

`tests/test_model_fault_injection.py` (15 tests) and
`tests/test_training_checkpoint.py`'s validation-rejection tests (2 more)
together cover every failure mode listed in the Phase 1 instructions'
"test the tests" requirement: future-token leakage, top-1 routing, top-3
routing, token dropping, missing shared expert, hidden MoE-output
averaging, broken top-2 renormalization, untied LM head, unintended
projection bias, wrong dense/MoE layer positions, wrong QK-Norm parameter
shape, FP16 router execution, incorrect RoPE theta, unshifted causal
loss, padding leaking into router statistics, missing RNG state from a
checkpoint, and missing sampler state from a checkpoint. Each test
constructs or monkeypatches a locally-scoped broken variant (never
mutating the real implementation beyond the test body) and asserts the
specific property that would have caught the mutation actually fails to
hold. Searched the test suite for weak patterns called out by the
instructions (`assert True`, tests that instantiate but assert nothing,
broad exception swallowing, unexplained skips): none found. The only
`pytest.mark.parametrize`/conditional-skip usage in the Phase 1 suite is
ordinary shape/config parametrization, not a disguised way to avoid
testing something.

### Pass I -- Serialization and recovery audit

Confirmed weight tying survives a `state_dict` save/load round trip via
object-identity re-verification after load, not just value equality
(`tests/test_model_serialization.py`). Scanned every new Phase 1 file for
absolute host-specific home-directory paths: none found outside
the already-permitted Phase 0 exceptions. Confirmed no hidden local
files, cache dependence, or undocumented environment-variable requirement
was introduced by the training/profiling code (it only touches
`REPO_ROOT`-relative paths and explicit function arguments). The genuine
fresh-clone recovery exercise for this candidate is recorded in
`docs/recovery/phase-1.md`.

### Pass J -- Security and repository hygiene

Re-ran Phase 0's `gate_repository_integrity` (credential-pattern scan,
prohibited file types, oversized-file check, absolute-path check) via
`scripts/validate_phase1.py`'s first gate. Manually confirmed no
checkpoint file (`.pt`) was ever committed -- `scripts/run_phase1_experiment.py`'s
`profile` subcommand writes a checkpoint probe file only to measure I/O
timing and deletes it (`ckpt_path.unlink(missing_ok=True)`) before the
process exits; verified no `.pt` file exists anywhere in the tracked or
untracked working tree. All `docs/experiments/results/*.json` files are
metrics/metadata only (12-96 KB each), never a weight dump.

### Pass K -- Scope and claim honesty

Grepped the full diff against Phase 0 for `pretrained`, `intelligent`,
`autonomous`, `expert specialization`, `self-improving`,
`production-ready`, `release checkpoint`, and related terms. Every match
found was either inherited unmodified Phase 0 research-question framing
(README's opening research question, which asks what the project *could*
become, not what it has become) or an explicit negation ("does not have
measured expert specialization", "not a prior release checkpoint -- none
exists yet"). No claim of trained capability, autonomy, or intelligence
was found anywhere in the Phase 1 additions.

### Pass L -- Git, CI, manifests, and reporting

Verified after the fixes above: working tree clean before the final
candidate commit (see `docs/recovery/phase-1.md`), Phase 1 artifact
hashes and test manifest regenerated after the Pass B/C fixes (not left
stale), `experiments/registry.yaml` and `docs/time/phase-hours.csv`
current, this report and the phase report current, and no failed/
negative-result experiment entry was deleted (exp-0008's original OOM
run is documented in its `result` field, not erased).

## Fix cycle summary

| Defect | Severity | Fixed | Regression test |
|---|---|---|---|
| 10 silently-unvalidated config fields (Pass B) | MAJOR | Yes | `tests/test_model_config_assumption_guards.py`, `tests/test_model_rope.py` (initial_scaling) |
| MoE combination-weight dtype mismatch under autocast (Pass C, found during implementation) | BLOCKING (would break real FP16-autocast training) | Yes | `tests/test_model_moe.py::test_router_logits_and_softmax_are_fp32_under_cpu_autocast_bf16`, verified live under real CUDA FP16 autocast |
| Profiling-script GPU memory leak causing false OOM (Pass G, found during implementation) | MAJOR (would misrepresent a hardware limitation) | Yes | Rerun to a clean result; documented in exp-0008 rather than deleted |
| Dead `MoELayer` coefficient attributes (Pass B) | MINOR | Yes | -- (removal verified by full suite still passing) |

No unresolved BLOCKING or MAJOR defect remains as of this report.

## Reruns performed after fixes

Full `pytest tests/ -q`: 258/258 passing (up from 214 before this
review's Pass B fixes and their regression tests). Official parameter
counts re-verified unchanged (150,031,360 sparse / 79,252,480 active /
79,191,040 dense). Both tiny-overfit experiments and the checkpoint/
resume equivalence experiment were re-run after the Pass B/E fixes;
results unchanged in substance (as expected for `residual.scale=1.0`).
Both FLOWBOX hardware profiles were re-run from a clean process after the
Pass G fix.

Note on experiment metadata: the `git_commit` field recorded inside each
`docs/experiments/results/exp-000N-*.json` file reflects `git rev-parse
HEAD` at the moment that experiment script ran, which for the Pass-B/C
reruns above is the pre-self-review commit
(`241295a2fc99a0289bace5ca734d600090a84f53`) -- these reruns executed
against the actual working-tree code (including the Pass B/C fixes),
which had not yet been committed at run time. This is recorded here
explicitly rather than silently left inconsistent: the numerical results
are accurate for the code that produced them, but that exact commit hash
was never pushed on its own -- the final candidate commit (see the final
handoff message) is the first commit that actually contains this code,
and reproducing these experiments against that commit is expected to
reproduce the same results (confirmed identical across the pre- and
post-fix reruns here, since `residual.scale=1.0` and the other Pass B
fixes are behavioral no-ops for the frozen v0.1 configs).

## Remaining limitations

See `docs/phases/phase-1-architecture.md`'s "Accepted limitations"
section (CUDA determinism not claimed; no sequence-length curriculum;
16K context not validated; reference MoE dispatch not optimized;
tokenizer/corpus/checkpoint remain not-yet-created). None of these are
newly discovered by this self-review; all were already honestly recorded
during implementation.

## Remote CI

`.github/workflows/phase-1-validation.yml` -- exact run ID for the final
candidate commit recorded in the final handoff message.

## Recovery result

See `docs/recovery/phase-1.md`.

## Final Sonnet candidate identity

See the final handoff message at the end of this Phase 1 engineering
session.

---

**SELF-REVIEW PASSED - INDEPENDENT REVIEW REQUIRED**
