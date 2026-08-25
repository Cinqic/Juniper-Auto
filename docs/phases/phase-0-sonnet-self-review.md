# Phase 0 Sonnet Self-Review

**SELF-REVIEW PASSED -- INDEPENDENT REVIEW STILL REQUIRED.**

This is Claude Sonnet 5's self-review of its own Phase 0 implementation
work. This is **not** independent review. Independent review is GPT-5.6
Sol's job and has not happened yet -- see
[docs/phases/phase-0-foundation.md](phase-0-foundation.md), Approval
status.

## Candidate commit reviewed

Working state as of this review: local commits `232b775` and `89bbb76` on
top of the pre-existing `a991db7`, plus the fix-cycle changes described
below, committed as the final candidate at `PENDING_CANDIDATE_COMMIT`
(filled in immediately before push -- see final commit hash in
`docs/phases/phase-0-foundation.md`).

## Date

2026-08-25.

## Review duration

Self-review was interleaved with the tail of the engineering session rather
than run as a fully separate block of time; see
[docs/time/phase-hours.csv](../time/phase-hours.csv) for the best-available
split between the `phase-0` engineering row and the `phase-0` self-review
row. This is recorded as approximate, per governance rule 6 and rule 14
(truth over confidence) -- a fabricated precise split would be less honest
than an approximate one.

## Checks performed

### Pass A -- Requirements traceability

Built an explicit mapping from every numbered step in the Phase 0
instructions (Steps 1-30 and their sub-items) to the file, test, manifest
entry, or CI check that satisfies it. Two design choices worth recording
explicitly (not defects, but decisions a reviewer should be able to see):

- "Incorrect vocabulary size" (Step 7's config-validation requirement) is
  caught at the **frozen-value layer** (`juniper_auto/config/frozen.py`,
  tested in `tests/test_architecture_constants.py`), not the general
  Pydantic schema layer. The general schema accepts any positive
  `vocab_size` for forward-compatibility with future architecture
  versions; only a config claiming to *be* `ja150m-v0.1` is held to the
  exact 36,864 value. See
  [ADR-0003](../adr/0003-configuration-format-and-validation.md).
- `checkpoints/` and `releases/` (listed in the Phase 0 instructions'
  recommended repository structure) were not created as tracked
  directories. Git does not track empty directories, and both are
  `.gitignore`d as disposable/local (transient checkpoints, model weights),
  so creating them empty now would only produce meaningless placeholders.
  They will appear automatically when the first real artifact lands in
  them, in a later phase.

Two real gaps were found by this pass and fixed during the self-review fix
cycle (see below): `juniper_auto/util/logging.py` was only import-checked,
never behavior-tested, and `FoundationProbe` did not actually call it
despite the Phase 0 spec requiring the probe to prove "logging functions."

### Pass B -- Architecture audit

Independently recomputed every parameter count by hand (not by re-running
the same code) for both `ja150m-v0.1` and `ja150m-v0.1-dense`, cross-checked
against `juniper_auto/accounting/parameter_count.py`'s output and the exact
component-level figures given in the Phase 0 instructions (embeddings
18,874,368; attention 13,107,200; five dense FFNs 11,796,480; routed
experts 94,371,840; shared experts 11,796,480; routers 61,440; QK norms
2,560; transformer RMSNorms 20,480; final RMSNorm 512; total 150,031,360;
standard active 79,252,480; dense control 79,191,040). All matched exactly.
Confirmed 20 layers, dense anchors at 1/5/10/15/20, exactly 15 MoE layers,
8 Q heads, 2 KV heads, 64-dim heads, 1536 dense FFN, 8 routed + 1 shared
expert, top-2, expert dim 512, 36,864 vocabulary, dropless routing (no
token dropping), and all normalization/positional settings against both
the YAML configs and the frozen-value module -- all consistent.

### Pass C -- Reproducibility audit

Grepped the codebase for `os.environ`, `getenv`, `os.getcwd`, `Path.home`,
and `expanduser` usage. Findings: `juniper_auto/util/seed.py` sets
`CUBLAS_WORKSPACE_CONFIG` via `setdefault` (a value the seed module sets
itself, not a hidden precondition on the caller); `Path.home()` is used
only inside the absolute-path *detector* (`scripts/validate_repo.py`,
`tests/test_absolute_paths.py`), i.e. as part of checking for the
violation, not as a runtime dependency. No hidden required environment
variables, no dependency on an existing `.venv`, no dependency on globally
installed packages, no shell aliases referenced. This was additionally
verified empirically by the fresh-clone recovery exercise (see below),
which used a virtual environment and cache state independent of the
primary working checkout.

### Pass D -- Git audit

`git status`, `git diff`, untracked-file listing, remote URL, branch, and
HEAD-vs-`origin/main` relationship were all checked (see
[docs/phases/phase-0-foundation.md](phase-0-foundation.md) for the recorded
starting and final commits). Remote is confirmed as
`https://github.com/Cinqic/Juniper-Auto.git`; branch is `main`. Working
tree was brought to clean before the candidate push (verified as the last
step of the fix cycle, below).

### Pass E -- Security and repository hygiene

Scanned all `git ls-files`-tracked files for common secret patterns (AWS
access key IDs, PEM private key headers, GitHub `ghp_`/`gho_` tokens,
OpenAI-style `sk-` keys): no matches. Confirmed no `.env`, no SSH key
material, no downloaded data, and no checkpoints are tracked (also enforced
mechanically by `tests/test_repository_integrity.py`, including a new
1&nbsp;MB tracked-file-size safety net added during this review -- see Pass
G).

### Pass F -- Claim honesty

Grepped `README.md`, `docs/phases/phase-0-foundation.md`, and
`docs/research/project-charter.md` for overclaiming language ("trained on",
"has learned", "beats the dense", "autonomous execution", "persistent
memory", "self-improv[ing/ed]", "native vision/audio", "16K context"). Every
match was in an honest context: either part of the *research question*
phrasing (the charter's primary question explicitly asks how
self-improving the model *could become*, not that it already has), an
explicit negation ("has not been trained on any tokens"), or inside the
v0.1 non-goals list. No exaggerated capability claims found.

### Pass G -- Test the tests

Checked for `assert True`, bare `except: pass`, and unexplained
`pytest.mark.skip` across `tests/` -- none found. Checked assertion density
per test file to catch files that exist but don't actually check anything
meaningful -- all files have multiple substantive assertions (see per-file
counts recorded during this review; the thinnest file,
`test_absolute_paths.py`, still has 4 real assertions across 2 tests, both
of which fail meaningfully if the underlying detector regresses).

This pass is also what surfaced the logging gap in Pass A: importing a
module is not the same as testing its behavior.
`tests/test_logging.py` was added specifically to close this gap, including
an end-to-end assertion that `run_foundation_probe` actually emits
`foundation_probe.start` and `foundation_probe.complete` structured log
events, not just that `juniper_auto.util.logging` can be imported.

Also removed `juniper_auto.util.environment.is_ci()`: it was defined,
exported, and had zero callers anywhere in the codebase -- speculative code
with no test exercising real behavior, which is exactly the kind of thing
this pass exists to catch (dead code cannot be meaningfully tested, and an
unused "future convenience" function is scope the project doesn't
currently need).

### Pass H -- Recovery audit

The clean-clone recovery exercise (Step 25) was executed once, then
re-verified after the fix-cycle changes described below, since
`juniper_auto/foundation/probe.py` (a reproducibility-relevant module) was
modified after the first run. See "Recovery result" below for the final,
post-fix result. `docs/recovery/README.md` was checked line-by-line against
the commands actually run in both recovery exercises; no undocumented step
was required either time.

### Pass I -- Scope audit

Grepped for `class.*Transformer`, `class.*Attention`, `class.*MoE`,
`RotaryEmbedding`, and `def forward` across `juniper_auto/`. The only
matches are `AttentionConfig` (a Pydantic *config schema* class, not an
attention mechanism) and `FoundationProbe.forward` (the deliberately
model-unrelated two-linear-layer smoke test, explicitly documented as not
the Juniper Auto model). No tokenizer training code, no data acquisition
code, no runtime execution engine, and no training loop exist anywhere in
the repository. Also grepped for `TODO`/`FIXME`/`XXX` across all tracked
source and doc files: zero matches, so there is no unexplained deferred
work hiding in a comment.

### Pass J -- Failure and limitation audit

See "Remaining limitations" below. The most important finding from this
pass: the Phase 0 task instructions repeatedly refer to "the complete
Juniper Auto End-to-End Development Instructions supplied with this task"
and "capabilities listed in the Phase 0 specification" as authoritative
source material for the charter's v0.1 model/runtime scope sections. No
such separate document was actually provided beyond the Phase 0 task
description itself. Rather than inventing a plausible-sounding capability
list to fill that gap, the charter's v0.1 model/runtime scope sections were
written narrowly, grounded only in what *was* explicitly given (the frozen
architecture values, the explicit non-goals list, and the governance
rules) -- and this gap is now recorded here and in the phase report rather
than silently absorbed.

## Self-review fix cycle

Two rounds of fixes were made during self-review, each followed by a full
re-run of the test suite and `scripts/validate_repo.py --all`:

1. **Logging not exercised** (Pass A/G): added `tests/test_logging.py`
   (behavioral tests: JSON structure, required fields, `LogContext`
   None-omission, handler-dedup, and an end-to-end check that
   `run_foundation_probe` emits real log events), and wired
   `juniper_auto/foundation/probe.py` to actually call
   `juniper_auto.util.logging.log_event` at the start and end of a probe
   run.
2. **Missing coverage** (Pass A): added `tests/test_experiment_registry.py`
   (schema completeness, unique ids, valid status values, no fabricated
   tokenizer/dataset/checkpoint identities) and a tracked-file-size safety
   net (`test_no_unexpectedly_large_tracked_files`, 1&nbsp;MB threshold) in
   `tests/test_repository_integrity.py`.
3. **Dead code** (Pass G/I): removed the unused
   `juniper_auto.util.environment.is_ci()` function and its export.

After these fixes: **80 tests pass** (up from 69), and all 10
`scripts/validate_repo.py --all` gates pass.

## Recovery result (post-fix)

Re-ran the isolated fresh-clone/fresh-venv recovery exercise after the
fix-cycle changes above (since `foundation/probe.py` changed). Same
procedure as the first run (see
[docs/phases/phase-0-foundation.md](phase-0-foundation.md), "Recovery
status"): fresh `git clone` into an isolated scratch directory, fresh
`venv`, `pip install -r requirements-lock.txt`, `pip install -e .
--no-deps`, `python scripts/validate_repo.py --all`. Result: **all gates
passed**, 80/80 tests passed, foundation-probe checksum identical to the
primary checkout's value under the same seed.

## Final local validation result

`python scripts/validate_repo.py --all`: **all 10 gates passed** (environment
sanity, imports, config validation, parameter accounting, frozen artifact
manifest, time-accounting schema, repository integrity, artifact hashes,
deterministic foundation probe, pytest suite -- 80 tests).

## Final remote CI result

Recorded in [docs/phases/phase-0-foundation.md](phase-0-foundation.md), "CI
workflow / run", after the candidate commit is pushed.

## Remaining limitations

- **No separate authoritative specification document was available beyond
  the Phase 0 task instructions themselves** (see Pass J above). The
  charter and governance documents are Sonnet's best-faith operationalization
  of what was actually provided, not a transcription of a fuller external
  document the task instructions reference but that was never supplied to
  this session. A future reviewer with access to that document should
  diff it against `docs/research/project-charter.md` and
  `docs/research/project-governance.md`.
- License selection beyond the pre-existing MIT `LICENSE` remains an open
  governance item, deliberately not decided here.
- CUDA/GPU bitwise determinism is not claimed or tested -- only best-effort
  CPU determinism.
- The recovery exercise is an isolated fresh-clone/fresh-venv test (twice)
  plus clean GitHub Actions CI, not a literal from-scratch OS
  reinstallation of the physical development machine.
- No linter (e.g. ruff/flake8) was adopted for Phase 0, to keep the
  dependency surface minimal per
  [ADR-0004](../adr/0004-dependency-locking-approach.md); CI's "static
  checks where adopted" gate is currently limited to a lock-file/
  `pyproject.toml` consistency check.
- `docs/adr/`, `experiments/registry.yaml`, and time-accounting values with
  `PENDING_CANDIDATE_COMMIT`/similar placeholders are updated with real
  values as the very last step before push -- see
  [docs/phases/phase-0-foundation.md](phase-0-foundation.md) for the final
  values.

## Final candidate commit

See [docs/phases/phase-0-foundation.md](phase-0-foundation.md), "Final
commit".

---

**SELF-REVIEW PASSED -- INDEPENDENT REVIEW STILL REQUIRED.** This document
and its findings are Sonnet's own assessment of Sonnet's own work; they are
not, and do not substitute for, GPT-5.6 Sol's independent review.
