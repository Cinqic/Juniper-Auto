# Phase 2 Recovery Exercise

This documents an actual fresh-clone/fresh-venv recovery exercise
performed for the Phase 2 candidate (not a plausibility claim -- see
[project-governance](../research/project-governance.md) rule 15,
executed-action truthfulness). It extends
[docs/recovery/README.md](README.md) and
[docs/recovery/phase-1.md](phase-1.md) with the Phase 2-specific
validation step.

**Note on clone source:** at the time this exercise was run, the Phase 2
candidate had been committed locally but push to `origin/main` was
explicitly deferred pending the user's confirmation (a hard-to-reverse,
shared-state action per this session's operating rules). Both recovery
attempts below therefore clone from this machine's local repository
(`git clone /home/.../Juniper-Auto <fresh-dir>`) rather than from
`https://github.com/Cinqic/Juniper-Auto.git` -- `git clone` against a
local path still produces a genuinely independent working tree and
`.git` history sourced entirely from committed refs (never a copy of the
working directory's uncommitted state), so it exercises the same
"does the committed repository actually contain everything it needs"
property a remote clone does. A third recovery pass against the real
GitHub remote, after push, is recorded in this phase's metadata-closure
commit once push is authorized.

## Procedure

```bash
# 1. A genuinely fresh directory, not a copy of the working checkout.
mkdir -p <scratch>/phase2-recovery-N && cd <scratch>/phase2-recovery-N

# 2. Clone at the exact candidate commit (local repo path; see note above).
git clone <local-repo-path> .
git log --oneline -1   # confirm the expected commit

# 3. A new Python 3.12 virtual environment.
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 4. Install solely from the committed, hash-pinned dependency lock.
pip install -r requirements-lock.txt
pip install -e . --no-deps

# 5. Phase 0 validation.
python scripts/validate_repo.py --all

# 6. Phase 1 validation.
python scripts/validate_phase1.py --all

# 7. Phase 2 validation (frozen-architecture/parameter-count checks,
#    reference/optimized MoE equivalence, dropless invariants,
#    diagnostics/ablation/context-sensitivity smoke, reproducibility,
#    full pytest suite, documentation/hash/experiment/time-accounting
#    checks).
python scripts/validate_phase2.py --all

# 8. A small reference-vs-optimized comparison and routing trace, to
#    confirm the experiment runner itself works from the clean clone.
python scripts/run_phase2_experiment.py equivalence --output /tmp/recovery-equivalence.json --n-cases 5
python scripts/run_phase2_experiment.py routing-trace --output /tmp/recovery-trace.json
```

## Result

Two recovery attempts were run, into two separate genuinely fresh
directories.

### Attempt 1: FAILED (caught a real, pre-existing gap; fixed)

Commit tested: `954675802a9d0a761f4c3fdcaccf892df07939df` (the Phase 2
documentation/experiments/validator/CI commit, before the fix below).

Steps 1-5 (fresh clone, fresh venv, install, Phase 0 validation) passed
cleanly: 636 tests, `FoundationProbe deterministic under seed=1234 on cpu:
checksum=-42.402915954589844`, all 11 Phase 0 gates green.

Step 6 (Phase 1 validation) **failed** at the `phase 1 artifact hashes`
gate:

```
=== [phase 1 artifact hashes] ===

FAILED at gate 'phase 1 artifact hashes': manifests/phase-1-test-manifest.yaml is stale relative to the actual test files
```

Root cause: `tests/test_model_fault_injection.py` is listed in both
`PHASE_1_TEST_FILES` (it originated in Phase 1) and `PHASE_2_TEST_FILES`
(Phase 2 legitimately extended it with 20 new fault-injection tests, per
Phase 2 instructions section 19). Extending the file changed its hash,
which staled `manifests/phase-1-test-manifest.yaml` -- a real,
previously-unnoticed consequence of Phase 2 evolving a Phase-1-owned
artifact, caught only by actually running the recovery exercise rather
than by the working-tree pytest/validator runs performed during
engineering (which never re-checked Phase 1's *own* hash manifest against
Phase 2's edits, since `scripts/validate_phase2.py`'s
`gate_phase1_baseline` step does call `validate_phase1.py --all`, but that
had not yet been run end-to-end against the actual committed candidate
until this recovery exercise).

Fix: ran `python scripts/generate_phase1_test_manifest.py` followed by
`python scripts/hash_manifest.py --phase 1` in the main working directory
(not the recovery clone) to regenerate both `manifests/phase-1-test-manifest.yaml`
and `manifests/phase-1-artifact-hashes.yaml` against the current, Phase-2-
extended content of the shared files. This is consistent with
`docs/adr/0009-moe-dispatch-backend-selection.md`'s framing: Phase 1's
*actual* immutable evidence is the `phase-1-architecture` git tag
(unchanged, still resolves to `073acf46e04241ed35d00bc4b4c29ac463ee744d`),
not this mutable hash manifest, which is a drift-detection convenience
for the current working tree and is expected to move forward when a later
phase legitimately (and, per governance, deliberately/reviewably) extends
a file it lists.

### Attempt 2: PASSED

Commit tested: the follow-up commit containing the regenerated
`manifests/phase-1-test-manifest.yaml`, `manifests/phase-1-artifact-hashes.yaml`,
`manifests/phase-2-test-manifest.yaml`, `manifests/phase-2-artifact-hashes.yaml`,
and this document.

- Fresh clone, fresh venv, install: clean.
- `python scripts/validate_repo.py --all`: **all 11 gates passed.**
- `python scripts/validate_phase1.py --all`: **all 9 gates passed**
  (including the artifact-hashes gate this time).
- `python scripts/validate_phase2.py --all`: **all 13 gates passed** --
  frozen architecture/parameter counts unchanged, reference backend
  default confirmed, reference/optimized equivalence on the official
  architecture (routing identical, max abs output diff far inside
  tolerance), dropless invariants, diagnostics/trace/ablation smoke (no
  leakage into normal inference), context-sensitivity infrastructure
  smoke (labeling intact), reproducibility, full pytest suite (636
  passed), required documentation present, Phase 2 artifact/test hashes
  verified, and at least 7 phase-2 experiment registry entries plus a
  phase-2 time-accounting row found.
- `run_phase2_experiment.py equivalence --n-cases 5` and `routing-trace`:
  both ran successfully from the clean clone and produced
  `gate_passed: true` results, confirming the experiment runner itself
  (not just the pytest suite) works end to end from a fresh install.

## Remote (GitHub) recovery and CI verification

Push to `origin/main` was authorized and performed for commit
`63b05c5b9ec1a3eec21bf129c99b4c48d0fd0407`; `git ls-remote origin main`
confirmed the remote ref matches exactly. All three required GitHub
Actions workflows ran on this exact commit and succeeded:

| Workflow | Run ID | Conclusion | Duration |
|---|---|---|---|
| Phase 0 Validation | 33010601469 | success | 2m4s |
| Phase 1 Validation | 33010601512 | success | 2m45s |
| Phase 2 Validation | 33010601464 | success | 4m9s |

Independent review found that the historical workflows used the checkout
action's shallow, tagless default. Consequently the Phase 1 golden
comparison silently skipped even though the run was green. These run IDs are
retained as historical evidence, but they are not accepted as proof of the
advertised golden comparison. The repaired workflow fetches complete history
and tags, and both the validator and golden test now fail if the approved tag
is missing or moved. The first independent-review push proved why this must
apply to every workflow: Phase 0 run `33232007878` and Phase 1 run
`33232007877` failed closed because those inherited baseline workflows still
lacked the tag, while Phase 2 run `33232007902` used the repaired checkout.
All three workflow files were then aligned and protected by a regression test.

## Independent-review recovery: PASSED

Commit tested: `af633d4aa5bfa18cff71393ce54445e544b5beb2`, containing the
substantive repair commit plus the independent report and canonical exp-0022/
exp-0023 artifacts.

On 2026-08-28 GPT-5.6 Sol created `/tmp/juniper-phase2-recovery-EuUDnn`
with `mktemp`, cloned the committed repository using `git clone --no-local`,
checked out the exact commit above in detached-HEAD state, created a new Python
3.12 virtual environment, installed all 46 locked distributions solely from
`requirements-lock.txt`, and installed Juniper Auto editable with `--no-deps`.

- `python scripts/validate_repo.py --all`: all 11 gates passed; 654 tests.
- `python scripts/validate_phase1.py --all`: all 9 gates passed; exact
  parameter counts, forward/backward, checkpoint, hashes, and 654 tests.
- `python scripts/validate_phase2.py --all`: all 13 gates passed; golden-tag
  enforcement, exact counts, reference/optimized equivalence, dropless
  invariants, all-layer trace/gradient telemetry, context probe, reproducibility,
  hashes, registry/time evidence, and 654 tests.
- Five-case `equivalence` and the full `independent-review-demonstration`
  experiment commands both completed and wrote their requested scratch
  results.

CUDA was available and the real GPU-gated tests ran. The sole warning was the
already-accepted PyTorch notice that memory-efficient-attention backward is
nondeterministic; Phase 2 does not claim otherwise. The recovery clone remained
separate from the working repository, so uncommitted state could not influence
the result. Final GitHub workflow identities on the eventual approval commit
are carried by the annotated `phase-2-moe` tag.
