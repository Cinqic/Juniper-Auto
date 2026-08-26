# Phase 1 Recovery Exercise

This documents an actual fresh-clone/fresh-venv recovery exercise
performed for the Phase 1 candidate (not a plausibility claim -- see
[project-governance](../research/project-governance.md) rule 15,
executed-action truthfulness). It extends
[docs/recovery/README.md](README.md) (the general Phase 0 procedure,
unchanged) with the Phase 1-specific validation step.

## Procedure

```bash
# 1. A genuinely fresh directory, not a copy of the working checkout.
mkdir -p /tmp/juniper-auto-phase1-recovery && cd /tmp/juniper-auto-phase1-recovery

# 2. Clone from the actual GitHub remote at the exact candidate commit.
git clone https://github.com/Cinqic/Juniper-Auto.git .
git checkout <candidate-commit>

# 3. A new Python 3.12 virtual environment.
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 4. Install solely from the committed, hash-pinned dependency lock.
pip install -r requirements-lock.txt
pip install -e . --no-deps

# 5. Phase 0 validation.
python scripts/validate_repo.py --all

# 6. Phase 1 validation (imports, official parameter counts, forward/
#    backward smoke, checkpoint round trip, full pytest suite,
#    documentation/hash/experiment/time-accounting checks).
python scripts/validate_phase1.py --all
```

## Result

Sonnet's recovery exercise was run twice, into two separate genuinely
fresh directories, from two different intermediate commits. Its final
metadata candidate was later closed at `6a741013...`; therefore the
passing `22893b7...` exercise proves the substantive Sonnet candidate, not
the later metadata-only identity.

**First run, commit `e4cca8a141ff5205556642b5d28cd5c2b8648d19`: FAILED.**
`scripts/validate_repo.py --all` failed at the "repository integrity"
gate:

```
FAILED at gate 'repository integrity': files reference the current
user's absolute home path outside documented illustrative contexts:
['docs/phases/phase-1-sonnet-self-review.md']
```

Root cause: the self-review report's own Pass I text described its
absolute-path scan using the literal FLOWBOX home-directory path as an
example -- which is itself exactly the kind of host-specific path the
gate exists to catch, and that file is not in the gate's documented
exception list (only `environment-specification.md`/
`phase-0-sonnet-self-review.md` are). Fixed by rewording the sentence to
describe the check without hardcoding the literal path (commit
`22893b7d580504a047254fe261799f1060dc2f88`) -- deliberately not repeated
here either, so this recovery report doesn't reintroduce the same
violation while documenting it. This is recorded rather than silently
re-run past: it is exactly the kind of defect a genuine fresh-clone
exercise is supposed to surface, and it did.

**Second run, commit `22893b7d580504a047254fe261799f1060dc2f88`: PASSED.**
Run into a brand-new directory that had never held any part of this
project, using only `git clone` + the committed lock file + `pip install`
-- no copying, no reused venv, no FLOWBOX-specific setup.

```
$ python scripts/validate_repo.py --all
...
=== [repository integrity] ===
118 tracked files scanned, no prohibited artifacts found
tracked-file size and credential-pattern scans: OK
no unjustified absolute host-specific paths found

=== [artifact hashes] ===
11 artifact hashes verified against manifests/phase-0-artifact-hashes.yaml

=== [deterministic foundation probe] ===
FoundationProbe deterministic under seed=1234 on cpu: checksum=-42.402915954589844

=== [pytest suite] ===
258 passed in 31.13s

All Phase 0 validation gates passed.

$ python scripts/validate_phase1.py --all
...
=== [official model construction and parameter counts] ===
sparse total (method B): 150031360 OK
sparse total (method A): 150031360 OK
sparse active (method A): 79252480 OK
dense total (method B): 79191040 OK
dense total (method A): 79191040 OK
QK-Norm parameter count: 2560 OK
weight tying: OK
bias/dropout audit: OK
layer placement: OK

=== [forward/backward smoke (official models, tiny batch, CPU)] ===
ja150m-v0.1: forward+backward OK, loss=11.2098
ja150m-v0.1-dense: forward+backward OK, loss=10.9320

=== [checkpoint round-trip smoke (tiny config, CPU)] ===
checkpoint save/load round trip: OK

=== [pytest suite (full)] ===
258 passed in <30s

=== [phase 1 documentation] ===
6 required Phase 1 documents present

=== [phase 1 artifact hashes] ===
phase-1-test-manifest.yaml: 20 test file hashes verified
20 Phase 1 artifact hashes verified against manifests/phase-1-artifact-hashes.yaml

=== [phase 1 experiment registry and time accounting] ===
6 phase-1 experiment registry entries found
2 phase-1 time-accounting rows found

All Phase 1 validation gates passed.
```

Both commands exited `0`. Note the forward/backward smoke-test loss
values (11.21 / 10.93) differ from the tiny-overfit experiment's starting
loss (10.60 / 10.61) -- this is expected and not a discrepancy: the smoke
test uses unseeded random `input_ids` on a freshly re-initialized model
in a different process, so it is a different random forward pass, not a
repeat of exp-0004/exp-0005.

## What this exercise proves

- The exact Phase 1 candidate commit is fully reconstructible from GitHub
  alone -- no dependency on local FLOWBOX state, cached wheels, or any
  file outside the committed repository and `requirements-lock.txt`'s
  pinned PyPI packages.
- `scripts/validate_phase1.py --all` genuinely passes from a cold clone,
  not only in the environment that developed it.
- No local, uncommitted, or FLOWBOX-specific state was required at any
  step -- see the absolute-path scan in
  `docs/phases/phase-1-sonnet-self-review.md`, Pass I.

## Known Phase 1-specific recovery notes

- **GPU-dependent experiments are not re-run by this exercise.** CI and
  this recovery procedure are both CPU-only by design (see ADR-0006,
  inherited from Phase 0); `docs/experiments/results/exp-0004` through
  `exp-0008` are the recorded evidence of the GPU-dependent tiny-overfit
  and hardware-profiling experiments, executed once on the actual FLOWBOX
  machine, not re-executed here. `scripts/run_phase1_experiment.py`
  documents the exact commands to reproduce them on hardware with a
  CUDA-capable GPU.
- All other Phase 0 troubleshooting notes in
  [docs/recovery/README.md](README.md) apply unchanged (no standalone
  global `pip3`, large `torch` download, no CUDA/GPU determinism claim).

## GPT-5.6 Sol independent recovery

After repairs and clean canonical reruns were committed and pushed, Sol
performed a new recovery from GitHub at exact substantive reviewed SHA
`9555bbcb43d7b4f63762a5f11c2cea13e11fa7c8` on 2026-08-26:

1. `git clone https://github.com/Cinqic/Juniper-Auto.git` into a new
   temporary directory;
2. detached checkout of exact `9555bbcb...`;
3. new Python 3.12.3 venv;
4. pip upgrade, locked requirements install, editable `--no-deps` install;
5. both canonical validators from inside the clone directory.

Results:

```text
python scripts/validate_repo.py --all
  126 tracked files scanned
  11 Phase 0 hashes verified
  289 passed
  exit 0

python scripts/validate_phase1.py --all
  sparse total 150031360; active 79252480
  dense total 79191040
  official sparse/dense forward+backward passed
  checkpoint round trip passed
  289 passed
  22 protected test files / 33 Phase 1 hashes verified
  12 Phase 1 registry entries found
  exit 0
```

The CUDA-capable recovery emitted PyTorch's documented warning that its
memory-efficient attention backward is nondeterministic; CUDA bitwise
determinism remains an accepted limitation. An initial invocation using
an absolute script path while the shell remained in `/tmp` also passed,
but the standalone foundation log reported `git_commit=unknown` because
that utility discovers Git from the process working directory. Both
validators were rerun exactly as documented from inside the clone and
logged `9555bbcb...`; this methodology warning is preserved here.
