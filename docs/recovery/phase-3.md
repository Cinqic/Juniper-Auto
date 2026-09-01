# Phase 3 Recovery Exercise — ja-tokenizer-v0.1

Extends [docs/recovery/README.md](README.md) and
[docs/recovery/phase-2.md](phase-2.md) with the Phase 3 tokenizer artifacts.
A genuine fresh-clone / fresh-venv exercise from the canonical pushed
candidate, in a location with no prior project state.

## What Phase 3 adds to recovery

- `data/tokenizer/ja-tokenizer-v0.1/` — the frozen tokenizer artifact
  (committed; loads with **no network access**).
- `data/tokenizer/corpus/` — 24 committed training-corpus shards +
  `corpus-manifest.json` (per-shard SHA-256; the building machine's absolute
  home path is redacted).
- `data/tokenizer/eval/held-out-eval-fixture.json` — held-out evaluation
  fixture v1.0.0.
- `manifests/phase-3-artifact-hashes.yaml`, `manifests/phase-3-test-manifest.yaml`.
- `scripts/validate_phase3.py` — canonical Phase 3 validation entrypoint.

Nothing Phase 3 adds requires a GPU, a network connection (the optional
GPT-2 comparator in `exp-0027` is not part of loading or validating the
tokenizer), or any artifact hosted outside this git repository. There is no
`~/.cache`, home-directory, or local-corpus dependency for ordinary
tokenizer loading — proven below by loading and rebuilding entirely from the
fresh clone.

## Procedure (exactly as executed)

```bash
git clone https://github.com/Cinqic/Juniper-Auto.git
cd Juniper-Auto
git fetch --tags
python3 -m venv .venv && source .venv/bin/activate
python -m ensurepip --upgrade
python -m pip install --upgrade pip
pip install -r requirements-lock.txt
pip install -e . --no-deps
```

## Result — fresh clone at `5e4dfd5d6a32a7ee95d21171c83aac9feb9e6879`

Executed 2026-09-01 on a Linux Mint 22.3 machine (Python 3.12.3) in a
directory that had never held the project, with its own new virtualenv.

| Step | Result |
|---|---|
| `git clone` + `git fetch --tags` | OK — tags `phase-0-foundation`, `phase-1-architecture`, `phase-2-moe` present; HEAD `5e4dfd5` |
| `pip install -r requirements-lock.txt` | OK (hash-pinned; rc=0) |
| `pip install -e . --no-deps` | OK (rc=0) |
| Load `ja-tokenizer-v0.1` (`load_canonical_tokenizer()`), **no network** | OK — `vocab_size == 36864`; hash manifest verified |
| Round-trip check on 6 mixed-domain samples (prose, code, JSON, CJK+emoji, Windows path, math with a real minus sign) | 0 failures |
| Control-token safety (`encode("<\|system\|>")` has no id 259; `build_sequence([SYSTEM])[0] == 259`) | OK |
| Deterministic rebuild (`train_tokenizer` into a temp dir) | 36,352 merges; **every artifact SHA-256 identical** to the committed frozen artifact |
| `grep -r /home/<user>` over the working tree (excluding `.git`, `.venv`) | no matches — no machine-specific path leaked into any committed file |
| `python scripts/validate_phase3.py --all` (includes the Phase 0/1/2 baselines and the full pytest suite) | **PASS** — see below |

```
Juniper Auto Phase 3 repository validation
Gates: 17
=== [phase 2 baseline (includes phases 0 and 1)] === PASS
...
=== [phase 3 documentation] === 8 required Phase 3 docs present; status is candidate-pending; approval tag absent
=== [phase 3 experiment registry and time accounting] === 7 phase-3 experiments, 2 phase-3 time rows OK
=== [phase 3 artifact + test-manifest hashes] === 47 Phase 3 artifact hashes + 10 test-file hashes verified OK
=== [earlier-phase manifests still current] === phase-1 and phase-2 artifact manifests are current OK
=== [pytest suite (full)] === 766 passed, 1 warning (the pre-existing CUDA memory-efficient-attention non-determinism warning, unchanged from Phase 2)

All Phase 3 validation gates passed.
(exit code 0)
```

## First-attempt note

The first attempt succeeded. No first-attempt failure occurred for the
fresh-clone recovery of the Phase 3 candidate. (Two engineering-time issues —
a `merges.txt` `#`-prefix parsing bug and an eval-fixture / home-path
contamination of the training corpus — were found and fixed *before* the
candidate was committed; see
[docs/phases/phase-3-sonnet-self-review.md](../phases/phase-3-sonnet-self-review.md).
Those are not recovery failures.)
