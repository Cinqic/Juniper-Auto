# Phase 3 Report — Unified Tokenizer and Control Vocabulary

## Phase

Phase 3 — Unified Tokenizer (`ja-tokenizer-v0.1`).

## Objective

Create and freeze one unified UTF-8 byte-level BPE tokenizer for the single
Juniper Auto cognitive model — exactly 36,864 ids, lossless round-trip for
all valid UTF-8, byte fallback with no `<unk>`, identity normalization, a
frozen 15-token core control vocabulary, and a contiguous reserved
future-control range — with a provenanced training corpus, a held-out
evaluation, baseline comparisons, deterministic-rebuild proof, an adversarial
test suite, a Phase 3 validator, CI, and full documentation, suitable for
independent review by GPT-5.6 Sol.

## Starting commit

`05fc185a573504fea4901845bc114d3fb79d8567` — `main` HEAD at Phase 3 start,
which is exactly the commit the approved `phase-2-moe` annotated tag points
to (the tag *object* hash is `bf08fd40…`; it dereferences to this commit).
Working tree clean; Phase 0/1/2 validators green at start (655 pytest
passed, 1 pre-existing CUDA memory-efficient-attention non-determinism
warning).

## Final commit

`CANDIDATE - PENDING INDEPENDENT REVIEW`.

- Sonnet substantive candidate (implementation + tests + docs, experiments
  executed against it): `6a6ef177328d187670f2fe3e4f2ccb11ccb3660d`
- Sonnet closure candidate (experiments, validator, CI, manifests,
  frozen-artifact registry): `5e4dfd5d6a32a7ee95d21171c83aac9feb9e6879` —
  all four CI workflows green (see CI section).
- Sonnet metadata-closure HEAD (this section, CI/recovery evidence): a later
  commit, necessarily.

The final approved commit and its immutable CI identity are resolved by the
annotated `phase-3-tokenizer` approval tag **after** GPT-5.6 Sol's
independent review. `phase-3-tokenizer` has **not** been created.

## Implementation summary

- `juniper_auto/tokenizer/constants.py` — the single source of truth for the
  36,864-id layout: 256 byte tokens `[0,255]`, 15 core control tokens
  `[256,270]`, 241 reserved control ids `[271,511]`, 36,352 learned merges
  `[512,36863]`. Self-checks the sums at import.
- `juniper_auto/tokenizer/bytelevel.py` — GPT-2 byte↔unicode bijection (for
  auditable serialization) and the lossless full-partition pre-tokenizer.
- `juniper_auto/tokenizer/bpe.py` — deterministic incremental byte-level BPE
  trainer (heap-backed pair counts; `argmax` by `(count, smallest pair)`;
  collision-free learned vocab; optional frequency-1 tail-fill) and the
  greedy lowest-rank-first merge-application encoder.
- `juniper_auto/tokenizer/tokenizer.py` — `JuniperTokenizer`: `encode`
  (ordinary/untrusted text — never emits control ids), `build_sequence` /
  `encode_control` (deliberate protocol insertion), `decode`,
  `token_pieces`, `assert_model_compatible`, `save` / `load` with a
  merge↔vocab cross-check and hard invariant validation. `ControlToken` enum
  and `ReservedControl` handle.
- `juniper_auto/tokenizer/artifacts.py` — frozen artifact paths, SHA-256
  manifest (`hashes.json`), fail-loud verification, offline `load_canonical_tokenizer`.
- `juniper_auto/tokenizer/corpus.py` — corpus assembly with per-shard
  provenance; committed shards are canonical; `load_corpus_shards` verifies
  every shard hash.
- `juniper_auto/tokenizer/train.py` — end-to-end training into the frozen
  artifact directory; deterministic given committed shards + config + seed.
- `juniper_auto/tokenizer/evaluation.py` — per-domain metrics
  (chars/token, bytes/token, tokens/line, tokens/expression, structural
  fragmentation, byte-fallback rate, round-trip failures) + byte baseline.
- `juniper_auto/tokenizer/comparators.py` — GPT-2 comparator (fetch pinned
  revision, verify SHA-256, gitignored cache, never a runtime dependency).
- `juniper_auto/tokenizer/config.py` — schema + cross-checks for
  `configs/tokenizer/ja-tokenizer-v0.1.yaml` against the code and artifact.
- Scripts: `scripts/build_tokenizer_corpus.py`,
  `scripts/build_tokenizer_eval_fixture.py`, `scripts/train_tokenizer.py`,
  `scripts/run_phase3_experiment.py`, `scripts/validate_phase3.py`,
  `scripts/generate_phase3_test_manifest.py`; `scripts/hash_manifest.py`
  extended with `--phase 3`.
- Frozen artifact `data/tokenizer/ja-tokenizer-v0.1/` (`tokenizer.json`,
  `vocab.json`, `merges.txt`, `special_tokens.json`, `tokenizer_config.json`,
  `hashes.json`).
- Corpus `data/tokenizer/corpus/` (24 shards + `corpus-manifest.json`).
- Held-out fixture `data/tokenizer/eval/held-out-eval-fixture.json` (v1.0.0).
- Tests: `tests/tokenizer_fixtures.py` + 9 `tests/test_tokenizer_*.py`
  (vocab invariants, round-trip + property, normalization, determinism,
  control tokens, failure handling, fault injection, config/model, evaluation).
- Docs: this report, `docs/phases/phase-3-requirements-traceability.md`,
  `docs/phases/phase-3-sonnet-self-review.md`, `docs/recovery/phase-3.md`,
  `docs/architecture/tokenizer-design.md`, ADR-0010/0011/0012.
- CI: `.github/workflows/phase-3-validation.yml` (full history + tags,
  clean `ubuntu-latest`, Python 3.12, locked install, `validate_phase3.py --all`).

The frozen `ja150m-v0.1` / `ja150m-v0.1-dense` architectures are unchanged;
`JuniperAutoModel.forward()` still consumes integer token ids only.

## Architecture / configuration IDs

Applies to `ja150m-v0.1` and `ja150m-v0.1-dense` (both freeze
`embeddings.vocab_size: 36864`). New artifact id: `ja-tokenizer-v0.1`. New
config: `configs/tokenizer/ja-tokenizer-v0.1.yaml`. No architecture config
changed.

## Environment

[docs/architecture/environment-specification.md](../architecture/environment-specification.md).
FLOWBOX (Ryzen 7 5700G, RTX 2060 6 GB, 16 GB RAM), Python 3.12.3,
torch 2.13.0+cu130. Tokenizer training/encoding/decoding is pure Python and
**torch-free**; no GPU is used or required. CI runs CPU-only on
`ubuntu-latest`.

## Artifacts

Per `manifests/frozen-artifacts.yaml`:

| Artifact | Status |
|---|---|
| `tokenizer` — `ja-tokenizer-v0.1` (`data/tokenizer/ja-tokenizer-v0.1/`) | `frozen` |
| `special_token_map` — `ja-tokenizer-v0.1-special-tokens` (`data/tokenizer/ja-tokenizer-v0.1/special_tokens.json`) | `frozen` |
| tokenizer training corpus (`data/tokenizer/corpus/`) | committed, hash-manifested |
| held-out eval fixture v1.0.0 | committed, hash-manifested |
| `ja150m-v0.1` / `ja150m-v0.1-dense` architecture | `frozen` (unchanged) |
| `base_checkpoint` / `instruction_checkpoint` / `pretraining_dataset` / runtime / evals | `not-yet-created` (unchanged) |

## Hashes

`manifests/phase-3-artifact-hashes.yaml` (generated by
`scripts/hash_manifest.py --phase 3`) and
`manifests/phase-3-test-manifest.yaml` (generated by
`scripts/generate_phase3_test_manifest.py`). The frozen tokenizer artifact
additionally carries its own `data/tokenizer/ja-tokenizer-v0.1/hashes.json`.
`manifests/phase-1-artifact-hashes.yaml` and
`manifests/phase-2-artifact-hashes.yaml` were regenerated for the three
globally-evolving files they share with Phase 3
(`docs/time/phase-hours.csv`, `experiments/registry.yaml`, `README.md`); the
approved Phase 0/1/2 states remain pinned by their immutable git tags.

Canonical artifact SHA-256 (recorded here for the reviewer):

```
tokenizer.json        26b49727e15ebd1754ad4544cb4e66a89c32bb0da46a3593a532fa51d919835a
vocab.json            d84243f070bb808d99f4e9bcb32b03d5e0714b8f7492a40a2889ad6bfb862285
merges.txt            64b34f50e33b5dae1e3a54e24e0b1ff867fe460c223cf7ca8fcb9fcf30ed95d9
special_tokens.json   7a870b42edc2843694dc83e64b42ba2169f2fadfa22038cfbc845fb4d36644c6
tokenizer_config.json 0098eda74cdcd43eee6a7dfb5f1d1176094c41c562cec275f4e6838d06629c60
corpus-manifest.json  2fcc2b029aede670a3ef2b599047cae80c032a0419963c6850b3ce75c852066d
eval fixture v1.0.0   e9a134d51941a7af8183ca4f5bf6eaa56adf44a94c864ed6cab60e3a26a55051
```

(The training corpus was rebuilt once late in engineering to redact the
building machine's absolute home path from every shard — see the
self-review; these are the post-redaction canonical hashes.)

## Tests

`tests/test_tokenizer_*.py` — 110+ cases across vocabulary invariants,
exact round-trip on 15+ domains, randomized valid-UTF-8 property round-trip
(BMP + full plane), lossless-partition pre-tokenizer property, normalization
non-interference, save/reload/cross-process/full-rebuild determinism,
control-token safety, malformed-artifact fail-closed behaviour, and
deliberate fault injection proving each gate is load-bearing. Full suite
(`python -m pytest tests/ -q`): **766 passed, 1 warning** locally, in CI on
`5e4dfd5` (all four workflows), and in the fresh-clone recovery run. The one
warning (CUDA memory-efficient-attention non-determinism in
`tests/test_training_checkpoint.py`) is unchanged from the Phase 2 baseline
and unrelated to Phase 3.

## Evaluations

`exp-0026` (efficiency, held-out fixture v1.0.0) — chars/token by domain:

```
general_prose 3.95  technical_prose 4.74  python 3.31  javascript 2.47
c_cpp 2.66  shell 2.61  json 2.36  yaml 3.02  xml_html 2.30  math 2.25
urls 2.71  file_paths 2.50  tool_traces 3.33  state_records 3.28  memory_records 2.88
```

Overall 2.89 chars/token; **0 round-trip failures on every domain**;
byte-fallback rate 0.11–0.53 (always lossless). `exp-0027` (baseline
comparison): 2.3×–4.7× more compact than raw UTF-8 bytes on every domain;
vs GPT-2, Juniper is more compact on python (1.61×), yaml (1.35×), shell
(1.28×), tool_traces (1.22×), file_paths (1.10×), state/memory records,
c_cpp, javascript, math; roughly equal on technical_prose / xml_html /
urls; and less compact on general_prose (1.30× GPT-2's favour) and json
(1.14× GPT-2's favour). Acceptance criteria
(frozen in `configs/tokenizer/ja-tokenizer-v0.1.yaml` before the canonical
evaluation) are all met — see requirements traceability.

## Ablations (where relevant)

`not-applicable` in the model sense (no model trained). The corpus-scale
sensitivity was observed during engineering (a ~4.6 MB corpus needed
frequency-1 tail-fill; the ~8.9 MB canonical corpus reaches all 36,352
merges at frequency ≥ 2) and is recorded in the self-review, not as a
registry ablation.

## CI workflow / run

All four workflows ran on the Sonnet substantive+closure candidate
`5e4dfd5d6a32a7ee95d21171c83aac9feb9e6879` and concluded **success**:

| Workflow | Run ID | Conclusion | Commit tested |
|---|---|---|---|
| Phase 0 Validation | 33481751022 | success | `5e4dfd5` |
| Phase 1 Validation | 33481751004 | success | `5e4dfd5` |
| Phase 2 Validation | 33481750894 | success | `5e4dfd5` |
| Phase 3 Validation | 33481750910 | success | `5e4dfd5` |

`.github/workflows/phase-3-validation.yml` (name: **Phase 3 Validation**)
uses `fetch-depth: 0` + `fetch-tags: true`. The metadata-closure commit adds
its own CI evidence; the immutable resolution of the final approved commit
and CI identity is the `phase-3-tokenizer` tag after GPT-5.6 Sol's
independent review.

## Recovery status

Performed and **passed on the first attempt**. A fresh `git clone` of
`5e4dfd5` on a machine with no prior project state, its own new virtualenv,
locked-dependency install: the tokenizer loads offline, round-trips, its
control-token safety holds, a full retrain reproduces every artifact
SHA-256, no machine-specific path is present in any committed file, and
`python scripts/validate_phase3.py --all` passes (766 pytest passed). Exact
commands and output: [docs/recovery/phase-3.md](../recovery/phase-3.md).

## Engineering hours

`docs/time/phase-hours.csv` `phase-3` rows: ~3.9 h AI-assisted engineering +
~0.8 h self-review/closure. `active_human_hours` is `PENDING` (not
observable by the implementer — the human reviewer fills it in).

## Independent review hours

`0` / `pending` — GPT-5.6 Sol has not reviewed Phase 3.

## GPU hours

`0`. Byte-level BPE tokenizer engineering is CPU-only; no GPU work occurred
in any Phase 3 activity, and `exp-0029` records `gpu_hours: 0.0`.

## CPU / data-processing hours

`docs/time/phase-hours.csv` `phase-3` rows: ~0.9 h total — corpus assembly,
repeated full tokenizer training/rebuild, the cross-domain evaluation +
GPT-2 comparator fetch, four-deep nested validator/pytest invocations, and
the fresh-clone recovery install + validation.

## Project elapsed days

Initial commit `a991db7` 2026-08-25; Phase 3 closure candidate `5e4dfd5`
2026-09-01 → **7 calendar days** of project elapsed time (Phases 0–3).

## Known failures

- None outstanding. During engineering: (1) `merges.txt` parsing initially
  treated a learned merge piece beginning with `#` as a comment, losing 21
  merges on reload — fixed with an exact header sentinel and a positional
  parser, regression-tested. (2) The first, smaller corpus put the held-out
  eval-fixture builder script into the training corpus (`repo-python`
  glob) — fixed with `CORPUS_EXCLUDE_GLOBS` and a disjointness test, corpus
  rebuilt. Both are recorded in the self-review and were resolved before the
  candidate.

## Negative results

No `experiments/registry.yaml` entry for Phase 3 has `status: failed` or
`status: negative-result`. The corpus-scale observation above (frequency-1
tail-fill needed at ~4.6 MB) is a documented engineering finding, not a
failed experiment; the canonical run does not use tail-fill.

## Accepted limitations

- **Corpus scale (~8.9 MB)** is small for a 36,864-vocab tokenizer. English
  prose efficiency and adversarial-input byte-fallback rate would both
  improve with a much larger organic corpus. A production retrain
  (`ja-tokenizer-v0.2`) should use one.
- **Synthetic domain coverage** — JS/C/C++, shell, math, structured-data,
  and tool-trace corpus content is project-authored synthetic (labelled as
  such), representative of token shapes but not organic breadth.
- **GPT-2 comparator pre-tokenizer** is a `re` approximation of GPT-2's
  `regex`-module pattern; comparator counts are indicative within a few
  percent.
- **Corpus rebuild is not idempotent** against a later repository state or a
  different Python 3.12 patch level (it globs live files); the committed
  shards are the reproducibility anchor and deterministic retrain-from-shards
  is proven.
- No claim that the tokenizer improves model capability, that any model has
  been trained, or that specialization exists.

## Reproducibility procedure

From a clean clone (see [docs/recovery/phase-3.md](../recovery/phase-3.md)):

```bash
git clone https://github.com/Cinqic/Juniper-Auto.git && cd Juniper-Auto
git fetch --tags
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-lock.txt
pip install -e . --no-deps
python scripts/validate_phase3.py --all
# optional: prove deterministic rebuild
python -c "from juniper_auto.tokenizer.train import train_tokenizer; import tempfile,pathlib; \
  t,r=train_tokenizer(out_dir=pathlib.Path(tempfile.mkdtemp())/'x'); print(r.bpe_steps_completed)"
```

## Reviewer identity

Primary implementer / self-review: **Claude Sonnet 5**
(`docs/phases/phase-3-sonnet-self-review.md`).

Independent reviewer / approval authority: **GPT-5.6 Sol — pending**.

## Approval status

`CANDIDATE - PENDING INDEPENDENT REVIEW`

Reviewer: GPT-5.6 Sol (pending). The approved commit, exact CI identities,
and independent approval will be resolved by the annotated `phase-3-tokenizer`
tag once GPT-5.6 Sol completes an independent repository-based review. This
report must not be read as evidence of approval.
