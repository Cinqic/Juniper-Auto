# Phase 3 — Claude Sonnet 5 Adversarial Self-Review

Implementer self-review. **Not** the independent review — GPT-5.6 Sol
performs that and holds sole approval authority (governance rule 27).

Method: after the implementation looked complete, stop defending the design
and try to reject it. Passes A–I below follow the Phase 3 instructions'
self-review section. Every issue is either repaired + regression-tested, or
recorded as an explicit candidate limitation.

---

## Pass A — Specification coverage

Built `docs/phases/phase-3-requirements-traceability.md` mapping every Phase
3 master requirement to implementation + test + experiment + doc. Result:
every requirement is mapped. Items initially missing and then added during
this pass:

- **Acceptance criteria not frozen before the eval.** Added an
  `acceptance_criteria` block to `configs/tokenizer/ja-tokenizer-v0.1.yaml`
  (hard gates + domain efficiency gates) with the rationale that hard gates
  are absolute and efficiency gates were set from the byte-floor and GPT-2
  comparison, not tuned post-hoc.
- **`tokens_per_expression` (math) metric** not originally computed — added
  to `evaluation.DomainMetrics`.
- **Cross-process determinism** not originally tested — added
  `test_tokenizer_determinism::test_cross_process_encode_matches`.
- **Reserved-id-from-whole-corpus check** strengthened from a spot check to
  a corpus-wide sweep in `exp-0028`.

## Pass B — Token correctness (try to break it)

Attempted breaks and outcomes (all pass):

| Attack | Result |
|---|---|
| 20k random full-plane Unicode strings | 0 round-trip failures (`exp-0025`, `test_property_random_full_unicode_roundtrip`) |
| all 256 raw byte values as latin-1 text | round-trips (`exp-0025` `all_256_bytes_roundtrip_ok`) |
| leading/trailing/repeated spaces, tabs, CRLF, deep indentation | preserved |
| NFC vs NFD "é", NFKC ligatures, full-width digits, `²`, `½` | preserved and never conflated |
| emoji ZWJ sequences, flags, skin-tone modifiers, combining marks, zero-width joiners, RTL marks | round-trip exactly (`exp-0030`) |
| CJK, mixed scripts | round-trip exactly |
| literal `<\|system\|>`, `<\|<\|system\|>\|>`, all 15 control strings concatenated, `<\|reserved_0\|>` | never emit a control-block id; round-trip exactly |
| `merges.txt` piece beginning with `#` | **was a real bug** (parser treated it as a comment, lost 21 merges on reload). Fixed: exact header sentinel `MERGES_HEADER` + positional parser + `test_missing_merges_header_fails`. |

Found no surviving token-correctness defect.

## Pass C — Artifact reproducibility

- `exp-0024`: two independent full rebuilds from the committed corpus
  produce byte-identical `tokenizer.json` / `vocab.json` / `merges.txt` /
  `special_tokens.json` / `tokenizer_config.json` — every SHA-256 matches
  the frozen artifact.
- `test_full_rebuild_from_committed_corpus_is_hash_identical` runs a full
  rebuild in the pytest suite.
- Checked for hidden state: no `~`/home-directory dependency in the
  tokenizer package (repo-integrity absolute-path gate passes); the frozen
  artifact directory and corpus shards are committed; `load_canonical_tokenizer`
  needs no network (verified via subprocess with no network in
  `test_cross_process_encode_matches`, and the recovery exercise ran the
  validator offline for the tokenizer portion).
- **Known non-idempotency:** `scripts/build_tokenizer_corpus.py` globs live
  repo files + the running interpreter's stdlib, so regenerating the corpus
  against a later repo state or different Python 3.12.x differs. Documented
  as an accepted limitation; the committed shards are the anchor and
  retrain-from-shards is deterministic. This is disclosed, not hidden.
- **Machine-specific paths in the corpus (found during this pass).** The
  first corpus builds swept the building machine's absolute home path
  (`str(Path.home())`) into shards via allowlisted docs
  (`environment-specification.md`, the Phase 0 self-review) and one of my
  own synthetic path snippets, and put the running interpreter's absolute
  stdlib path into `corpus-manifest.json`. **Fixed:** `build_corpus` now
  redacts the home path to `/home/user` in every shard
  (`_HOME_REDACTIONS`, recorded in `corpus-manifest.json` →
  `global_transformations`), the synthetic snippet was changed to
  `/home/user`, and the manifest records `stdlib_relative_root` instead of
  an absolute path. Corpus + tokenizer + all Phase 3 manifests were rebuilt;
  `grep` over the committed shards and manifest confirms no home path
  remains, so the repository's absolute-path hygiene checks (which now cover
  the corpus once it is tracked) pass.

## Pass D — Protocol safety

- Literal control strings in ordinary and hostile text never yield a
  control-block id (`exp-0028` `literal_control_string_leak: false`;
  `test_tokenizer_control_tokens`, `test_tokenizer_fault_injection`).
- The `encode` (untrusted) vs `build_sequence`/`encode_control` (deliberate)
  split is the entire API surface for getting a control id into a sequence;
  `encode` has no parameter that would let control ids through.
- Restated: tokenization is **not** a security boundary — Phase 4 runtime
  enforces authority. This is stated in `docs/architecture/tokenizer-design.md`
  §11 and ADR-0011, not implied away.

## Pass E — Dependency & licensing audit

- Tokenizer implementation: project code, MIT (repo `LICENSE`). No new
  runtime dependency; `pyproject.toml` / `requirements-lock.txt` unchanged;
  `pip check` still clean (Phase 0 baseline gate).
- Comparator: GPT-2 `vocab.json` / `merges.txt` from
  `openai-community/gpt2` (MIT). Fetched into a **gitignored** cache
  (`data/tokenizer/comparators/`), never committed, never imported at
  runtime. Revision + per-file SHA-256 recorded in `exp-0027`.
- Corpus sources: repo files (MIT), CPython stdlib (PSF License 2.0 —
  redistributable, attributed in `corpus-manifest.json` and design §6),
  project-authored synthetic (MIT). No redistribution violation.
- No undeclared runtime downloads: `load_canonical_tokenizer` and the whole
  `juniper_auto.tokenizer` runtime path make zero network calls;
  `comparators.py` is the only module that touches the network and it is
  evaluation-only.

## Pass F — Frozen architecture regression

`scripts/validate_phase3.py::gate_frozen_architecture_unchanged` +
`gate_phase2_baseline` prove Phase 3 did not change:

- sparse total 150,031,360 / sparse standard active 79,252,480 / dense total
  79,191,040 (unchanged, asserted by `juniper_auto/accounting/` and the
  frozen constants);
- layer partition, MoE behaviour, 4,096 context, precision policy — no file
  under `juniper_auto/model/` or `configs/architecture/` was touched
  (git diff is empty for those paths);
- `JuniperAutoModel.forward()` still takes integer token ids only — no
  tokenizer import was added to the model package.

## Pass G — Failure-path review

`test_tokenizer_failure_handling.py` corrupts a copy of the frozen artifact
13 ways (missing file, wrong vocab size, too few merges, missing header,
duplicate-output merge, corrupt vocab JSON, vocab/merge disagreement, id
mismatch, special-map drift, hash mismatch, missing hash manifest,
wrong-tokenizer hash manifest, architecture vocab mismatch). Every one
raises `TokenizerArtifactError` (or `JSONDecodeError` for invalid JSON) with
a useful message; none silently repairs or loads a broken tokenizer.
`test_tokenizer_fault_injection.py` additionally proves 12 specific broken
behaviours are each caught by a specific gate.

## Pass H — Claim audit

Searched `README.md`, `docs/`, and the phase report for unbacked claims.
Findings:

- No claim that a model has been trained, that the tokenizer improved model
  intelligence, that expert specialization exists, that the runtime/tools/
  memory exist, or that the Phase 6 pretraining corpus exists — checked and
  absent. README's Phase 3 section says exactly what exists (a frozen
  tokenizer) and repeats the standing "no model trained" disclaimer.
- Efficiency language is hedged: numbers are "modest", the corpus is "small
  for a 36,864-vocab tokenizer", and the GPT-2 comparison explicitly names
  the domains where Juniper **loses** (general_prose, json).
- Status wording is `CANDIDATE - PENDING INDEPENDENT REVIEW` everywhere;
  independent-review time is `0`/`pending`; `phase-3-tokenizer` tag does not
  exist (`validate_phase3` asserts this).

## Pass I — Repository-integrity audit

- Clean working tree at the candidate commit (checked with
  `git status --porcelain`).
- `.gitignore` updated so `data/tokenizer/comparators/` (the only
  large, non-redistributable-in-bulk cache) is never tracked; verified with
  `git check-ignore`.
- No tracked file exceeds 1 MB (largest new tracked file: `vocab.json`
  ~760 KB; corpus shards ≤ 480 KB) — Phase 0 repo-integrity gate passes.
- No secret files, no `.env`, no keys, no `.pt`/`.safetensors`.
- No absolute FLOWBOX home path in any tracked file (including the 24 corpus
  shards and `corpus-manifest.json` — see Pass C for the redaction that made
  this true) — repo-integrity absolute-path gate and
  `tests/test_absolute_paths.py` pass.
- All required generated artifacts are tracked: the frozen tokenizer
  directory, the 24 corpus shards + manifest, the eval fixture, the phase-3
  manifests.
- All important artifacts are hash-covered:
  `manifests/phase-3-artifact-hashes.yaml` (explicit list) + per-shard
  hashes in `corpus-manifest.json` (re-verified by the validator) + the
  artifact's own `hashes.json`.

---

## Issues found and their disposition

| # | Issue | Disposition |
|---|---|---|
| 1 | `merges.txt` parser treated a `#`-prefixed merge piece as a comment, dropping 21 merges on reload | **Fixed** — exact header sentinel + positional parser; `test_missing_merges_header_fails`, `test_too_few_merges_fails` |
| 2 | The first ~4.6 MB corpus included the held-out eval-fixture builder script (`repo-python` glob) | **Fixed** — `CORPUS_EXCLUDE_GLOBS`; `test_eval_fixture_is_disjoint_from_training_corpus`; corpus rebuilt |
| 3 | The ~4.6 MB corpus needed frequency-1 tail-fill to reach 36,352 merges | **Resolved** — canonical corpus grown to ~8.9 MB; all merges now occur ≥ 2× (`tail_fill_used: false`). Tail-fill path retained + documented as a deterministic guarantee. |
| 4 | Acceptance criteria not frozen before evaluation | **Fixed** — `acceptance_criteria` block added to the config before the canonical `exp-0026` run |
| 5 | No `tokens_per_expression` / cross-process determinism / corpus-wide reserved-id sweep | **Fixed** — added |
| 6 | Building machine's absolute home path + absolute stdlib path leaked into corpus shards / manifest | **Fixed** — home-path redaction in `build_corpus`; manifest records a relative stdlib root; corpus + tokenizer + manifests rebuilt (Pass C) |

## Accepted limitations carried to the candidate

See `docs/phases/phase-3-tokenizer.md` "Accepted limitations": corpus scale,
synthetic domain coverage, GPT-2 comparator pre-tokenizer approximation,
corpus-rebuild non-idempotency, and the `corpus-manifest.json` stdlib-path
field (Pass C). None of these blocks the hard correctness/determinism/
safety gates.

## Areas for Sol to scrutinise especially closely

1. `juniper_auto/tokenizer/bpe.py` incremental pair-count updates — is the
   `pair_where` lazy-membership handling actually correct under all merge
   patterns? (Cross-checked against a naive reference on small inputs in
   `test_small_bpe_train_is_order_deterministic`, but a naive full-corpus
   equivalence check would be stronger.)
2. The pre-tokenizer regex partition guarantee — property-tested, but Sol
   should try pathological inputs (lone combining marks at string start,
   very long digit runs, mixed BiDi).
3. `exp-0024` determinism — Sol should run a third independent rebuild on a
   different machine / Python 3.12 patch level and confirm the artifact
   hashes still match (the corpus shards are committed, so they should).
4. The GPT-2 comparator approximation — is the `re` pre-tokenizer close
   enough that the "Juniper wins on code" conclusion holds under a real
   GPT-2 tokenizer?
5. Whether ~8.9 MB is an acceptable corpus scale for freezing v0.1, or
   whether the candidate should be rejected pending a larger organic corpus.
