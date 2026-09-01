# ja-tokenizer-v0.1 — Design & Engineering Report

Status: Phase 3 candidate. Related: [[project-charter]], [[project-governance]],
[ADR-0010](../adr/0010-tokenizer-implementation-choice.md),
[ADR-0011](../adr/0011-tokenizer-special-token-and-reserved-id-layout.md),
[ADR-0012](../adr/0012-tokenizer-normalization-and-pretokenization-policy.md).

This document is the implementation-level companion to the Phase 3 report
([docs/phases/phase-3-tokenizer.md](../phases/phase-3-tokenizer.md)). It
records what `ja-tokenizer-v0.1` is, how it was built, how it was
evaluated, and what it does and does not guarantee.

## 1. Objective

Produce **one** unified tokenizer for the single Juniper Auto cognitive
model, covering general language, technical writing, documentation,
software and code, comments, mathematics, numerical expressions, structured
data (JSON/YAML/XML/HTML), shell and terminal content, logs, paths, URLs,
Git output, tool schemas/calls/results/errors, and objective/state/memory/
observation/action records. There is no separate language, code, math, or
tool tokenizer, and none will be added (governance rule 1).

Phase 3 is tokenizer engineering only. It builds no Phase 4 runtime, no
Phase 5 evaluations, no Phase 6 pretraining corpus, and trains no model.

## 2. Algorithm

UTF-8 **byte-level BPE**. Every input is first UTF-8 encoded; merges operate
on byte sequences. The 256 raw byte values are permanent base tokens, so any
valid UTF-8 text is representable without an `<unk>` token.

* Merge selection during training: at each step pick the byte pair with the
  highest corpus count; ties broken by the **lexicographically smallest**
  byte pair. Never by hash order or `dict` order.
* Merge application during encoding: greedy, **lowest-rank-first**.
* A merge whose resulting bytes already exist as a token is applied to the
  corpus (so training keeps making progress) but not recorded as a new
  vocabulary entry, so the learned vocabulary is collision-free (no shadowed
  / dead ids). In the canonical run this never triggered (`collisions_skipped
  = 0`).

Implementation: `juniper_auto/tokenizer/bpe.py`, pure Python, no third-party
tokenizer dependency (ADR-0010).

## 3. Implementation / library decision

See [ADR-0010](../adr/0010-tokenizer-implementation-choice.md). Summary: the
byte-level BPE machinery is implemented in-project (~200 lines) rather than
adopting `tokenizers` / `sentencepiece` / `tiktoken`. Reasons: full
determinism control, transparent freezable artifacts, exact special-token
control, zero new runtime dependency, no runtime network access, and
governance rule 31 (modifiability — no opaque third-party black box). GPT-2's
tokenizer is used as a **comparator only** (`juniper_auto/tokenizer/
comparators.py`); it is never a runtime dependency and the canonical
tokenizer loads with no network access.

## 4. Normalization policy

**Identity — no normalization at all.** No lowercasing, no stripping, no
whitespace collapsing, no Unicode NFC/NFD/NFKC, no path-separator rewriting,
no BOM insertion/removal. Rationale ([ADR-0012](../adr/0012-tokenizer-normalization-and-pretokenization-policy.md)):
code, paths, mathematical notation, and structured data must round-trip
byte-for-byte; any destructive normalization corrupts them. `tokenizer_config.json`
records `"normalization": "none"`, and `tests/test_tokenizer_normalization.py`
proves case, whitespace, NFC-vs-NFD, NFKC ligatures/full-width forms, path
separators, and structured syntax are all preserved and never conflated.

## 5. Byte-level design

* `juniper_auto/tokenizer/bytelevel.py` provides the GPT-2
  `bytes_to_unicode` bijection: each byte value maps to one printable,
  non-whitespace code point, purely so `vocab.json` / `merges.txt` are
  human-auditable in a text editor. It is a lossless lookup table
  (`token_string_to_bytes(bytes_to_token_string(b)) == b` for all 256
  values).
* Pre-tokenization splits text into chunks with a regex that is a
  **guaranteed full partition**: `"".join(pretokenize(s)) == s` for every
  `str` (property-tested over random Unicode). The pattern
  (`configs/tokenizer/ja-tokenizer-v0.1.yaml` → `pre_tokenization.pattern`)
  keeps whitespace runs, leading spaces, digit runs, letter runs, and
  punctuation runs as separate chunks so BPE never merges across a newline
  or collapses indentation.
* Encoding: pre-tokenize → UTF-8 encode each chunk → apply merges → map each
  resulting byte piece to its id. Decoding: id → bytes (byte or merge) or
  special-token surface string → concatenate → `bytes.decode("utf-8")`.

## 6. Corpus composition & provenance

The tokenizer-training corpus (`data/tokenizer/corpus/`, 24 shards, ~8.91 MB,
every shard < 480 KB) is **committed** — it is the canonical training input.
`corpus-manifest.json` records, per shard: category, source, license,
redistribution status, transformation, byte count, SHA-256, and
source-detail (file list + Python version for stdlib, generator+seed for
synthetic). `scripts/build_tokenizer_corpus.py` documents the derivation and
can regenerate it, but the committed shards are canonical and the training
path (`load_corpus_shards`) verifies every shard hash before use.

| Category | Bytes | Source | License |
|---|---|---|---|
| `repo-python` | ~0.60 MB | this repo's `juniper_auto/`, `scripts/`, `tests/` (held-out fixture files excluded) | MIT |
| `repo-docs` | ~0.28 MB | this repo's `docs/`, `README.md` | MIT |
| `repo-config` | ~0.18 MB | this repo's configs/manifests/experiments/`pyproject.toml` | MIT |
| `stdlib-python` | ~5.16 MB | CPython 3.12 stdlib modules + packages (argparse, asyncio, email, http, json, logging, re, unittest, urllib, xml, …) | PSF License 2.0 |
| `synthetic-javascript` | ~0.33 MB | project-authored JS templates | MIT |
| `synthetic-c` / `synthetic-cpp` | ~0.21 MB each | project-authored C / C++ templates | MIT |
| `synthetic-math` | ~0.30 MB | project-authored mathematical / numeric notation | MIT |
| `synthetic-shell-logs` | ~0.39 MB | project-authored shell sessions, terminal output, logs, git output | MIT |
| `synthetic-paths-urls` | ~0.23 MB | project-authored Unix/Windows paths, URLs | MIT |
| `synthetic-structured` | ~0.33 MB | project-authored JSON/YAML/XML/HTML | MIT |
| `synthetic-tool-traces` | ~0.33 MB | project-authored tool-call/result/error + objective/state/memory/observation/action records | MIT |
| `synthetic-prose` | ~0.36 MB | project-authored general/technical/educational prose | MIT |

Generic prose is deliberately capped (~0.36 MB of ~8.9 MB) so code,
structured data, and tool traces are not drowned out. All synthetic content
is labelled synthetic (governance rule 36). No private files, secrets, or
credentials are included; the corpus is scanned for this in the repository
integrity checks.

Held-out evaluation material is explicitly excluded from the corpus
(`CORPUS_EXCLUDE_GLOBS`), and `tests/test_tokenizer_evaluation.py::
test_eval_fixture_is_disjoint_from_training_corpus` proves no held-out
sample appears verbatim in any shard.

### Reproducibility note

`scripts/build_tokenizer_corpus.py` globs live repository files and the
running interpreter's stdlib, so re-running it against a *later* repository
state or a different Python 3.12 patch level will produce different shards.
That is why the **committed shards are canonical**: the guarantee that
matters — retrain from the committed shards + config + seed and get every
artifact hash back — is what `tests/test_tokenizer_determinism.py::
test_full_rebuild_from_committed_corpus_is_hash_identical` and
`exp-0024` verify.

## 7. BPE training method

`juniper_auto/tokenizer/train.py`:

1. Load + hash-verify the 24 committed shards in manifest order.
2. Pre-tokenize every shard; build a `{pre-token: frequency}` table
   (~48,500 distinct pre-tokens, ~2.1M occurrences).
3. `train_bpe(..., num_merges=36352, min_pair_frequency=2, tail_fill=True)`.
   Incremental heap-backed pair counting; ~9 s wall on FLOWBOX.
4. In the canonical run, all 36,352 merges were learned with every merged
   pair occurring **at least twice** (`merges_at_primary_threshold = 36352`,
   `tail_fill_used = false`). The tail-fill path (drop to frequency 1 to
   fill the exact budget) exists as a deterministic guarantee but was not
   needed.
5. Serialize to `data/tokenizer/ja-tokenizer-v0.1/` and write `hashes.json`.

Seeds: `corpus_seed = training_seed = 20260901`. The training seed only
affects synthetic-corpus generation ordering during a corpus rebuild;
`train_bpe` itself is seed-free and fully deterministic given the word table.

## 8. Vocabulary accounting

| Range | Count | Contents |
|---|---|---|
| `[0, 255]` | 256 | raw byte tokens (base alphabet / fallback) |
| `[256, 270]` | 15 | core control tokens (fixed spelling + id + semantics) |
| `[271, 511]` | 241 | reserved future-control ids (contiguous, unused) |
| `[512, 36863]` | 36,352 | learned byte-level BPE merge tokens |
| **total** | **36,864** | == `ja150m-v0.1.embeddings.vocab_size` |

Highest valid id is 36,863 < 36,864, so it fits the frozen model embedding
table. `juniper_auto/tokenizer/constants.py` is the single source of truth
and self-checks these sums at import.

## 9. Core control token ids (frozen)

| id | token | purpose |
|---|---|---|
| 256 | `<\|bos\|>` | beginning of sequence / document boundary |
| 257 | `<\|eos\|>` | end of sequence / document boundary |
| 258 | `<\|pad\|>` | right-padding filler; masked in loss |
| 259 | `<\|system\|>` | system-authority turn (runtime-inserted only) |
| 260 | `<\|user\|>` | user turn |
| 261 | `<\|assistant\|>` | assistant turn |
| 262 | `<\|objective\|>` | serialized objective record |
| 263 | `<\|state\|>` | serialized runtime-state record |
| 264 | `<\|memory\|>` | serialized persistent-memory record |
| 265 | `<\|tool_call\|>` | tool invocation emitted by the assistant |
| 266 | `<\|tool_result\|>` | successful tool result supplied to the model |
| 267 | `<\|tool_error\|>` | failed tool result / error supplied to the model |
| 268 | `<\|observation\|>` | environment observation supplied to the model |
| 269 | `<\|action\|>` | action record in an autonomous-control trace |
| 270 | `<\|final\|>` | assistant's final answer segment |

These ids, spellings, and semantics are frozen and asserted by
`tests/test_tokenizer_vocab.py` and `scripts/validate_phase3.py`. `add_bos` /
`add_eos` default to **false**; nothing is inserted implicitly.

## 10. Reserved future-control range (frozen)

`[271, 511]`, 241 contiguous ids, surface pattern `<|reserved_{index}|>`.
Rationale ([ADR-0011](../adr/0011-tokenizer-special-token-and-reserved-id-layout.md)):
future protocol versions can add framing/control tokens (new turn kinds,
new record kinds, versioned protocol markers) without retraining the
tokenizer — the byte merges and learned vocabulary are untouched. Reserved
ids:

* never overlap core control ids (proven);
* are never produced by normal BPE tokenization (proven over the whole
  corpus and 4k+ random cases — `exp-0028`);
* have stable numeric identities across serialization/reload/rebuild;
* count toward the 36,864 total.

**Per-tool tokens are explicitly not allocated here.** Adding a calculator,
filesystem reader, Git tool, compiler, or API in Phase 4+ must not require
retraining the tokenizer; tool identity lives in structured schemas.

## 11. Control-token safety semantics

Two distinct API paths:

* **Ordinary / untrusted text** — `tokenizer.encode(text)` (alias
  `encode_ordinary`). Every byte of `text` is encoded as bytes. If `text`
  contains the literal string `<|system|>`, it is tokenized as the ordinary
  bytes `<`, `|`, `system`, `|`, `>` and **never** yields id 259. Proven by
  `tests/test_tokenizer_control_tokens.py` and `exp-0028` over hostile
  inputs.
* **Deliberate protocol insertion** — `tokenizer.build_sequence([...])` /
  `encode_control(...)`. The only way a control-block id (256–511) enters a
  sequence is by naming a `ControlToken` / `ReservedControl`, or passing a
  raw `int` id (range-checked).

The tokenizer is **not** a security boundary — the Phase 4 runtime enforces
authority. But the ordinary-text path cannot accidentally manufacture a
control token, which removes an avoidable footgun. This contract is
documented here for Phase 4 and tested now.

## 12. Serialization format & artifact layout

`data/tokenizer/ja-tokenizer-v0.1/`:

| file | contents |
|---|---|
| `tokenizer.json` | id, algorithm, vocab size, layout ranges, special-token map, `merges_count`, training metadata |
| `vocab.json` | `{byte-mapped-string: id}` for all 256 byte + 36,352 merge tokens (human-auditable) |
| `merges.txt` | one `A B` merge per line under an exact header sentinel; rank == 0-based line index |
| `special_tokens.json` | core control map + per-token semantics + reserved-range spec |
| `tokenizer_config.json` | normalization/pre-tokenizer/bos-eos policy |
| `hashes.json` | SHA-256 of the five files above |

`JuniperTokenizer.load()` reconstructs the vocabulary by replaying
`merges.txt`, then **cross-checks** it against `vocab.json` (mismatch → hard
error). Loading verifies `hashes.json` by default and needs no network.
Every failure path (missing file, wrong vocab size, too few merges, missing
header, duplicate merge, corrupt vocab, id/map drift, hash mismatch, arch
mismatch) raises `TokenizerArtifactError` and is covered by
`tests/test_tokenizer_failure_handling.py`.

## 13. Evaluation methodology

Held-out fixture: `data/tokenizer/eval/held-out-eval-fixture.json`
(v1.0.0, 15 domains, 39 difficult samples, disjoint from training). Metrics
(`juniper_auto/tokenizer/evaluation.py`, definitions in that module's
docstring): `chars_per_token`, `bytes_per_token`, `tokens_per_line` (code),
`tokens_per_expression` (math), `structural_fragmentation`, `byte_fallback_rate`,
`roundtrip_failures`.

`structural_fragmentation` = (emitted tokens whose decoded bytes are exactly
one char from `{}[]()<>:;,="'`|/\` + tab) / (count of those chars in the
text). 1.0 = every such char is its own token; lower = punctuation absorbed
into multi-char tokens.

### Canonical results (exp-0026), chars/token by domain

| domain | ja | domain | ja | domain | ja |
|---|---|---|---|---|---|
| general_prose | 3.95 | shell | 2.61 | urls | 2.71 |
| technical_prose | 4.74 | json | 2.36 | file_paths | 2.50 |
| python | 3.31 | yaml | 3.02 | tool_traces | 3.33 |
| javascript | 2.47 | xml_html | 2.30 | state_records | 3.28 |
| c_cpp | 2.66 | math | 2.25 | memory_records | 2.88 |

Overall 2.89 chars/token; **0 round-trip failures on every domain**;
`byte_fallback_rate` 0.11–0.53 (always lossless). Full per-domain metrics
including `structural_fragmentation` and `tokens_per_line` are in
`exp-0026`.

## 14. Comparison methodology

Comparators (`exp-0027`):

* **`utf8-bytes`** — the lossless byte-level floor (`bytes_per_token = 1.0`);
  no external artifact.
* **`gpt2`** — `openai-community/gpt2` (124M) `vocab.json` + `merges.txt`,
  pinned revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`, SHA-256
  recorded in `exp-0027`, fetched once into a gitignored cache. The GPT-2
  pre-tokenizer here is a `re`-based approximation of GPT-2's original
  `regex`-module pattern (the `regex` module is not a project dependency),
  so GPT-2 counts are indicative within a few percent, not bit-exact.
  Special tokens are excluded on both sides (raw text only).

### vs `utf8-bytes`

`ja-tokenizer-v0.1` is 2.3×–4.7× more compact than raw UTF-8 bytes on every
domain (weakest: math 2.25×; strongest: technical_prose 4.74×).

### vs GPT-2 (ratio = gpt2_tokens / juniper_tokens; > 1 ⇒ Juniper more compact)

| Juniper more compact | ≈ equal | GPT-2 more compact |
|---|---|---|
| python 1.61, yaml 1.35, shell 1.28, tool_traces 1.22, file_paths 1.10, state_records 1.11, memory_records 1.10, c_cpp 1.06, javascript 1.05, math 1.03 | technical_prose 1.00, xml_html 0.99, urls 0.96 | general_prose 0.77, json 0.88 |

Juniper trades English-prose efficiency (GPT-2's much larger English
training corpus wins there) for better code, structured-data, and
agent-trace efficiency — which matches the project's autonomous-agent
target. This is **not** a claim of universal tokenizer superiority.

## 15. Performance (FLOWBOX-class, `exp-0029`)

| metric | value |
|---|---|
| tokenizer training wall time | ~9 s (standalone); ~44.5 s under `tracemalloc` in `exp-0029` |
| BPE merge-learning time | ~33 s (tracemalloc-instrumented) / ~7 s standalone |
| peak Python heap during training | ~124 MB |
| final artifact size | ~1.17 MB |
| encode throughput | ~4.3 M chars/s (~4.3 MB/s) |
| decode throughput | ~10.7 M tokens/s |
| GPU-hours | 0 (BPE is CPU-only; no GPU work occurred) |

## 16. Known limitations

* **Corpus scale.** ~8.9 MB is small for a 36,864-vocab tokenizer. English
  prose efficiency and `byte_fallback_rate` on adversarial inputs would both
  improve with a much larger organic corpus. A production retrain
  (`ja-tokenizer-v0.2`) should use one; the pipeline already supports
  swapping in organic sources with provenance.
* **Synthetic domain coverage.** JS/C/C++, shell, math, tool-trace, and
  structured-data coverage is project-authored synthetic, not organic. It is
  representative of the token shapes those domains produce, but a real
  corpus would be broader.
* **GPT-2 comparator pre-tokenizer** is a `re` approximation (see §14).
* **Corpus rebuild is not idempotent** against a later repo state (§6); the
  committed shards are the reproducibility anchor.
* No claim is made that the tokenizer improves model intelligence, that any
  model has been trained, or that expert specialization exists.

## 17. Future protocol compatibility

* New control/framing tokens: activate reserved ids `[271, 511]` in a future
  protocol spec — no tokenizer retrain.
* New tools: structured schemas in Phase 4+, not new tokens.
* A vocabulary or algorithm change is a new tokenizer version
  (`ja-tokenizer-v0.2`) with a superseding ADR — never an in-place edit
  (governance rule 4).
