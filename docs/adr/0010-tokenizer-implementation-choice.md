# ADR-0010: Tokenizer implementation — in-project byte-level BPE, no third-party tokenizer dependency

Status: accepted
Date: 2026-09-01

## Context

Phase 3 must produce `ja-tokenizer-v0.1`: one unified UTF-8 byte-level BPE
tokenizer, exactly 36,864 ids, byte fallback (no `<unk>`), identity
normalization, an explicit frozen special-token map, a reserved
future-control range, and full deterministic reproducibility. The repository
had no tokenizer library dependency before Phase 3.

The choice is between (a) adopting an open tokenizer library
(`tokenizers` / `sentencepiece` / `tiktoken`) and (b) implementing the
byte-level BPE machinery in-project.

Phase 3's instructions explicitly warn against choosing "based purely on
which requires fewer lines of code" and require evaluating: deterministic
training, byte-fallback semantics, artifact transparency, ability to freeze
vocab/merges, exact special-token control, portability, license, long-term
maintainability, runtime dependency size, CPU performance, and modifiability
by Juniper Auto users.

## Decision

**Implement byte-level BPE in-project** (`juniper_auto/tokenizer/`, ~600
lines total across `bpe.py`, `bytelevel.py`, `tokenizer.py`, `train.py`).
Add **no** new runtime dependency. Use HuggingFace GPT-2's `vocab.json` /
`merges.txt` only as a **comparator** in the Phase 3 evaluation
(`juniper_auto/tokenizer/comparators.py`), fetched into a gitignored cache,
never imported at runtime, never required to load `ja-tokenizer-v0.1`.

## Alternatives considered

- **`huggingface/tokenizers` (Rust-backed).** Rejected. Its BPE trainer's
  parallelism and internal ordering make bit-for-bit deterministic retrain
  hard to guarantee and hard to audit; the serialized `tokenizer.json` is a
  single opaque blob; freezing / diffing merges and special ids is awkward;
  it is a large compiled runtime dependency; and it would be an opaque
  third-party component that the project owner cannot fully see or modify
  (governance rule 31). It also does not obviously satisfy "runtime
  tokenizer loading does not depend on downloading files from the internet"
  without care.
- **`sentencepiece`.** Rejected. Unigram is the idiomatic sentencepiece
  model and is not what Phase 3 specifies; the byte-level BPE path is less
  first-class; the trained model is an opaque protobuf; special-token and
  byte-fallback handling are configurable but indirect; and it is a
  compiled dependency.
- **`tiktoken`.** Rejected. Encode-only (no trainer), and it fetches vocab
  files from a remote blob store at first use — a runtime network
  dependency that directly violates the offline-load requirement.
- **In-project, but reuse a vendored minimal BPE (`minbpe`-style).**
  Partially adopted — the algorithm is textbook — but the naive whole-corpus
  implementation is too slow for 36,352 merges over ~9 MB, so `bpe.py` uses
  an incremental heap-backed pair-count updater (word-frequency form).

## Consequences

- Full control over determinism: merge selection is `argmax` by
  `(count, lexicographically smallest byte pair)`; merge application is
  greedy lowest-rank-first; no hash/`dict` order anywhere. Deterministic
  retrain is *proven* (`exp-0024`, `tests/test_tokenizer_determinism.py`).
- Artifacts are human-auditable: `vocab.json` and `merges.txt` are text, and
  `load()` cross-checks them against each other.
- Zero new runtime dependency; `pyproject.toml` / `requirements-lock.txt`
  are unchanged. The tokenizer is torch-free and loads with no network.
- The project owner can read and modify every line of the tokenizer
  (governance rule 31).
- Maintenance cost: the project now owns ~600 lines of tokenizer code and
  its tests. This is accepted; the code is small, well-tested (10 test
  files, 100+ cases, deliberate fault injection), and the algorithm is
  stable.
- Performance is adequate on FLOWBOX (train ~9 s, encode ~3.7 MB/s,
  decode ~10 M tokens/s — `exp-0029`). If encode throughput ever becomes a
  bottleneck for training data loading, an optional compiled fast-path can
  be added behind the same API without changing the artifact format — that
  would be its own ADR.
- Because `tokenizers`/`tiktoken` are not present, the Phase 3 comparison
  uses GPT-2's raw vocab/merges with a `re`-based pre-tokenizer
  approximation; comparator token counts are indicative within a few
  percent, disclosed in `docs/architecture/tokenizer-design.md` §14.
