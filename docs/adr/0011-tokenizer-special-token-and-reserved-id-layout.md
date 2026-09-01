# ADR-0011: Special-token and reserved-control-id layout for ja-tokenizer-v0.1

Status: accepted
Date: 2026-09-01

## Context

`ja-tokenizer-v0.1` has a fixed 36,864-id budget (it must equal
`ja150m-v0.1.embeddings.vocab_size`). Phase 3 requires: at least 15 named
core control tokens with frozen spelling / id / semantics; a contiguous,
documented reserved range of otherwise-unused ids for future protocol
expansion; a guarantee that ordinary BPE tokenization never emits any of
these ids; and a guarantee that adding future tools does **not** require
retraining the tokenizer. The instructions also warn: "Do not casually
depend on insertion order in a Python dictionary to define an architectural
protocol forever."

## Decision

A fixed **control block** occupying ids `[256, 511]` — 256 ids immediately
above the 256 raw byte tokens and immediately below the learned vocabulary:

```
[0,     255]   256 raw byte tokens          (byte-level base alphabet / fallback)
[256,   270]    15 core control tokens      (fixed spelling + id + semantics)
[271,   511]   241 reserved future-control  (contiguous, unused)
[512, 36863] 36352 learned BPE merge tokens
```

Core control ids are written down explicitly in
`juniper_auto/tokenizer/constants.py` as a literal `(str, int)` list, not
derived from any dict/enum iteration order. Reserved ids have surface
strings `<|reserved_{index}|>` for `index` in `[0, 240]`.

Tokenizer freezes **identity and meaning** of the 15 core tokens (see the
table in `docs/architecture/tokenizer-design.md` §9). The Phase 4 runtime
enforces authority.

## Alternatives considered

- **Special tokens appended above the learned vocabulary (ids
  36864-N .. 36863), byte + learned vocab at the bottom.** This is the
  common HuggingFace pattern. Rejected: it makes the number of learned
  merges depend on how many specials exist, so adding a reserved token
  later would either shrink the learned vocabulary or push the top id past
  36,863 (out of the frozen embedding table) — exactly the failure mode
  Phase 3 warns about ("Do not accidentally create 36,864 ordinary BPE
  tokens and then append control tokens beyond the model's 36,864-entry
  embedding table"). Putting the whole control block at a fixed low offset
  makes the learned-vocab count a simple constant (`36864 - 512`).
- **A much larger reserved range (e.g. 1,024 ids).** Rejected: every
  reserved id is a permanently dead embedding row until activated. 241
  reserved slots is already generous for framing/turn/record-kind/protocol-
  version markers, which is all the reserved range is for; tool identity
  does not live here. 256 total control-block ids (byte-aligned) is a clean
  budget.
- **One reserved token per anticipated future tool.** Explicitly rejected by
  the Phase 3 instructions and by governance rule 1's spirit: a calculator,
  filesystem reader, Git tool, compiler, or API must be addable in Phase 4+
  without retraining the tokenizer. Tool identity belongs in structured
  schemas.
- **Define core ids by `enum` declaration order.** Rejected: an `enum` /
  dict order is an implementation detail; a protocol that other code and
  future checkpoints depend on forever must be an explicit written mapping.
  `ControlToken` (an `enum`) exists for ergonomics, but its `.id` reads from
  the explicit `CORE_CONTROL_STR_TO_ID` table, and a test asserts the two
  agree.

## Consequences

- The learned-vocabulary size is the constant `36864 - 512 = 36352`,
  asserted at import and in `scripts/validate_phase3.py`.
- Future protocol versions activate reserved ids by writing a spec + tests;
  no retrain, no artifact-hash change to `merges.txt` / `vocab.json`.
- `tests/test_tokenizer_vocab.py` and `exp-0028` prove: core ids stable,
  core/reserved disjoint, reserved ids never emitted from text, control-block
  ids never emitted from ordinary `encode()`.
- Changing any core id, the reserved range bounds, or the block position is
  a new tokenizer version with a superseding ADR — never an in-place edit.
- `<|pad|>` is id 258 (inside the control block), so a model's loss mask
  keys on `token_id == 258`, not on a magic index chosen elsewhere.
