# ADR-0012: Tokenizer normalization and pre-tokenization policy for ja-tokenizer-v0.1

Status: accepted
Date: 2026-09-01

## Context

Phase 3 requires that `ja-tokenizer-v0.1` round-trips arbitrary valid UTF-8
losslessly and preserves spaces, tabs, newlines, indentation, repeated
whitespace, code punctuation, and structured-data syntax exactly — "No
lowercasing. No trimming. No whitespace collapsing. No destructive Unicode
normalization." It also needs a pre-tokenization step so that code
indentation, JSON punctuation, repository paths, mathematical syntax, and
tool traces do not fragment absurdly, without that step ever losing
information.

## Decision

### Normalization: identity (none)

No normalizer runs at all. `tokenizer_config.json` records
`"normalization": "none"`. Specifically: no case folding, no
stripping/trimming, no whitespace collapsing, no Unicode NFC/NFD/NFKC, no
compatibility folding (ligatures, full-width forms), no path-separator
rewriting, no BOM insertion or removal, no line-ending normalization.

The identity normalizer *is* the correct solution here (Phase 3 explicitly
allows saying so) because the tokenizer serves code, paths, math, and
structured data, where any of the above transformations changes meaning.

### Pre-tokenization: a lossless full-partition regex

```
'(?:[sdmtSDMT]|ll|LL|ve|VE|re|RE)   # common English contractions
| ?\d+                               # optional leading space + digit run
| ?[^\s\d\W]+                        # optional leading space + letter/underscore run
| ?[^\s\w]+                          # optional leading space + punctuation run
|\s+(?!\S)                           # trailing whitespace before end
|\s+                                # any other whitespace run
```

This pattern is a **guaranteed full partition** of any `str`: every
character is consumed by exactly one alternative (the last two cover all
whitespace; the middle three cover every non-whitespace character at least
as a single char), so `"".join(pretokenize(s)) == s` for every string. That
invariant is property-tested over random Unicode
(`tests/test_tokenizer_roundtrip.py::test_pretokenizer_is_a_lossless_partition`)
and re-checked inside the trainer's paranoid path.

BPE merges are applied *within* each chunk only, so a merge can never span a
newline or fuse indentation with the following token.

## Alternatives considered

- **NFC normalization on input** (very common). Rejected: it silently
  rewrites NFD text (e.g. macOS filenames, decomposed accents) into a
  different byte sequence, so `decode(encode(x)) != x` for a large class of
  real inputs. Losslessness is a hard Phase 3 gate.
- **NFKC** (used by some models). Rejected outright: folds `ﬁ`→`fi`,
  full-width digits→ASCII, `²`→`2`, etc. — actively destructive for code and
  math.
- **Lowercasing / casefold.** Rejected: destroys code identifiers, env-var
  names, acronyms, and case-sensitive paths.
- **GPT-2's exact pre-tokenizer regex.** Not adopted verbatim because it
  needs the `regex` module's `\p{L}` / `\p{N}` classes, which are not a
  project dependency. The pattern above approximates it with `re`'s Unicode
  `\w` / `\d` / `\s` classes. The partition guarantee is what matters and it
  is proven; the exact split boundaries differ slightly from GPT-2's, which
  only affects efficiency, not correctness.
- **Whitespace-splitting only (no punctuation grouping).** Rejected:
  `[^\s\w]+` grouping lets BPE learn `):`, `),`, `];`, `":`, `=>` etc. as
  units, which materially reduces structural fragmentation on code and JSON.
- **No pre-tokenization (BPE over the whole byte stream).** Rejected: merges
  would routinely span spaces and newlines, coupling a token to its
  neighbours and hurting generalization to unseen layouts.

## Consequences

- `decode(encode(x)) == x` for every valid `str` — proven on 15 domains,
  the held-out fixture, the whole training corpus, and 20k+ random
  valid-UTF-8 property cases (`exp-0025`, `tests/test_tokenizer_roundtrip.py`).
- `tests/test_tokenizer_normalization.py` locks in "no lowercasing / strip /
  collapse / NFC-NFD conflation / NFKC folding / path rewriting / BOM
  change" as regression gates.
- Because there is no normalizer, there is no normalization config to drift;
  a future tokenizer that wants normalization is a new version with a
  superseding ADR.
- The pre-tokenizer pattern is recorded in
  `configs/tokenizer/ja-tokenizer-v0.1.yaml`, `tokenizer_config.json`, and
  `constants.PRETOKEN_PATTERN`, and `validate_tokenizer_config` asserts all
  three agree.
