"""Frozen identity and vocabulary layout for ``ja-tokenizer-v0.1``.

This module is the single source of truth for the tokenizer's numeric
contract. Nothing here is derived from a Python dict insertion order or a
third-party library default -- every id is written down explicitly so the
control protocol is stable across serialization, reload, and deterministic
rebuild (see docs/research/project-governance.md rule 4 and
docs/adr/0011-tokenizer-special-token-and-reserved-id-layout.md).

Vocabulary layout (total = 36,864 ids, matching ``ja150m-v0.1``'s frozen
``embeddings.vocab_size``):

    [0,     255]   256 raw byte tokens (byte-level base alphabet / fallback)
    [256,   270]    15 core control tokens (fixed spelling + id + semantics)
    [271,   511]   241 reserved future-control ids (contiguous, unused)
    [512, 36863] 36352 learned byte-level BPE merge tokens

The [256, 511] span is the "control block": 256 ids that normal BPE
tokenization can never emit, because their surface strings live only in the
explicit special-token map and are never produced from byte merges.
"""

from __future__ import annotations

TOKENIZER_ID = "ja-tokenizer-v0.1"
ALGORITHM = "utf8-byte-level-bpe"

VOCAB_SIZE = 36_864

# --- byte base alphabet -------------------------------------------------
BYTE_TOKEN_COUNT = 256
BYTE_RANGE = (0, 255)  # inclusive

# --- control block ----------------------------------------------------
CONTROL_BLOCK_START = 256
CONTROL_BLOCK_END = 511  # inclusive
CONTROL_BLOCK_COUNT = CONTROL_BLOCK_END - CONTROL_BLOCK_START + 1  # 256

# Core control tokens: (surface string, id). Order is the frozen protocol
# order; ids are explicit, not positional-by-accident.
CORE_CONTROL_TOKENS: list[tuple[str, int]] = [
    ("<|bos|>", 256),
    ("<|eos|>", 257),
    ("<|pad|>", 258),
    ("<|system|>", 259),
    ("<|user|>", 260),
    ("<|assistant|>", 261),
    ("<|objective|>", 262),
    ("<|state|>", 263),
    ("<|memory|>", 264),
    ("<|tool_call|>", 265),
    ("<|tool_result|>", 266),
    ("<|tool_error|>", 267),
    ("<|observation|>", 268),
    ("<|action|>", 269),
    ("<|final|>", 270),
]
CORE_CONTROL_TOKEN_COUNT = len(CORE_CONTROL_TOKENS)  # 15
CORE_CONTROL_ID_TO_STR: dict[int, str] = {i: s for s, i in CORE_CONTROL_TOKENS}
CORE_CONTROL_STR_TO_ID: dict[str, int] = {s: i for s, i in CORE_CONTROL_TOKENS}

# Human-readable semantic purpose for each core control token. Frozen
# alongside the ids; the runtime (Phase 4) enforces authority, the tokenizer
# only freezes identity and meaning.
CORE_CONTROL_SEMANTICS: dict[str, str] = {
    "<|bos|>": "beginning of a token sequence / document boundary",
    "<|eos|>": "end of a token sequence / document boundary",
    "<|pad|>": "right-padding filler; never carries content; masked in loss",
    "<|system|>": "opens a system-authority turn (runtime-inserted only)",
    "<|user|>": "opens a user turn",
    "<|assistant|>": "opens an assistant turn",
    "<|objective|>": "opens a serialized objective record",
    "<|state|>": "opens a serialized runtime-state record",
    "<|memory|>": "opens a serialized persistent-memory record",
    "<|tool_call|>": "opens a tool invocation emitted by the assistant",
    "<|tool_result|>": "opens a successful tool result supplied to the model",
    "<|tool_error|>": "opens a failed tool result / error supplied to the model",
    "<|observation|>": "opens an environment observation supplied to the model",
    "<|action|>": "opens an action record in an autonomous-control trace",
    "<|final|>": "marks the assistant's final answer segment",
}

# --- reserved future-control range ----------------------------------
# Contiguous, documented, otherwise-unused. Future protocol versions may
# activate/version these ids WITHOUT retraining the tokenizer (the byte
# merges and learned vocabulary are unaffected). Per-tool tokens are
# explicitly NOT allocated here -- tool identity belongs in Phase 4+
# structured schemas.
RESERVED_CONTROL_START = 271
RESERVED_CONTROL_END = 511  # inclusive
RESERVED_CONTROL_COUNT = RESERVED_CONTROL_END - RESERVED_CONTROL_START + 1  # 241


def reserved_control_token_str(index: int) -> str:
    """Surface string for reserved-control slot ``index`` (0-based)."""
    if not 0 <= index < RESERVED_CONTROL_COUNT:
        raise ValueError(
            f"reserved-control index {index} out of range "
            f"[0, {RESERVED_CONTROL_COUNT - 1}]"
        )
    return f"<|reserved_{index}|>"


RESERVED_CONTROL_TOKENS: list[tuple[str, int]] = [
    (reserved_control_token_str(k), RESERVED_CONTROL_START + k)
    for k in range(RESERVED_CONTROL_COUNT)
]
RESERVED_CONTROL_ID_TO_STR: dict[int, str] = {i: s for s, i in RESERVED_CONTROL_TOKENS}
RESERVED_CONTROL_STR_TO_ID: dict[str, int] = {s: i for s, i in RESERVED_CONTROL_TOKENS}

# --- learned vocabulary ------------------------------------------------
LEARNED_VOCAB_START = 512
NUM_MERGES = VOCAB_SIZE - LEARNED_VOCAB_START  # 36,352
LEARNED_VOCAB_END = VOCAB_SIZE - 1  # 36,863

# Full special-token map (core + reserved). Every id here is unique and
# lives inside the control block; none is producible by byte-level merges.
ALL_SPECIAL_TOKENS: list[tuple[str, int]] = CORE_CONTROL_TOKENS + RESERVED_CONTROL_TOKENS
SPECIAL_STR_TO_ID: dict[str, int] = {s: i for s, i in ALL_SPECIAL_TOKENS}
SPECIAL_ID_TO_STR: dict[int, str] = {i: s for s, i in ALL_SPECIAL_TOKENS}

# --- pre-tokenization -------------------------------------------------
# A guaranteed FULL PARTITION of any str: every character is consumed by
# exactly one alternative and matched pieces concatenate back to the input.
# Proven by tests/test_tokenizer_normalization.py (property test) and
# juniper_auto.tokenizer.bytelevel.pretokenize's own assertion in --check
# mode. Order matters: contractions, then digits, then letters/underscore,
# then punctuation, then trailing whitespace, then any whitespace run.
PRETOKEN_PATTERN = (
    r"'(?:[sdmtSDMT]|ll|LL|ve|VE|re|RE)"
    r"| ?\d+"
    r"| ?[^\s\d\W]+"
    r"| ?[^\s\w]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


def _self_check() -> None:
    ids = [i for _, i in ALL_SPECIAL_TOKENS]
    assert len(ids) == len(set(ids)), "duplicate special-token id"
    assert CORE_CONTROL_TOKEN_COUNT == 15
    assert RESERVED_CONTROL_COUNT == 241
    assert BYTE_TOKEN_COUNT + CONTROL_BLOCK_COUNT + NUM_MERGES == VOCAB_SIZE
    assert CONTROL_BLOCK_COUNT == CORE_CONTROL_TOKEN_COUNT + RESERVED_CONTROL_COUNT
    assert min(ids) == CONTROL_BLOCK_START
    assert max(ids) == CONTROL_BLOCK_END
    assert set(CORE_CONTROL_ID_TO_STR) & set(RESERVED_CONTROL_ID_TO_STR) == set()
    assert LEARNED_VOCAB_START == CONTROL_BLOCK_END + 1


_self_check()
