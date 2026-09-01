"""Byte-level plumbing: a reversible byte<->unicode bijection for
human-auditable vocab/merge serialization, and a lossless pre-tokenizer.

The bijection is the well-known GPT-2 ``bytes_to_unicode`` construction: it
maps each of the 256 byte values to a distinct printable, non-whitespace
Unicode code point so that ``vocab.json`` / ``merges.txt`` can be inspected
in a text editor without control characters. It is a pure lookup table with
no information loss -- ``unicode_to_bytes(bytes_to_unicode())`` is the
identity on ``range(256)``.
"""

from __future__ import annotations

import functools
import re

from juniper_auto.tokenizer.constants import PRETOKEN_PATTERN


@functools.lru_cache(maxsize=1)
def byte_to_unicode() -> dict[int, str]:
    """256-entry {byte_value: single_char} bijection (GPT-2 construction)."""
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    mapped = list(printable)
    n = 0
    for b in range(256):
        if b not in printable:
            printable.append(b)
            mapped.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(printable, mapped)}


@functools.lru_cache(maxsize=1)
def unicode_to_byte() -> dict[str, int]:
    return {c: b for b, c in byte_to_unicode().items()}


def bytes_to_token_string(raw: bytes) -> str:
    """Render a byte piece as its auditable unicode-mapped string."""
    table = byte_to_unicode()
    return "".join(table[b] for b in raw)


def token_string_to_bytes(text: str) -> bytes:
    """Inverse of :func:`bytes_to_token_string`."""
    table = unicode_to_byte()
    return bytes(table[c] for c in text)


@functools.lru_cache(maxsize=1)
def _pretoken_re() -> re.Pattern[str]:
    return re.compile(PRETOKEN_PATTERN)


def pretokenize(text: str, *, check: bool = False) -> list[str]:
    """Split ``text`` into pre-token chunks.

    The split is a pure partition: ``"".join(pretokenize(text)) == text`` for
    every ``str``. Pass ``check=True`` to assert that invariant at call time
    (used by tests and the trainer's paranoid path).
    """
    pieces = [m.group(0) for m in _pretoken_re().finditer(text)]
    if check and "".join(pieces) != text:
        raise AssertionError("pretokenize is not a lossless partition of the input")
    return pieces
