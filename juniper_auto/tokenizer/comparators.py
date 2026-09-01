"""Baseline tokenizer comparators for the Phase 3 evaluation.

Two comparators:

* ``utf8-bytes`` -- the lossless byte-level floor (bytes/token == 1.0), a
  reasonable general-purpose reference requiring no external artifact.
* ``gpt2`` -- the tokenizer of a small (124M) model. Its ``vocab.json`` and
  ``merges.txt`` are fetched once from a pinned Hugging Face revision,
  verified by SHA-256, and cached under ``data/tokenizer/comparators/``
  (gitignored). This is a COMPARATOR-ONLY artifact: it is never a runtime
  dependency of ``ja-tokenizer-v0.1`` and the canonical tokenizer loads
  with no network access. If the cache is absent and the network is
  unavailable, GPT-2 comparison is skipped with a clear message and the
  numbers recorded in the committed experiment JSON stand.

The GPT-2 pre-tokenizer here is a ``re``-based approximation of GPT-2's
original ``regex``-module pattern (the ``regex`` module and its ``\\p{L}``
classes are not a project dependency); token counts are therefore
indicative, within a few percent, not bit-exact to a reference GPT-2
implementation. This is disclosed in the Phase 3 report.
"""

from __future__ import annotations

import functools
import hashlib
import json
import re
import urllib.request
from pathlib import Path

from juniper_auto.tokenizer.bpe import apply_merges
from juniper_auto.tokenizer.bytelevel import byte_to_unicode

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPARATOR_DIR = REPO_ROOT / "data" / "tokenizer" / "comparators" / "gpt2"

GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
GPT2_BASE_URL = f"https://huggingface.co/openai-community/gpt2/resolve/{GPT2_REVISION}"
GPT2_FILES = {
    "vocab.json": f"{GPT2_BASE_URL}/vocab.json",
    "merges.txt": f"{GPT2_BASE_URL}/merges.txt",
}

_GPT2_PRETOKEN_RE = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+"
)


class ComparatorUnavailable(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_gpt2_artifacts(*, allow_download: bool = True) -> dict[str, str]:
    """Return {filename: sha256}, downloading into the gitignored cache if
    needed and permitted. Raises ComparatorUnavailable if the artifacts are
    absent and cannot be fetched."""
    COMPARATOR_DIR.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for name, url in GPT2_FILES.items():
        path = COMPARATOR_DIR / name
        if not path.is_file():
            if not allow_download:
                raise ComparatorUnavailable(f"gpt2 comparator artifact missing: {name}")
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310
                    path.write_bytes(resp.read())
            except Exception as exc:  # network failure, offline CI, etc.
                raise ComparatorUnavailable(f"could not fetch {url}: {exc}") from exc
        hashes[name] = _sha256(path)
    return hashes


@functools.lru_cache(maxsize=1)
def _gpt2_tables() -> tuple[dict[bytes, int], dict[tuple[bytes, bytes], int]]:
    u2b = {c: b for b, c in byte_to_unicode().items()}
    raw_vocab = json.loads((COMPARATOR_DIR / "vocab.json").read_text(encoding="utf-8"))
    vocab: dict[bytes, int] = {}
    for tok_str, idx in raw_vocab.items():
        vocab[bytes(u2b[ch] for ch in tok_str)] = idx
    merges: dict[tuple[bytes, bytes], int] = {}
    lines = (COMPARATOR_DIR / "merges.txt").read_text(encoding="utf-8").splitlines()
    rank = 0
    for line in lines:
        if line.startswith("#") or not line.strip():
            continue
        a, b = line.split(" ")
        merges[(bytes(u2b[c] for c in a), bytes(u2b[c] for c in b))] = rank
        rank += 1
    return vocab, merges


def gpt2_encode(text: str) -> list[int]:
    vocab, ranks = _gpt2_tables()
    out: list[int] = []
    for chunk in _GPT2_PRETOKEN_RE.findall(text):
        symbols = tuple(bytes([x]) for x in chunk.encode("utf-8"))
        for piece in apply_merges(symbols, ranks):
            out.append(vocab.get(piece, -1))
    return out


def gpt2_available() -> bool:
    return all((COMPARATOR_DIR / n).is_file() for n in GPT2_FILES)
