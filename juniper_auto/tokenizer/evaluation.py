"""Tokenizer engineering evaluation: per-domain efficiency, structural
fragmentation, byte-fallback behaviour, and baseline comparisons.

Definitions (so another engineer can reproduce the numbers):

* ``chars_per_token``  = ``len(text)`` (Unicode scalar values) / ``n_tokens``
* ``bytes_per_token``  = ``len(text.encode("utf-8"))`` / ``n_tokens``
* ``tokens_per_line``  = ``n_tokens`` / ``max(1, text.count("\\n") + 1)``
  (reported for code domains)
* ``tokens_per_expression`` = ``n_tokens`` / (number of non-empty lines)
  (reported for the math domain, where each fixture line is one expression)
* ``structural_fragmentation`` = (number of emitted tokens whose decoded
  bytes are exactly one character from the fixed STRUCTURAL_CHARS set) /
  (number of STRUCTURAL_CHARS characters present in the text). 1.0 means
  every bracket/brace/quote/colon/comma is its own token; lower means the
  tokenizer absorbed structural punctuation into multi-character tokens.
* ``byte_fallback_rate`` = (number of emitted tokens with id < 256) /
  ``n_tokens``. Non-zero is fine and always lossless; it just means the
  learned vocabulary did not cover that text well.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from juniper_auto.tokenizer.tokenizer import JuniperTokenizer

STRUCTURAL_CHARS = set("{}[]()<>:;,=\"'`|/\\\t")

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_FIXTURE_PATH = REPO_ROOT / "data" / "tokenizer" / "eval" / "held-out-eval-fixture.json"


@dataclass
class DomainMetrics:
    domain: str
    n_samples: int
    n_chars: int
    n_bytes: int
    n_tokens: int
    chars_per_token: float
    bytes_per_token: float
    tokens_per_line: float
    tokens_per_expression: float
    structural_fragmentation: float
    byte_fallback_rate: float
    roundtrip_failures: int


def _structural_and_fallback(tokenizer: JuniperTokenizer, ids: list[int], text: str) -> tuple[int, int, int]:
    structural_present = sum(1 for ch in text if ch in STRUCTURAL_CHARS)
    single_structural_tokens = 0
    fallback = 0
    for tid in ids:
        if tid < 256:
            fallback += 1
        b = tokenizer.id_to_token_bytes(tid)
        if b is not None and len(b) == 1:
            try:
                ch = b.decode("utf-8")
            except UnicodeDecodeError:
                continue
            if ch in STRUCTURAL_CHARS:
                single_structural_tokens += 1
    return structural_present, single_structural_tokens, fallback


def evaluate_domain(
    tokenizer: JuniperTokenizer, domain: str, samples: list[str]
) -> DomainMetrics:
    n_chars = n_bytes = n_tokens = 0
    n_lines = n_exprs = 0
    structural_present = single_structural = fallback = 0
    roundtrip_failures = 0
    for s in samples:
        ids = tokenizer.encode(s)
        if tokenizer.decode(ids) != s:
            roundtrip_failures += 1
        n_chars += len(s)
        n_bytes += len(s.encode("utf-8"))
        n_tokens += len(ids)
        n_lines += s.count("\n") + 1
        n_exprs += sum(1 for line in s.splitlines() if line.strip())
        sp, sst, fb = _structural_and_fallback(tokenizer, ids, s)
        structural_present += sp
        single_structural += sst
        fallback += fb
    nt = max(1, n_tokens)
    return DomainMetrics(
        domain=domain,
        n_samples=len(samples),
        n_chars=n_chars,
        n_bytes=n_bytes,
        n_tokens=n_tokens,
        chars_per_token=round(n_chars / nt, 4),
        bytes_per_token=round(n_bytes / nt, 4),
        tokens_per_line=round(n_tokens / max(1, n_lines), 4),
        tokens_per_expression=round(n_tokens / max(1, n_exprs), 4),
        structural_fragmentation=round(single_structural / max(1, structural_present), 4),
        byte_fallback_rate=round(fallback / nt, 6),
        roundtrip_failures=roundtrip_failures,
    )


def load_eval_fixture(path: Path = EVAL_FIXTURE_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def evaluate_all_domains(tokenizer: JuniperTokenizer, fixture: dict | None = None) -> dict:
    fixture = fixture or load_eval_fixture()
    results = {}
    for domain, samples in fixture["domains"].items():
        results[domain] = asdict(evaluate_domain(tokenizer, domain, samples))
    total_tokens = sum(r["n_tokens"] for r in results.values())
    total_chars = sum(r["n_chars"] for r in results.values())
    total_bytes = sum(r["n_bytes"] for r in results.values())
    total_fail = sum(r["roundtrip_failures"] for r in results.values())
    return {
        "tokenizer_id": tokenizer.tokenizer_id,
        "fixture_id": fixture["fixture_id"],
        "fixture_version": fixture["version"],
        "per_domain": results,
        "overall": {
            "n_tokens": total_tokens,
            "chars_per_token": round(total_chars / max(1, total_tokens), 4),
            "bytes_per_token": round(total_bytes / max(1, total_tokens), 4),
            "roundtrip_failures": total_fail,
        },
    }


# ------------------------------------------------------------------
# Baseline comparators
# ------------------------------------------------------------------
def byte_baseline_metrics(fixture: dict | None = None) -> dict:
    """UTF-8 byte tokenizer: bytes_per_token == 1.0 by construction; the
    lossless floor for a byte-level scheme."""
    fixture = fixture or load_eval_fixture()
    out = {}
    for domain, samples in fixture["domains"].items():
        nb = sum(len(s.encode("utf-8")) for s in samples)
        nc = sum(len(s) for s in samples)
        out[domain] = {
            "n_tokens": nb,
            "bytes_per_token": 1.0,
            "chars_per_token": round(nc / max(1, nb), 4),
        }
    return {"comparator": "utf8-bytes", "per_domain": out}


def compare_tokens_per_domain(
    tokenizer: JuniperTokenizer,
    other_encode,
    fixture: dict | None = None,
) -> dict:
    """Compare token counts per domain against any ``other_encode(str)->list``.
    Returns Juniper tokens, comparator tokens, and the ratio (>1 => Juniper
    is more compact)."""
    fixture = fixture or load_eval_fixture()
    out = {}
    for domain, samples in fixture["domains"].items():
        ja = sum(len(tokenizer.encode(s)) for s in samples)
        ot = sum(len(other_encode(s)) for s in samples)
        out[domain] = {
            "juniper_tokens": ja,
            "comparator_tokens": ot,
            "compression_ratio_vs_comparator": round(ot / max(1, ja), 4),
        }
    return out
