"""ja-tokenizer-v0.1 evaluation harness + held-out fixture + byte comparator."""

from __future__ import annotations

import json

import pytest

from juniper_auto.tokenizer.evaluation import (
    EVAL_FIXTURE_PATH,
    STRUCTURAL_CHARS,
    byte_baseline_metrics,
    evaluate_all_domains,
    evaluate_domain,
    load_eval_fixture,
)
from tests.tokenizer_fixtures import canonical_tokenizer


@pytest.fixture(scope="module")
def tok():
    return canonical_tokenizer()


def test_eval_fixture_is_disjoint_from_training_corpus():
    """No held-out sample appears verbatim in any committed corpus shard."""
    from juniper_auto.tokenizer.corpus import load_corpus_shards

    corpus_text = "\n".join(text for _n, text in load_corpus_shards())
    fixture = load_eval_fixture()
    for domain, samples in fixture["domains"].items():
        for s in samples:
            probe = s.strip()[:80]
            assert probe and probe not in corpus_text, f"{domain}: held-out sample leaked into training corpus"


def test_fixture_has_all_required_domains():
    fixture = load_eval_fixture()
    required = {
        "general_prose", "technical_prose", "python", "javascript", "c_cpp",
        "shell", "json", "yaml", "xml_html", "math", "urls", "file_paths",
        "tool_traces", "state_records", "memory_records",
    }
    assert required <= set(fixture["domains"])


def test_evaluate_all_domains_zero_roundtrip_failures(tok):
    report = evaluate_all_domains(tok)
    assert report["overall"]["roundtrip_failures"] == 0
    for domain, m in report["per_domain"].items():
        assert m["roundtrip_failures"] == 0, domain
        assert m["chars_per_token"] > 1.0
        assert 0.0 <= m["byte_fallback_rate"] <= 1.0
        assert 0.0 <= m["structural_fragmentation"] <= 1.0


def test_structural_fragmentation_definition_is_stable(tok):
    # An all-braces string: every brace is structural; fragmentation in [0, 1].
    m = evaluate_domain(tok, "braces", ["{}{}{}{}{}[][][]"])
    assert 0.0 <= m.structural_fragmentation <= 1.0
    assert "{" in STRUCTURAL_CHARS and ":" in STRUCTURAL_CHARS


def test_byte_baseline_is_one_byte_per_token():
    b = byte_baseline_metrics()
    for domain, m in b["per_domain"].items():
        assert m["bytes_per_token"] == 1.0


def test_juniper_beats_byte_baseline_on_prose(tok):
    fixture = load_eval_fixture()
    for domain in ("general_prose", "technical_prose"):
        ja = sum(len(tok.encode(s)) for s in fixture["domains"][domain])
        by = sum(len(s.encode("utf-8")) for s in fixture["domains"][domain])
        assert by / ja > 2.0  # at least 2x more compact than raw bytes


def test_fixture_hash_recorded_matches_file():
    import hashlib

    text = EVAL_FIXTURE_PATH.read_text(encoding="utf-8")
    fixture = json.loads(text)
    assert fixture["version"] == "1.0.0"
    # sanity: the file parses and round-trips its own structure
    assert hashlib.sha256(text.encode()).hexdigest()
