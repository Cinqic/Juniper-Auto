"""ja-tokenizer-v0.1 round-trip correctness -- exact for every valid str."""

from __future__ import annotations

import random

import pytest

from juniper_auto.tokenizer.bytelevel import pretokenize
from tests.tokenizer_fixtures import DOMAIN_SAMPLES, canonical_tokenizer


@pytest.fixture(scope="module")
def tok():
    return canonical_tokenizer()


@pytest.mark.parametrize("domain", sorted(DOMAIN_SAMPLES))
def test_domain_roundtrip_exact(tok, domain):
    for s in DOMAIN_SAMPLES[domain]:
        assert tok.decode(tok.encode(s)) == s


def test_empty_string(tok):
    assert tok.encode("") == []
    assert tok.decode([]) == ""


def test_leading_trailing_and_repeated_whitespace_preserved(tok):
    for s in ["   x", "x   ", "\n\n\nx\n\n\n", "\t\t\tx", "a\t \t b", " " * 17]:
        assert tok.decode(tok.encode(s)) == s


def test_crlf_preserved(tok):
    s = "a\r\nb\r\n\r\nc"
    assert tok.decode(tok.encode(s)) == s


def test_indentation_preserved(tok):
    s = "def f():\n    if x:\n        return 1\n\telif y:\n\t\treturn 2\n"
    assert tok.decode(tok.encode(s)) == s


def test_all_256_single_bytes_roundtrip(tok):
    s = bytes(range(256)).decode("latin-1")
    assert tok.decode(tok.encode(s)) == s


def test_bos_eos_helpers(tok):
    ids = tok.encode("hi", add_bos=True, add_eos=True)
    assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id
    assert tok.decode(ids, skip_special=True) == "hi"


@pytest.mark.parametrize("seed", range(4))
def test_property_random_bmp_roundtrip(tok, seed):
    rng = random.Random(1000 + seed)
    for _ in range(3000):
        n = rng.randint(0, 64)
        s = "".join(chr(rng.randint(0, 0xFFFF)) for _ in range(n))
        try:
            s.encode("utf-8")
        except UnicodeEncodeError:
            continue  # lone surrogate -- not a valid str for utf-8
        assert tok.decode(tok.encode(s)) == s


def test_property_random_full_unicode_roundtrip(tok):
    rng = random.Random(99)
    checked = 0
    for _ in range(8000):
        n = rng.randint(0, 40)
        s = "".join(chr(rng.randint(0, 0x10FFFF)) for _ in range(n))
        try:
            s.encode("utf-8")
        except UnicodeEncodeError:
            continue
        assert tok.decode(tok.encode(s)) == s
        checked += 1
    assert checked > 3000


def test_pretokenizer_is_a_lossless_partition():
    rng = random.Random(3)
    for _ in range(5000):
        s = "".join(chr(rng.randint(0, 0x3FFF)) for _ in range(rng.randint(0, 50)))
        try:
            s.encode("utf-8")
        except UnicodeEncodeError:
            continue
        assert "".join(pretokenize(s)) == s


def test_token_pieces_concatenate_to_utf8_bytes(tok):
    for domain in DOMAIN_SAMPLES.values():
        for s in domain:
            assert b"".join(tok.token_pieces(s)) == s.encode("utf-8")


def test_decode_rejects_unknown_id(tok):
    from juniper_auto.tokenizer.tokenizer import TokenizerArtifactError

    with pytest.raises(TokenizerArtifactError):
        tok.decode([99_999])
