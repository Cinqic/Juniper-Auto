"""ja-tokenizer-v0.1 does no silent normalization."""

from __future__ import annotations

import unicodedata

import pytest

from tests.tokenizer_fixtures import canonical_tokenizer


@pytest.fixture(scope="module")
def tok():
    return canonical_tokenizer()


def test_case_is_preserved(tok):
    for s in ["HELLO", "Hello", "hello", "HeLLo WoRLD", "CamelCase snake_case"]:
        assert tok.decode(tok.encode(s)) == s
    lower_ids = tok.encode("hello")
    upper_ids = tok.encode("HELLO")
    assert lower_ids != upper_ids


def test_whitespace_is_not_stripped_or_collapsed(tok):
    for s in ["  x  ", "a\t\tb", "a\n\n\nb", "   ", "\n", "\t"]:
        assert tok.decode(tok.encode(s)) == s


def test_no_unicode_normalization(tok):
    # NFC vs NFD forms of the same grapheme must not be conflated.
    nfc = unicodedata.normalize("NFC", "é")       # U+00E9
    nfd = unicodedata.normalize("NFD", "é")       # U+0065 U+0301
    assert nfc != nfd
    assert tok.decode(tok.encode(nfc)) == nfc
    assert tok.decode(tok.encode(nfd)) == nfd
    assert tok.encode(nfc) != tok.encode(nfd)


def test_no_nfkc_folding(tok):
    # ﬁ ligature (U+FB01) must not become "fi"; full-width digits stay full-width.
    for s in ["ﬁ", "ﬂ", "１２３", "ＡＢＣ", "①②③", "™ ® ½"]:
        assert tok.decode(tok.encode(s)) == s
        assert tok.decode(tok.encode(s)) != unicodedata.normalize("NFKC", s) or s == unicodedata.normalize("NFKC", s)


def test_path_separators_not_rewritten(tok):
    for s in ["a/b/c", "a\\b\\c", "a//b", "./x", "../y", "C:\\x/y"]:
        assert tok.decode(tok.encode(s)) == s


def test_structured_syntax_not_destructively_normalized(tok):
    for s in ['{ "a" : 1 }', "{\n\t\"a\": 1\n}", "[1,2 ,3]", "<a  b='1'  />"]:
        assert tok.decode(tok.encode(s)) == s


def test_no_bom_insertion_or_stripping(tok):
    s = "\ufeffhello"
    assert tok.decode(tok.encode(s)) == s
    assert tok.decode(tok.encode("hello")) == "hello"


def test_trailing_newline_significant(tok):
    assert tok.encode("x") != tok.encode("x\n")
    assert tok.decode(tok.encode("x\n")) == "x\n"
