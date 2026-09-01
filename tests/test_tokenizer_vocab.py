"""ja-tokenizer-v0.1 vocabulary invariants."""

from __future__ import annotations

import pytest

from juniper_auto.tokenizer import constants as C
from juniper_auto.tokenizer.tokenizer import ControlToken
from tests.tokenizer_fixtures import canonical_tokenizer


@pytest.fixture(scope="module")
def tok():
    return canonical_tokenizer()


def test_exact_vocab_size(tok):
    assert tok.vocab_size == 36_864
    assert C.VOCAB_SIZE == 36_864


def test_id_space_is_contiguous_and_hole_free(tok):
    ids = set(tok._id_to_bytes) | set(tok._special_id_to_str)
    assert ids == set(range(36_864))
    assert len(tok._id_to_bytes) + len(tok._special_id_to_str) == 36_864


def test_byte_and_special_ids_disjoint(tok):
    assert set(tok._id_to_bytes).isdisjoint(tok._special_id_to_str)


def test_all_ids_in_valid_range(tok):
    for i in list(tok._id_to_bytes) + list(tok._special_id_to_str):
        assert 0 <= i < 36_864


def test_256_byte_base_tokens(tok):
    for b in range(256):
        assert tok._id_to_bytes[b] == bytes([b])
        assert tok.id_to_token_bytes(b) == bytes([b])


def test_every_core_control_token_exists_with_expected_id(tok):
    expected = {
        "<|bos|>": 256, "<|eos|>": 257, "<|pad|>": 258, "<|system|>": 259,
        "<|user|>": 260, "<|assistant|>": 261, "<|objective|>": 262,
        "<|state|>": 263, "<|memory|>": 264, "<|tool_call|>": 265,
        "<|tool_result|>": 266, "<|tool_error|>": 267, "<|observation|>": 268,
        "<|action|>": 269, "<|final|>": 270,
    }
    assert dict(C.CORE_CONTROL_TOKENS) == expected
    for s, i in expected.items():
        assert tok._special_id_to_str[i] == s
    assert [t.id for t in ControlToken] == list(range(256, 271))


def test_core_ids_are_unique(tok):
    ids = [i for _, i in C.CORE_CONTROL_TOKENS]
    assert len(ids) == len(set(ids)) == 15


def test_reserved_range_is_exact_and_contiguous(tok):
    assert (C.RESERVED_CONTROL_START, C.RESERVED_CONTROL_END, C.RESERVED_CONTROL_COUNT) == (271, 511, 241)
    reserved_ids = [i for _, i in C.RESERVED_CONTROL_TOKENS]
    assert reserved_ids == list(range(271, 512))
    for s, i in C.RESERVED_CONTROL_TOKENS:
        assert tok._special_id_to_str[i] == s


def test_core_and_reserved_do_not_overlap(tok):
    core = {i for _, i in C.CORE_CONTROL_TOKENS}
    reserved = {i for _, i in C.RESERVED_CONTROL_TOKENS}
    assert core.isdisjoint(reserved)
    assert core | reserved == set(range(256, 512))


def test_control_block_is_256_ids(tok):
    assert C.CONTROL_BLOCK_COUNT == 256
    assert C.BYTE_TOKEN_COUNT + C.CONTROL_BLOCK_COUNT + C.NUM_MERGES == 36_864


def test_learned_vocab_starts_at_512_and_has_exact_count(tok):
    assert C.LEARNED_VOCAB_START == 512
    assert C.NUM_MERGES == 36_352
    assert len(tok.merges) == 36_352
    learned_ids = [512 + r for r in range(len(tok.merges))]
    assert learned_ids[0] == 512
    assert learned_ids[-1] == 36_863


def test_learned_vocab_is_collision_free(tok):
    pieces = [a + b for a, b in tok.merges]
    assert len(pieces) == len(set(pieces))


def test_reserved_ids_never_produced_by_normal_tokenization(tok):
    import random

    rng = random.Random(7)
    for _ in range(4000):
        s = "".join(chr(rng.randint(0, 0x2FFF)) for _ in range(rng.randint(0, 60)))
        try:
            s.encode("utf-8")
        except UnicodeEncodeError:
            continue
        for t in tok.encode(s):
            assert not (256 <= t <= 511), f"control-block id {t} emitted for {s!r}"


def test_highest_id_within_frozen_model_vocab(tok):
    assert max(tok._id_to_bytes) == 36_863 < 36_864
