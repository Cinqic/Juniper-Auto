"""ja-tokenizer-v0.1 determinism: encode/decode, save/reload, rebuild."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from juniper_auto.tokenizer.artifacts import CANONICAL_ARTIFACT_DIR, compute_artifact_hashes
from juniper_auto.tokenizer.bpe import train_bpe
from juniper_auto.tokenizer.tokenizer import JuniperTokenizer
from tests.tokenizer_fixtures import DOMAIN_SAMPLES, canonical_tokenizer


@pytest.fixture(scope="module")
def tok():
    return canonical_tokenizer()


def test_repeated_encode_is_identical(tok):
    for domain in DOMAIN_SAMPLES.values():
        for s in domain:
            assert tok.encode(s) == tok.encode(s)


def test_repeated_decode_is_identical(tok):
    ids = tok.encode("deterministic decode\n\tcheck {a: [1,2]}")
    assert tok.decode(ids) == tok.decode(ids)


def test_save_reload_is_byte_identical(tok, tmp_path):
    d1 = tmp_path / "a"
    tok.save(d1)
    from juniper_auto.tokenizer.artifacts import write_hashes

    write_hashes(d1)
    reloaded = JuniperTokenizer.load(d1)
    for domain in DOMAIN_SAMPLES.values():
        for s in domain:
            assert reloaded.encode(s) == tok.encode(s)
    d2 = tmp_path / "b"
    reloaded.save(d2)
    for name in ("merges.txt", "vocab.json", "tokenizer.json", "special_tokens.json"):
        assert (d1 / name).read_bytes() == (d2 / name).read_bytes()


def test_cross_process_encode_matches(tok):
    code = (
        "from juniper_auto.tokenizer import load_canonical_tokenizer;"
        "t=load_canonical_tokenizer();"
        "import json,sys;"
        "print(json.dumps([t.encode(s) for s in "
        "['hello world','def f(x):\\n    return x','{\"a\":[1,2]}','café 🦊']]))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    expected = [tok.encode(s) for s in ["hello world", "def f(x):\n    return x", '{"a":[1,2]}', "café 🦊"]]
    assert json.loads(out.stdout) == expected


def test_special_ids_stable_after_reload(tok, tmp_path):
    tok.save(tmp_path)
    from juniper_auto.tokenizer.artifacts import write_hashes

    write_hashes(tmp_path)
    reloaded = JuniperTokenizer.load(tmp_path)
    assert reloaded._special_id_to_str == tok._special_id_to_str
    assert reloaded.bos_id == 256 and reloaded.eos_id == 257 and reloaded.pad_id == 258


def test_committed_artifact_hashes_match_manifest():
    manifest = json.loads((CANONICAL_ARTIFACT_DIR / "hashes.json").read_text())
    assert manifest["sha256"] == compute_artifact_hashes(CANONICAL_ARTIFACT_DIR)


def test_small_bpe_train_is_order_deterministic():
    words = {
        tuple(bytes([c]) for c in w.encode()): f
        for w, f in [("banana", 9), (" banana", 4), ("bandana", 3), ("ananas", 5), (" anagram", 2)]
    }
    r1 = train_bpe(dict(words), 30, min_pair_frequency=1)
    r2 = train_bpe(dict(reversed(list(words.items()))), 30, min_pair_frequency=1)
    # Insertion order of the word table does not change the learned merges,
    # because selection is argmax by (count, lexicographically smallest pair).
    assert r1.merges == r2.merges


def test_full_rebuild_from_committed_corpus_is_hash_identical(tmp_path):
    from juniper_auto.tokenizer.train import train_tokenizer

    _tok, report = train_tokenizer(out_dir=tmp_path / "rebuild")
    rebuilt = compute_artifact_hashes(tmp_path / "rebuild")
    canonical = compute_artifact_hashes(CANONICAL_ARTIFACT_DIR)
    assert rebuilt == canonical, "deterministic rebuild diverged from the frozen artifact"
    assert report.bpe_steps_completed == 36_352
