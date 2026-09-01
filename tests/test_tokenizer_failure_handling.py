"""ja-tokenizer-v0.1 fails closed on malformed / inconsistent artifacts."""

from __future__ import annotations

import json
import shutil

import pytest

from juniper_auto.tokenizer.artifacts import CANONICAL_ARTIFACT_DIR, verify_artifact_hashes, write_hashes
from juniper_auto.tokenizer.tokenizer import MERGES_HEADER, JuniperTokenizer, TokenizerArtifactError


@pytest.fixture
def artifact_copy(tmp_path):
    dst = tmp_path / "tok"
    shutil.copytree(CANONICAL_ARTIFACT_DIR, dst)
    return dst


def test_missing_file_fails(artifact_copy):
    (artifact_copy / "merges.txt").unlink()
    with pytest.raises(TokenizerArtifactError, match="missing files"):
        JuniperTokenizer.load(artifact_copy)


def test_wrong_vocab_size_in_tokenizer_json_fails(artifact_copy):
    tok = json.loads((artifact_copy / "tokenizer.json").read_text())
    tok["vocab_size"] = 32_768
    (artifact_copy / "tokenizer.json").write_text(json.dumps(tok))
    write_hashes(artifact_copy)
    with pytest.raises(TokenizerArtifactError, match="vocab_size"):
        JuniperTokenizer.load(artifact_copy)


def test_too_few_merges_fails(artifact_copy):
    lines = (artifact_copy / "merges.txt").read_text().split("\n")
    (artifact_copy / "merges.txt").write_text("\n".join(lines[:-50]))
    write_hashes(artifact_copy)
    with pytest.raises(TokenizerArtifactError, match="exactly 36352 merges"):
        JuniperTokenizer.load(artifact_copy)


def test_missing_merges_header_fails(artifact_copy):
    body = (artifact_copy / "merges.txt").read_text().split("\n", 1)[1]
    (artifact_copy / "merges.txt").write_text(body)
    write_hashes(artifact_copy)
    with pytest.raises(TokenizerArtifactError, match="header sentinel"):
        JuniperTokenizer.load(artifact_copy)


def test_duplicate_merge_producing_existing_bytes_fails(artifact_copy):
    lines = (artifact_copy / "merges.txt").read_text().split("\n")
    lines[1] = lines[2]  # first merge now duplicates the second's output
    (artifact_copy / "merges.txt").write_text("\n".join(lines))
    write_hashes(artifact_copy)
    with pytest.raises(TokenizerArtifactError):
        JuniperTokenizer.load(artifact_copy)


def test_corrupted_vocab_json_fails(artifact_copy):
    (artifact_copy / "vocab.json").write_text('{"broken": ')
    write_hashes(artifact_copy)
    with pytest.raises(json.JSONDecodeError):
        JuniperTokenizer.load(artifact_copy)


def test_vocab_merge_disagreement_fails(artifact_copy):
    vocab = json.loads((artifact_copy / "vocab.json").read_text())
    # flip one learned id to a wrong value
    for k, v in list(vocab.items()):
        if v == 20_000:
            vocab[k] = 20_001
            break
    (artifact_copy / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False))
    write_hashes(artifact_copy)
    with pytest.raises(TokenizerArtifactError, match="inconsistent with a merge replay"):
        JuniperTokenizer.load(artifact_copy)


def test_mismatched_tokenizer_id_fails(artifact_copy):
    tok = json.loads((artifact_copy / "tokenizer.json").read_text())
    tok["tokenizer_id"] = "ja-tokenizer-v0.2"
    (artifact_copy / "tokenizer.json").write_text(json.dumps(tok))
    write_hashes(artifact_copy)
    with pytest.raises(TokenizerArtifactError, match="expected"):
        JuniperTokenizer.load(artifact_copy)


def test_special_token_map_drift_fails(artifact_copy):
    tok = json.loads((artifact_copy / "tokenizer.json").read_text())
    tok["special_tokens"]["<|system|>"] = 999
    (artifact_copy / "tokenizer.json").write_text(json.dumps(tok))
    write_hashes(artifact_copy)
    with pytest.raises(TokenizerArtifactError, match="special-token map"):
        JuniperTokenizer.load(artifact_copy)


def test_hash_mismatch_detected(artifact_copy):
    lines = (artifact_copy / "merges.txt").read_text().split("\n")
    lines[1], lines[2] = lines[2], lines[1]  # real content change
    (artifact_copy / "merges.txt").write_text("\n".join(lines))
    # hashes.json now stale
    with pytest.raises(TokenizerArtifactError, match="corrupt / modified"):
        verify_artifact_hashes(artifact_copy)


def test_missing_hash_manifest_fails(artifact_copy):
    (artifact_copy / "hashes.json").unlink()
    with pytest.raises(TokenizerArtifactError, match="missing artifact hash manifest"):
        verify_artifact_hashes(artifact_copy)


def test_hash_manifest_for_wrong_tokenizer_fails(artifact_copy):
    h = json.loads((artifact_copy / "hashes.json").read_text())
    h["tokenizer_id"] = "ja-tokenizer-v9"
    (artifact_copy / "hashes.json").write_text(json.dumps(h))
    with pytest.raises(TokenizerArtifactError, match="different tokenizer id"):
        verify_artifact_hashes(artifact_copy)


def test_architecture_vocab_mismatch_fails(artifact_copy):
    from types import SimpleNamespace

    t = JuniperTokenizer.load(artifact_copy)
    bad = SimpleNamespace(architecture_id="x", embeddings=SimpleNamespace(vocab_size=32_000))
    with pytest.raises(TokenizerArtifactError, match="!="):
        t.assert_model_compatible(bad)
