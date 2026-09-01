"""ja-tokenizer-v0.1 canonical config validation + model compatibility."""

from __future__ import annotations

import pytest

from juniper_auto.config import load_architecture_config
from juniper_auto.tokenizer import constants as C
from juniper_auto.tokenizer.config import CANONICAL_CONFIG_PATH, validate_tokenizer_config
from juniper_auto.tokenizer.tokenizer import TokenizerArtifactError
from tests.tokenizer_fixtures import canonical_tokenizer


def test_canonical_config_validates():
    cfg = validate_tokenizer_config()
    assert cfg["tokenizer_id"] == "ja-tokenizer-v0.1"
    assert cfg["vocab_size"] == 36_864
    assert cfg["normalization"]["kind"] == "none"
    assert cfg["byte_level"]["unk_token"] is None


def test_config_drift_is_rejected(tmp_path):
    text = CANONICAL_CONFIG_PATH.read_text().replace("vocab_size: 36864", "vocab_size: 36000")
    bad = tmp_path / "bad.yaml"
    bad.write_text(text)
    with pytest.raises(TokenizerArtifactError, match="vocab_size"):
        validate_tokenizer_config(bad, check_artifact=False)


def test_pretokenizer_pattern_in_config_matches_code():
    cfg = validate_tokenizer_config()
    assert cfg["pre_tokenization"]["pattern"] == C.PRETOKEN_PATTERN


@pytest.mark.parametrize("arch", ["ja150m-v0.1", "ja150m-v0.1-dense"])
def test_tokenizer_vocab_equals_model_vocab(arch, repo_root):
    cfg = load_architecture_config(repo_root / "configs" / "architecture" / f"{arch}.yaml")
    assert cfg.embeddings.vocab_size == 36_864 == C.VOCAB_SIZE
    canonical_tokenizer().assert_model_compatible(cfg)


def test_sparse_and_dense_and_tokenizer_all_agree(repo_root):
    sparse = load_architecture_config(repo_root / "configs/architecture/ja150m-v0.1.yaml")
    dense = load_architecture_config(repo_root / "configs/architecture/ja150m-v0.1-dense.yaml")
    assert sparse.embeddings.vocab_size == dense.embeddings.vocab_size == canonical_tokenizer().vocab_size


def test_phase_3_required_docs_exist(repo_root):
    for rel in [
        "docs/phases/phase-3-tokenizer.md",
        "docs/phases/phase-3-requirements-traceability.md",
        "docs/phases/phase-3-sonnet-self-review.md",
        "docs/recovery/phase-3.md",
        "docs/architecture/tokenizer-design.md",
        "docs/adr/0010-tokenizer-implementation-choice.md",
        "docs/adr/0011-tokenizer-special-token-and-reserved-id-layout.md",
        "docs/adr/0012-tokenizer-normalization-and-pretokenization-policy.md",
        "configs/tokenizer/ja-tokenizer-v0.1.yaml",
        "data/tokenizer/corpus/corpus-manifest.json",
        "data/tokenizer/eval/held-out-eval-fixture.json",
    ]:
        assert (repo_root / rel).is_file(), f"missing required Phase 3 artifact: {rel}"
