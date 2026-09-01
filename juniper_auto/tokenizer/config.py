"""Schema + cross-checks for ``configs/tokenizer/ja-tokenizer-v0.1.yaml``.

The YAML is declarative; this module proves it agrees with
``juniper_auto.tokenizer.constants`` (the code source of truth) and, when
present, with the frozen artifact directory. Any drift raises loudly.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from juniper_auto.tokenizer import constants as C
from juniper_auto.tokenizer.tokenizer import TokenizerArtifactError

REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_CONFIG_PATH = REPO_ROOT / "configs" / "tokenizer" / "ja-tokenizer-v0.1.yaml"


def load_tokenizer_config(path: str | Path = CANONICAL_CONFIG_PATH) -> dict:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return raw


def validate_tokenizer_config(
    path: str | Path = CANONICAL_CONFIG_PATH, *, check_artifact: bool = True
) -> dict:
    cfg = load_tokenizer_config(path)

    def _require(cond: bool, msg: str) -> None:
        if not cond:
            raise TokenizerArtifactError(f"tokenizer config: {msg}")

    _require(cfg.get("tokenizer_id") == C.TOKENIZER_ID, "tokenizer_id drift")
    _require(cfg.get("algorithm") == C.ALGORITHM, "algorithm drift")
    _require(cfg.get("vocab_size") == C.VOCAB_SIZE, "vocab_size drift")
    _require(cfg["byte_level"]["base_alphabet_size"] == C.BYTE_TOKEN_COUNT, "base alphabet size")
    _require(cfg["byte_level"]["byte_fallback"] is True, "byte_fallback must be true")
    _require(cfg["byte_level"]["unk_token"] is None, "unk_token must be null")
    _require(cfg["normalization"]["kind"] == "none", "normalization must be 'none'")
    _require(cfg["pre_tokenization"]["pattern"] == C.PRETOKEN_PATTERN, "pretokenizer pattern drift")

    layout = cfg["layout"]
    _require(layout["byte_tokens"] == [0, 255], "byte_tokens layout")
    _require(layout["control_block"] == [C.CONTROL_BLOCK_START, C.CONTROL_BLOCK_END], "control_block layout")
    _require(layout["core_control"] == [256, 270], "core_control layout")
    _require(
        layout["reserved_control"] == [C.RESERVED_CONTROL_START, C.RESERVED_CONTROL_END],
        "reserved_control layout",
    )
    _require(layout["learned_vocab"] == [C.LEARNED_VOCAB_START, C.LEARNED_VOCAB_END], "learned_vocab layout")
    _require(layout["num_merges"] == C.NUM_MERGES, "num_merges layout")

    _require(cfg["core_control_tokens"] == {s: i for s, i in C.CORE_CONTROL_TOKENS}, "core control map drift")
    rr = cfg["reserved_control_range"]
    _require(
        (rr["start"], rr["end"], rr["count"]) == (C.RESERVED_CONTROL_START, C.RESERVED_CONTROL_END, C.RESERVED_CONTROL_COUNT),
        "reserved control range drift",
    )
    _require(
        cfg["model_compatibility"]["requires_vocab_size"] == C.VOCAB_SIZE,
        "model_compatibility.requires_vocab_size drift",
    )

    if check_artifact:
        from juniper_auto.tokenizer.artifacts import CANONICAL_ARTIFACT_DIR, verify_artifact_hashes

        if CANONICAL_ARTIFACT_DIR.is_dir():
            verify_artifact_hashes(CANONICAL_ARTIFACT_DIR)

    return cfg
