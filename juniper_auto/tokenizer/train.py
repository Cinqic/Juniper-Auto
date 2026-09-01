"""End-to-end ``ja-tokenizer-v0.1`` training: committed corpus shards ->
word-frequency table -> byte-level BPE merges -> frozen artifact directory.

Deterministic: same committed shards + same config + same seed => identical
``merges.txt`` / ``vocab.json`` / ``tokenizer.json`` and therefore identical
SHA-256 hashes. Proven by
``scripts/run_phase3_experiment.py rebuild-determinism``.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from juniper_auto.tokenizer import constants as C
from juniper_auto.tokenizer.artifacts import CANONICAL_ARTIFACT_DIR, write_hashes
from juniper_auto.tokenizer.bpe import Symbols, train_bpe
from juniper_auto.tokenizer.bytelevel import pretokenize
from juniper_auto.tokenizer.corpus import (
    CORPUS_DIR,
    CORPUS_SEED,
    corpus_manifest_sha256,
    load_corpus_shards,
)
from juniper_auto.tokenizer.tokenizer import JuniperTokenizer

TRAINING_SEED = 20260901
# Primary merges require a byte pair to occur at least twice; once no pair
# reaches that threshold the trainer drops to frequency 1 to fill the exact
# 36,352-merge budget deterministically from the committed corpus. The
# frequency-1 tail is a documented accepted limitation (a production retrain
# would use a much larger organic corpus) -- see
# docs/phases/phase-3-tokenizer.md.
MIN_PAIR_FREQUENCY = 2


@dataclass
class TrainingReport:
    tokenizer_id: str
    vocab_size: int
    num_merges: int
    corpus_seed: int
    training_seed: int
    min_pair_frequency: int
    corpus_manifest_sha256: str
    corpus_total_bytes: int
    corpus_shard_count: int
    distinct_pretokens: int
    total_pretoken_occurrences: int
    bpe_steps_requested: int
    bpe_steps_completed: int
    bpe_stopped_early: bool
    bpe_stop_reason: str
    bpe_collisions_skipped: int
    bpe_merges_at_primary_threshold: int
    bpe_tail_fill_used: bool
    wall_seconds_corpus_load: float
    wall_seconds_pretokenize: float
    wall_seconds_bpe: float
    wall_seconds_total: float
    artifact_dir: str
    artifact_hashes: dict = field(default_factory=dict)


def build_word_frequencies(shards: list[tuple[str, str]]) -> Counter[Symbols]:
    freqs: Counter[Symbols] = Counter()
    for _name, text in shards:
        for chunk in pretokenize(text):
            freqs[tuple(bytes([b]) for b in chunk.encode("utf-8"))] += 1
    return freqs


def train_tokenizer(
    *,
    corpus_dir: Path = CORPUS_DIR,
    out_dir: Path = CANONICAL_ARTIFACT_DIR,
    training_seed: int = TRAINING_SEED,
    min_pair_frequency: int = MIN_PAIR_FREQUENCY,
    logger=None,
) -> tuple[JuniperTokenizer, TrainingReport]:
    t0 = time.perf_counter()

    ta = time.perf_counter()
    shards = load_corpus_shards(corpus_dir)
    t_load = time.perf_counter() - ta

    manifest_sha = corpus_manifest_sha256()
    corpus_bytes = sum(len(text.encode("utf-8")) for _, text in shards)

    tb = time.perf_counter()
    word_freqs = build_word_frequencies(shards)
    t_pre = time.perf_counter() - tb

    total_occurrences = sum(word_freqs.values())

    tc = time.perf_counter()
    result = train_bpe(
        dict(word_freqs),
        C.NUM_MERGES,
        min_pair_frequency=min_pair_frequency,
        tail_fill=True,
        log_every=2000,
        logger=logger,
    )
    t_bpe = time.perf_counter() - tc

    if result.steps_completed != C.NUM_MERGES:
        raise RuntimeError(
            f"BPE training produced {result.steps_completed} merges, need exactly "
            f"{C.NUM_MERGES}: {result.stop_reason}"
        )

    metadata = {
        "corpus_manifest": "data/tokenizer/corpus/corpus-manifest.json",
        "corpus_manifest_sha256": manifest_sha,
        "corpus_seed": CORPUS_SEED,
        "training_seed": training_seed,
        "min_pair_frequency": min_pair_frequency,
        "algorithm": C.ALGORITHM,
        "pretokenizer_pattern": C.PRETOKEN_PATTERN,
        "normalization": "none",
        "byte_fallback": True,
        "collisions_skipped": result.collisions_skipped,
        "merges_at_primary_threshold": result.merges_at_primary_threshold,
        "tail_fill_used": result.tail_fill_used,
    }

    tokenizer = JuniperTokenizer(merges=result.merges, metadata=metadata)
    out_dir = Path(out_dir)
    tokenizer.save(out_dir)
    hashes_path = write_hashes(out_dir)
    import json

    artifact_hashes = json.loads(hashes_path.read_text())["sha256"]

    total = time.perf_counter() - t0
    report = TrainingReport(
        tokenizer_id=C.TOKENIZER_ID,
        vocab_size=C.VOCAB_SIZE,
        num_merges=len(result.merges),
        corpus_seed=CORPUS_SEED,
        training_seed=training_seed,
        min_pair_frequency=min_pair_frequency,
        corpus_manifest_sha256=manifest_sha,
        corpus_total_bytes=corpus_bytes,
        corpus_shard_count=len(shards),
        distinct_pretokens=len(word_freqs),
        total_pretoken_occurrences=total_occurrences,
        bpe_steps_requested=result.steps_requested,
        bpe_steps_completed=result.steps_completed,
        bpe_stopped_early=result.stopped_early,
        bpe_stop_reason=result.stop_reason,
        bpe_collisions_skipped=result.collisions_skipped,
        bpe_merges_at_primary_threshold=result.merges_at_primary_threshold,
        bpe_tail_fill_used=result.tail_fill_used,
        wall_seconds_corpus_load=round(t_load, 3),
        wall_seconds_pretokenize=round(t_pre, 3),
        wall_seconds_bpe=round(t_bpe, 3),
        wall_seconds_total=round(total, 3),
        artifact_dir=str(out_dir),
        artifact_hashes=artifact_hashes,
    )
    return tokenizer, report
