"""Deterministic byte-level BPE: merge learning and merge application.

Pure Python, no third-party tokenizer dependency (see
docs/adr/0010-tokenizer-implementation-choice.md). Everything here is a pure
function of its inputs:

* training input ordering is the caller's ``word_freqs`` iteration order
  (the corpus loader sorts shards and preserves in-file order);
* the merge selected at each step is ``argmax`` by (count, then
  lexicographically smallest byte pair) -- ties are never resolved by hash
  order or ``dict`` order;
* a merge whose resulting bytes already exist as a token is applied to the
  corpus (so training makes progress) but is not recorded as a new
  vocabulary entry -- every recorded merge yields a genuinely new byte
  string, so the learned vocabulary has no shadowed / dead ids;
* merge application (encoding) is greedy lowest-rank-first.
"""

from __future__ import annotations

import heapq
from collections import Counter
from dataclasses import dataclass

Symbols = tuple[bytes, ...]
Pair = tuple[bytes, bytes]


def adjacent_pairs(symbols: Symbols | list[bytes]) -> list[Pair]:
    return [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]


def apply_merges(symbols: Symbols, ranks: dict[Pair, int]) -> Symbols:
    """Greedy lowest-rank-first merge application (encoding path)."""
    if len(symbols) < 2:
        return symbols
    sym = list(symbols)
    while len(sym) >= 2:
        best_i = -1
        best_rank: int | None = None
        for i in range(len(sym) - 1):
            r = ranks.get((sym[i], sym[i + 1]))
            if r is not None and (best_rank is None or r < best_rank):
                best_rank = r
                best_i = i
        if best_i < 0:
            break
        sym[best_i : best_i + 2] = [sym[best_i] + sym[best_i + 1]]
    return tuple(sym)


def _merge_in_word(sym: list[bytes], a: bytes, b: bytes) -> list[bytes]:
    ab = a + b
    out: list[bytes] = []
    i = 0
    n = len(sym)
    while i < n:
        if i < n - 1 and sym[i] == a and sym[i + 1] == b:
            out.append(ab)
            i += 2
        else:
            out.append(sym[i])
            i += 1
    return out


@dataclass
class BPETrainingResult:
    merges: list[Pair]  # learned order; rank == index
    steps_requested: int
    steps_completed: int
    stopped_early: bool
    stop_reason: str
    collisions_skipped: int
    merges_at_primary_threshold: int  # learned before dropping to the tail threshold
    tail_fill_used: bool


def train_bpe(
    word_freqs: dict[Symbols, int],
    num_merges: int,
    *,
    min_pair_frequency: int = 2,
    tail_fill: bool = True,
    log_every: int = 0,
    logger=None,
) -> BPETrainingResult:
    """Learn ``num_merges`` byte-level merges from a word-frequency table.

    ``word_freqs`` maps a pre-token (a tuple of single-byte ``bytes``) to its
    corpus frequency. Returns the ordered merge list; ``rank == index``.
    """
    words: list[list[bytes]] = [list(k) for k in word_freqs]
    freqs: list[int] = list(word_freqs.values())

    produced: set[bytes] = {bytes([b]) for b in range(256)}

    pair_counts: Counter[Pair] = Counter()
    pair_where: dict[Pair, set[int]] = {}
    for wid, sym in enumerate(words):
        f = freqs[wid]
        for p in adjacent_pairs(sym):
            pair_counts[p] += f
            pair_where.setdefault(p, set()).add(wid)

    heap: list[tuple[int, Pair]] = [(-c, p) for p, c in pair_counts.items()]
    heapq.heapify(heap)

    merges: list[Pair] = []
    collisions_skipped = 0
    stop_reason = "reached requested merge count"
    stopped_early = False
    active_threshold = min_pair_frequency
    merges_at_primary_threshold = -1
    tail_fill_used = False
    max_iterations = num_merges * 4 + 5_000_000

    iterations = 0
    while len(merges) < num_merges:
        iterations += 1
        if iterations > max_iterations:
            stopped_early = True
            stop_reason = f"iteration cap hit after {len(merges)} recorded merges"
            break

        best: Pair | None = None
        while heap:
            neg_c, p = heapq.heappop(heap)
            cur = pair_counts.get(p, 0)
            if cur != -neg_c:
                continue  # stale heap entry
            if cur < active_threshold:
                heapq.heappush(heap, (neg_c, p))
                break
            best = p
            break
        if best is None:
            if tail_fill and active_threshold > 1:
                merges_at_primary_threshold = len(merges)
                tail_fill_used = True
                active_threshold = 1
                if logger is not None:
                    logger.info(
                        "bpe.train.tail_fill_start "
                        f"merges_done={len(merges)} dropping_threshold_to=1"
                    )
                continue
            stopped_early = True
            stop_reason = (
                f"no byte pair reaches threshold={active_threshold} "
                f"after {len(merges)} merges"
            )
            break

        a, b = best
        merged = a + b
        is_new = merged not in produced
        if is_new:
            merges.append(best)
            produced.add(merged)
        else:
            collisions_skipped += 1

        affected = pair_where.pop(best, set())
        del pair_counts[best]

        touched: set[Pair] = set()
        for wid in affected:
            sym = words[wid]
            if not any(sym[i] == a and sym[i + 1] == b for i in range(len(sym) - 1)):
                continue
            f = freqs[wid]
            for p in adjacent_pairs(sym):
                if p == best:
                    continue
                pair_counts[p] -= f
                touched.add(p)
            new_sym = _merge_in_word(sym, a, b)
            words[wid] = new_sym
            for p in adjacent_pairs(new_sym):
                pair_counts[p] += f
                pair_where.setdefault(p, set()).add(wid)
                touched.add(p)

        for p in touched:
            c = pair_counts.get(p, 0)
            if c > 0:
                heapq.heappush(heap, (-c, p))
            else:
                pair_counts.pop(p, None)

        if log_every and logger is not None and len(merges) % log_every == 0 and is_new:
            logger.info(
                "bpe.train.progress "
                f"merges_done={len(merges)} merges_total={num_merges} "
                f"distinct_pairs={len(pair_counts)} collisions_skipped={collisions_skipped}"
            )

    if merges_at_primary_threshold < 0:
        merges_at_primary_threshold = len(merges)
    return BPETrainingResult(
        merges=merges,
        steps_requested=num_merges,
        steps_completed=len(merges),
        stopped_early=stopped_early,
        stop_reason=stop_reason,
        collisions_skipped=collisions_skipped,
        merges_at_primary_threshold=merges_at_primary_threshold,
        tail_fill_used=tail_fill_used,
    )
