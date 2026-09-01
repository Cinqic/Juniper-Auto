#!/usr/bin/env python3
"""(Re)build the committed ``ja-tokenizer-v0.1`` training-corpus shards and
manifest under ``data/tokenizer/corpus/``.

    python scripts/build_tokenizer_corpus.py

Deterministic given the repository content, the running Python's stdlib
(version recorded in the manifest), and the fixed corpus seed. The committed
shards are the canonical training input; this script exists so the
derivation is auditable and repeatable, not because the shards are
regenerated on every train.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from juniper_auto.tokenizer.corpus import build_corpus  # noqa: E402


def main() -> int:
    manifest = build_corpus()
    print(
        f"corpus rebuilt: {manifest['shard_count']} shards, "
        f"{manifest['total_bytes']:,} bytes, categories={manifest['categories']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
