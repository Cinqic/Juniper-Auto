#!/usr/bin/env python3
"""Train (or retrain) ``ja-tokenizer-v0.1`` from the committed corpus into
the frozen artifact directory ``data/tokenizer/ja-tokenizer-v0.1/``.

    python scripts/train_tokenizer.py [--out DIR]

Deterministic: same committed shards + config + seed => identical artifact
hashes. After running this, regenerate the Phase 3 manifests
(scripts/generate_phase3_test_manifest.py then scripts/hash_manifest.py
--phase 3) and commit everything together.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from juniper_auto.tokenizer.artifacts import CANONICAL_ARTIFACT_DIR  # noqa: E402
from juniper_auto.tokenizer.train import train_tokenizer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(CANONICAL_ARTIFACT_DIR))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(message)s")
    _tok, report = train_tokenizer(out_dir=Path(args.out), logger=logging.getLogger("tokenizer.train"))
    print(json.dumps(dataclasses.asdict(report), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
