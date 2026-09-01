"""Frozen tokenizer artifact layout, hashing, and load-with-verification.

The canonical frozen artifact lives at
``data/tokenizer/ja-tokenizer-v0.1/`` and is fully self-contained: no
network access is required to load it. A per-file SHA-256 manifest
(``hashes.json``) lives alongside the artifact; corruption or a hash
mismatch raises :class:`TokenizerArtifactError` (fail loud, never silently
repair).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from juniper_auto.tokenizer.constants import TOKENIZER_ID
from juniper_auto.tokenizer.tokenizer import ARTIFACT_FILES, JuniperTokenizer, TokenizerArtifactError

REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_ARTIFACT_DIR = REPO_ROOT / "data" / "tokenizer" / TOKENIZER_ID
CANONICAL_CORPUS_DIR = REPO_ROOT / "data" / "tokenizer" / "corpus"
HASHES_FILENAME = "hashes.json"

# Files whose SHA-256 is recorded in the artifact's own hashes.json. The
# eval fixture is hashed here too so a Phase 3 efficiency number can always
# be tied back to the exact corpus it was measured on.
HASHED_ARTIFACT_FILES = tuple(ARTIFACT_FILES)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_artifact_hashes(directory: str | Path) -> dict[str, str]:
    directory = Path(directory)
    out: dict[str, str] = {}
    for name in HASHED_ARTIFACT_FILES:
        p = directory / name
        if not p.is_file():
            raise TokenizerArtifactError(f"artifact file missing for hashing: {name}")
        out[name] = sha256_file(p)
    return out


def write_hashes(directory: str | Path) -> Path:
    directory = Path(directory)
    hashes = compute_artifact_hashes(directory)
    payload = {"algorithm": "sha256", "tokenizer_id": TOKENIZER_ID, "sha256": hashes}
    out = directory / HASHES_FILENAME
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def verify_artifact_hashes(directory: str | Path) -> None:
    directory = Path(directory)
    manifest_path = directory / HASHES_FILENAME
    if not manifest_path.is_file():
        raise TokenizerArtifactError(f"missing artifact hash manifest: {manifest_path}")
    recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
    if recorded.get("tokenizer_id") != TOKENIZER_ID:
        raise TokenizerArtifactError("artifact hash manifest is for a different tokenizer id")
    actual = compute_artifact_hashes(directory)
    if set(recorded.get("sha256", {})) != set(actual):
        raise TokenizerArtifactError(
            "artifact hash manifest file set does not match the artifact directory"
        )
    stale = [k for k in actual if recorded["sha256"][k] != actual[k]]
    if stale:
        raise TokenizerArtifactError(f"corrupt / modified tokenizer artifact files: {stale}")


def load_canonical_tokenizer(*, verify_hashes: bool = True) -> JuniperTokenizer:
    """Load ``ja-tokenizer-v0.1`` from the committed repository artifact."""
    if not CANONICAL_ARTIFACT_DIR.is_dir():
        raise TokenizerArtifactError(
            f"canonical tokenizer artifact not found at {CANONICAL_ARTIFACT_DIR} -- "
            "run scripts/train_tokenizer.py to build it"
        )
    return JuniperTokenizer.load(CANONICAL_ARTIFACT_DIR, verify_hashes=verify_hashes)
