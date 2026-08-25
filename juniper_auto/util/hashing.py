"""SHA-256 hashing for the Phase 0 artifact hash manifest.

Used by scripts/hash_manifest.py (to generate/update the manifest) and
scripts/validate_repo.py (to detect drift between the manifest and the
actual file contents). The manifest itself is never included in its own
hash list -- that would be recursive self-hashing.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# Relative to repo root. Kept as an explicit list (not a glob) so adding a
# new hashed artifact is a deliberate, reviewable change.
PHASE_0_HASHED_ARTIFACTS: list[str] = [
    "configs/architecture/ja150m-v0.1.yaml",
    "configs/architecture/ja150m-v0.1-dense.yaml",
    "docs/research/project-charter.md",
    "docs/research/project-governance.md",
    "requirements-lock.txt",
    "juniper_auto/accounting/parameter_count.py",
    "juniper_auto/config/schema.py",
    "juniper_auto/config/frozen.py",
    "scripts/validate_repo.py",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_hashes(repo_root: Path, relative_paths: list[str] = PHASE_0_HASHED_ARTIFACTS) -> dict[str, str]:
    result = {}
    for rel in relative_paths:
        full = repo_root / rel
        if not full.is_file():
            raise FileNotFoundError(f"artifact listed for hashing does not exist: {rel}")
        result[rel] = sha256_file(full)
    return result
