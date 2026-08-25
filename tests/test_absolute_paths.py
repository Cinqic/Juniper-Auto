"""Detects unjustified dependencies on host-specific absolute paths in
versioned configuration/scripts. Documentation may mention illustrative
paths when clearly marked as examples (the environment specification and
self-review report, which record what was literally observed on FLOWBOX,
are the documented exception)."""

import subprocess
from pathlib import Path

ALLOWED_TO_MENTION_HOST_PATH = {
    "docs/architecture/environment-specification.md",
    "docs/phases/phase-0-sonnet-self-review.md",
    "docs/phases/phase-0-foundation.md",
}


def _tracked_files(repo_root):
    result = subprocess.run(
        ["git", "ls-files"], cwd=repo_root, capture_output=True, text=True, check=True
    )
    return result.stdout.splitlines()


def test_no_unjustified_absolute_home_paths(repo_root):
    home = str(Path.home())
    tracked = _tracked_files(repo_root)
    violations = []
    for rel in tracked:
        if rel in ALLOWED_TO_MENTION_HOST_PATH or rel.startswith(".git/"):
            continue
        full = repo_root / rel
        if not full.is_file():
            continue
        try:
            text = full.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if home in text:
            violations.append(rel)
    assert not violations, (
        f"files reference the absolute host path {home!r} without being in the "
        f"documented illustrative-path allowlist: {violations}"
    )


def test_config_files_contain_no_absolute_filesystem_paths(repo_root):
    """Architecture configs specifically must be portable -- they should
    never contain an absolute filesystem path at all."""
    for rel in [
        "configs/architecture/ja150m-v0.1.yaml",
        "configs/architecture/ja150m-v0.1-dense.yaml",
    ]:
        text = (repo_root / rel).read_text()
        assert "/home/" not in text
        assert "/Users/" not in text
        assert "C:\\" not in text
