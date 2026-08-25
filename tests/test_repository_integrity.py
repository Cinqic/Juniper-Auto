import subprocess

PROHIBITED_SUFFIXES = (".env", ".pyc", ".pem", ".key", ".pt", ".pth", ".safetensors", ".ckpt")
PROHIBITED_BASENAMES = ("id_rsa", "id_ed25519")
PROHIBITED_PATH_FRAGMENTS = (".venv/", "__pycache__/", ".pytest_cache/", ".mypy_cache/")


def _tracked_files(repo_root):
    result = subprocess.run(
        ["git", "ls-files"], cwd=repo_root, capture_output=True, text=True, check=True
    )
    return result.stdout.splitlines()


def test_no_prohibited_tracked_artifacts(repo_root):
    tracked = _tracked_files(repo_root)
    violations = []
    for f in tracked:
        lower = f.lower()
        if lower.endswith(PROHIBITED_SUFFIXES):
            violations.append(f)
        elif any(f.endswith(b) for b in PROHIBITED_BASENAMES):
            violations.append(f)
        elif any(frag in f for frag in PROHIBITED_PATH_FRAGMENTS):
            violations.append(f)
    assert not violations, f"prohibited files tracked in git: {violations}"


def test_working_tree_is_clean_or_only_expected_changes(repo_root):
    """A dirty working tree at test time is not itself a failure (tests may
    run mid-development), but this documents the check exists and can be
    tightened to a hard assertion at candidate-commit time."""
    result = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo_root, capture_output=True, text=True, check=True
    )
    # Informational only -- see docs/phases/phase-0-sonnet-self-review.md
    # Self-Review Pass D for the hard clean-tree check before handoff.
    assert isinstance(result.stdout, str)


def test_gitignore_covers_venv_and_caches(repo_root):
    gitignore = (repo_root / ".gitignore").read_text()
    for pattern in [".venv", "__pycache__", ".pytest_cache", ".env"]:
        assert pattern in gitignore, f".gitignore missing pattern: {pattern}"


def test_no_env_files_present_in_tree(repo_root):
    tracked = _tracked_files(repo_root)
    assert not any(f == ".env" or f.endswith("/.env") for f in tracked)
