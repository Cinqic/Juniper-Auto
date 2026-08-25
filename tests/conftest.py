from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def sparse_config_path(repo_root: Path) -> Path:
    return repo_root / "configs" / "architecture" / "ja150m-v0.1.yaml"


@pytest.fixture(scope="session")
def dense_config_path(repo_root: Path) -> Path:
    return repo_root / "configs" / "architecture" / "ja150m-v0.1-dense.yaml"
