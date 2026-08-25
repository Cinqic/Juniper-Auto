import yaml

from juniper_auto.util.hashing import PHASE_0_HASHED_ARTIFACTS, compute_hashes


def test_hash_manifest_exists_and_parses(repo_root):
    path = repo_root / "manifests" / "phase-0-artifact-hashes.yaml"
    assert path.is_file()
    with path.open() as f:
        manifest = yaml.safe_load(f)
    assert manifest["algorithm"] == "sha256"
    assert isinstance(manifest["sha256"], dict)


def test_hash_manifest_covers_exactly_the_hashed_artifact_list(repo_root):
    path = repo_root / "manifests" / "phase-0-artifact-hashes.yaml"
    with path.open() as f:
        manifest = yaml.safe_load(f)
    assert set(manifest["sha256"].keys()) == set(PHASE_0_HASHED_ARTIFACTS)


def test_hash_manifest_does_not_hash_itself(repo_root):
    path = repo_root / "manifests" / "phase-0-artifact-hashes.yaml"
    with path.open() as f:
        manifest = yaml.safe_load(f)
    assert "manifests/phase-0-artifact-hashes.yaml" not in manifest["sha256"]


def test_recorded_hashes_match_actual_file_contents(repo_root):
    path = repo_root / "manifests" / "phase-0-artifact-hashes.yaml"
    with path.open() as f:
        manifest = yaml.safe_load(f)
    actual = compute_hashes(repo_root)
    stale = {k: (v, actual[k]) for k, v in manifest["sha256"].items() if v != actual[k]}
    assert not stale, f"stale hashes (recorded, actual): {stale}"


def test_hashes_are_valid_sha256_hex(repo_root):
    path = repo_root / "manifests" / "phase-0-artifact-hashes.yaml"
    with path.open() as f:
        manifest = yaml.safe_load(f)
    for rel, digest in manifest["sha256"].items():
        assert len(digest) == 64, f"{rel}: hash is not 64 hex chars: {digest}"
        int(digest, 16)  # raises if not valid hex
