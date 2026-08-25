import yaml

REQUIRED_FIELDS = {
    "experiment_id",
    "date",
    "phase",
    "hypothesis",
    "git_commit",
    "architecture_id",
    "tokenizer_id",
    "dataset_id",
    "sampler_id",
    "starting_checkpoint",
    "configuration",
    "seed",
    "environment",
    "token_budget",
    "status",
    "result",
    "conclusion",
    "artifact_locations",
}

VALID_STATUSES = {"planned", "running", "completed", "failed", "negative-result"}

# Values that mean "this doesn't exist yet" -- allowed. Anything else in a
# tokenizer/dataset/checkpoint field must not look like a fabricated real
# identity for something Phase 0 never created.
HONEST_NOT_YET_CREATED_VALUES = {"not-yet-created", "not-applicable"}


def _load_registry(repo_root):
    path = repo_root / "experiments" / "registry.yaml"
    with path.open() as f:
        return yaml.safe_load(f)


def test_registry_exists_and_parses(repo_root):
    entries = _load_registry(repo_root)
    assert isinstance(entries, list)
    assert len(entries) >= 1


def test_every_entry_has_required_fields(repo_root):
    entries = _load_registry(repo_root)
    for entry in entries:
        missing = REQUIRED_FIELDS - set(entry.keys())
        assert not missing, f"{entry.get('experiment_id')}: missing fields {missing}"


def test_experiment_ids_are_unique(repo_root):
    entries = _load_registry(repo_root)
    ids = [e["experiment_id"] for e in entries]
    assert len(ids) == len(set(ids)), f"duplicate experiment_id values: {ids}"


def test_status_values_are_valid(repo_root):
    entries = _load_registry(repo_root)
    for entry in entries:
        assert entry["status"] in VALID_STATUSES, f"{entry['experiment_id']}: invalid status {entry['status']!r}"


def test_phase_0_entries_do_not_fabricate_tokenizer_or_dataset(repo_root):
    """Phase 0 has not created a tokenizer, dataset, or checkpoint -- every
    Phase 0 registry entry must say so honestly, not invent a plausible id."""
    entries = _load_registry(repo_root)
    for entry in entries:
        if entry["phase"] != "phase-0":
            continue
        assert entry["tokenizer_id"] in HONEST_NOT_YET_CREATED_VALUES, entry["experiment_id"]
        assert entry["dataset_id"] in HONEST_NOT_YET_CREATED_VALUES, entry["experiment_id"]
        assert entry["starting_checkpoint"] in HONEST_NOT_YET_CREATED_VALUES, entry["experiment_id"]
