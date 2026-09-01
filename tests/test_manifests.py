import yaml


def test_frozen_artifacts_manifest_parses(repo_root):
    path = repo_root / "manifests" / "frozen-artifacts.yaml"
    assert path.is_file()
    with path.open() as f:
        manifest = yaml.safe_load(f)
    assert isinstance(manifest, dict)


def test_frozen_artifacts_manifest_has_required_sections(repo_root):
    path = repo_root / "manifests" / "frozen-artifacts.yaml"
    with path.open() as f:
        manifest = yaml.safe_load(f)
    required = {
        "architecture",
        "tokenizer",
        "special_token_map",
        "runtime_protocol",
        "tool_schemas",
        "memory_schema",
        "state_schema",
        "permission_policy",
        "pretraining_dataset",
        "post_training_dataset",
        "evaluation_suite",
        "base_checkpoint",
        "instruction_checkpoint",
        "autonomous_system_release",
    }
    missing = required - set(manifest.keys())
    assert not missing, f"frozen-artifacts.yaml missing sections: {missing}"


def test_architecture_entries_are_frozen(repo_root):
    path = repo_root / "manifests" / "frozen-artifacts.yaml"
    with path.open() as f:
        manifest = yaml.safe_load(f)
    assert manifest["architecture"]["sparse"]["status"] == "frozen"
    assert manifest["architecture"]["sparse"]["id"] == "ja150m-v0.1"
    assert manifest["architecture"]["dense_baseline"]["status"] == "frozen"
    assert manifest["architecture"]["dense_baseline"]["id"] == "ja150m-v0.1-dense"


def test_not_yet_created_artifacts_have_no_fabricated_status(repo_root):
    """Everything that doesn't exist yet must say so honestly."""
    path = repo_root / "manifests" / "frozen-artifacts.yaml"
    with path.open() as f:
        manifest = yaml.safe_load(f)
    valid_statuses = {"frozen", "planned", "not-yet-created", "superseded"}

    def check(node, path_str=""):
        if isinstance(node, dict) and "status" in node:
            assert node["status"] in valid_statuses, f"{path_str}: invalid status {node['status']!r}"
        elif isinstance(node, dict):
            for k, v in node.items():
                check(v, f"{path_str}.{k}")

    check(manifest)

    # Phase 3 freezes `tokenizer` and `special_token_map` (see
    # docs/phases/phase-3-tokenizer.md). Every other future-artifact category
    # must still honestly say not-yet-created.
    still_not_yet_created = [
        "runtime_protocol",
        "tool_schemas",
        "memory_schema",
        "state_schema",
        "permission_policy",
        "pretraining_dataset",
        "post_training_dataset",
        "evaluation_suite",
        "base_checkpoint",
        "instruction_checkpoint",
        "autonomous_system_release",
    ]
    for key in still_not_yet_created:
        assert manifest[key]["status"] == "not-yet-created", f"{key} should be not-yet-created"
    assert manifest["tokenizer"]["status"] == "frozen"
    assert manifest["tokenizer"]["id"] == "ja-tokenizer-v0.1"
    assert manifest["special_token_map"]["status"] == "frozen"
