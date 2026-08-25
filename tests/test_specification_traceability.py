import re


def _section(text: str, heading: str) -> str:
    start = text.index(heading) + len(heading)
    match = re.search(r"^## ", text[start:], flags=re.MULTILINE)
    return text[start : start + match.start()] if match else text[start:]


def test_charter_preserves_exactly_fourteen_secondary_questions(repo_root):
    text = (repo_root / "docs/research/project-charter.md").read_text()
    section = _section(text, "## Secondary research questions")
    questions = re.findall(r"^(\d+)\. ", section, flags=re.MULTILINE)
    assert questions == [str(i) for i in range(1, 15)]


def test_charter_preserves_complete_model_and_runtime_scope(repo_root):
    text = (repo_root / "docs/research/project-charter.md").read_text().lower()
    model_targets = [
        "natural language", "instruction following", "coding", "mathematics",
        "research-oriented reasoning", "structured information synthesis",
        "productivity", "creativity", "planning", "tool selection", "tool calls",
        "tool-result interpretation", "verification", "failure recovery",
        "state interpretation", "memory-use decisions", "autonomous-control decisions",
    ]
    runtime_targets = [
        "objective management", "persistent state", "memory", "permissions",
        "tool registry", "sandboxed execution", "scheduling", "event handling",
        "resource limits", "process supervision", "checkpoint/resume", "action logging",
        "interruption", "rollback",
    ]
    assert not [term for term in model_targets + runtime_targets if term not in text]
    assert "defined system targets, not implemented functionality" in text


def test_all_explicit_non_goals_are_preserved(repo_root):
    text = (repo_root / "docs/research/project-charter.md").read_text().lower()
    required = [
        "native image encoding", "native audio encoding", "native speech synthesis",
        "native video understanding", "architectural recurrence", "adaptive neural halting",
        "hierarchical moe", "dynamic expert creation during ordinary inference",
        "unrestricted self-modification", "automatic promotion of self-generated checkpoints",
        "permanent specialist-model collections",
    ]
    assert not [term for term in required if term not in text]


def test_governance_preserves_exactly_forty_numbered_rules(repo_root):
    text = (repo_root / "docs/research/project-governance.md").read_text()
    numbers = re.findall(r"^## (\d+)\. ", text, flags=re.MULTILINE)
    assert numbers == [str(i) for i in range(1, 41)]
    assert "## 33. The official configuration is one configuration" in text
    assert re.search(r"reference,\s+not a cage", text)


def test_authoritative_roadmap_has_all_sixteen_phases(repo_root):
    text = (repo_root / "docs/research/project-roadmap.md").read_text()
    numbers = re.findall(r"^(\d+)\. ", text, flags=re.MULTILINE)
    assert numbers == [str(i) for i in range(16)]


def test_known_stale_phase_mappings_do_not_return(repo_root):
    tracked = [
        repo_root / "docs/research/project-charter.md",
        repo_root / "docs/research/project-governance.md",
        repo_root / "manifests/frozen-artifacts.yaml",
    ]
    text = "\n".join(path.read_text() for path in tracked)
    stale = ["Phase 3+ data", "Phase 4+", "Phase 5+", "Phase 3/4"]
    assert not [marker for marker in stale if marker in text]


def test_complete_precision_policy_is_versioned(repo_root):
    text = (repo_root / "docs/architecture/precision-policy.md").read_text().lower()
    required = [
        "fp16 mixed precision", "fp32 optimizer state", "fp32 master parameter state",
        "fp32 gradient accumulation", "fp32 rmsnorm reductions",
        "fp32 per-head qk-norm reductions", "fp32 router logits",
        "fp32 router softmax", "fp32 loss/logit accumulation",
        "dynamic gradient scaling", "global gradient clipping", "fp16 reference inference",
        "int8", "weight-only 4-bit",
    ]
    assert not [term for term in required if term not in text]
