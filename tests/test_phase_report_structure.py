REQUIRED_SECTIONS = [
    "## Phase",
    "## Objective",
    "## Starting commit",
    "## Final commit",
    "## Implementation summary",
    "## Architecture / configuration IDs",
    "## Environment",
    "## Artifacts",
    "## Hashes",
    "## Tests",
    "## Evaluations",
    "## CI workflow / run",
    "## Recovery status",
    "## Engineering hours",
    "## Independent review hours",
    "## GPU hours",
    "## CPU / data-processing hours",
    "## Known failures",
    "## Negative results",
    "## Accepted limitations",
    "## Reproducibility procedure",
    "## Reviewer identity",
    "## Approval status",
]


def test_phase_report_template_has_required_sections(repo_root):
    text = (repo_root / "docs" / "phases" / "phase-report-template.md").read_text()
    missing = [s for s in REQUIRED_SECTIONS if s not in text]
    assert not missing, f"phase-report-template.md missing sections: {missing}"


def test_phase_0_report_exists_and_has_required_sections(repo_root):
    path = repo_root / "docs" / "phases" / "phase-0-foundation.md"
    assert path.is_file(), "docs/phases/phase-0-foundation.md must exist"
    text = path.read_text()
    missing = [s for s in REQUIRED_SECTIONS if s not in text]
    assert not missing, f"phase-0-foundation.md missing sections: {missing}"


def test_phase_0_report_does_not_self_approve(repo_root):
    """Sonnet is the implementer, not the independent reviewer -- the report
    must never claim final approval on its own authority."""
    text = (repo_root / "docs" / "phases" / "phase-0-foundation.md").read_text()
    assert "APPROVED BY" not in text or "PENDING INDEPENDENT REVIEW" in text
    assert "COMPLETE" != text.strip()


def test_phase_0_report_approval_status_is_a_valid_value(repo_root):
    text = (repo_root / "docs" / "phases" / "phase-0-foundation.md").read_text()
    valid_markers = [
        "CANDIDATE - PENDING INDEPENDENT REVIEW",
        "APPROVED BY",
        "REJECTED BY",
    ]
    assert any(marker in text for marker in valid_markers), (
        "phase-0-foundation.md must declare an explicit, valid approval status"
    )
