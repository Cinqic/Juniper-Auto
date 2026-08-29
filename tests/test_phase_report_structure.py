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
    if "APPROVED" in text:
        assert "GPT-5.6 SOL MEDIUM" in text
        assert "phase-0-sol-independent-review.md" in text
        assert "phase-0-foundation" in text
    assert "COMPLETE" != text.strip()


def test_phase_0_report_approval_status_is_a_valid_value(repo_root):
    text = (repo_root / "docs" / "phases" / "phase-0-foundation.md").read_text()
    valid_markers = [
        "CANDIDATE - PENDING INDEPENDENT REVIEW",
        "APPROVED BY",
        "APPROVED WITH ACCEPTED LIMITATIONS BY",
        "REJECTED BY",
    ]
    assert any(marker in text for marker in valid_markers), (
        "phase-0-foundation.md must declare an explicit, valid approval status"
    )


def test_phase_0_report_distinguishes_candidate_handoff_and_approval(repo_root):
    text = (repo_root / "docs/phases/phase-0-foundation.md").read_text()
    assert "Sonnet substantive candidate" in text
    assert "Sonnet metadata/handoff HEAD" in text
    assert "Sol substantive reviewed repairs" in text
    assert "resolved by the annotated" in text


def test_phase_1_report_exists_and_has_required_sections(repo_root):
    path = repo_root / "docs" / "phases" / "phase-1-architecture.md"
    assert path.is_file(), "docs/phases/phase-1-architecture.md must exist"
    text = path.read_text()
    missing = [section for section in REQUIRED_SECTIONS if section not in text]
    assert not missing, f"phase-1-architecture.md missing sections: {missing}"


def test_phase_1_approval_has_independent_evidence_and_concrete_identity(repo_root):
    report = repo_root / "docs" / "phases" / "phase-1-sol-independent-review.md"
    assert report.is_file(), "independent Phase 1 review must be repository-contained"
    text = report.read_text()
    assert "PHASE 1 APPROVED" in text
    assert "GPT-5.6 Sol" in text
    assert "9555bbcb43d7b4f63762a5f11c2cea13e11fa7c8" in text
    assert "phase-1-architecture" in text

    records = "\n".join(
        (repo_root / path).read_text()
        for path in [
            "README.md",
            "docs/phases/phase-1-architecture.md",
            "docs/phases/phase-1-requirements-traceability.md",
            "docs/phases/phase-1-sonnet-self-review.md",
            "docs/time/phase-hours.csv",
        ]
    )
    assert "HEAD (this commit" not in records
    assert "final handoff message" not in records


def test_phase_2_report_exists_and_has_required_sections(repo_root):
    path = repo_root / "docs" / "phases" / "phase-2-moe.md"
    assert path.is_file(), "docs/phases/phase-2-moe.md must exist"
    text = path.read_text()
    missing = [section for section in REQUIRED_SECTIONS if section not in text]
    assert not missing, f"phase-2-moe.md missing sections: {missing}"


def test_phase_2_approval_has_independent_evidence_and_concrete_identity(repo_root):
    report = repo_root / "docs" / "phases" / "phase-2-sol-independent-review.md"
    assert report.is_file(), "independent Phase 2 review must be repository-contained"
    text = report.read_text()
    assert "**APPROVED.**" in text
    assert "GPT-5.6 Sol" in text
    assert "2d5a34f85c996bf0beededcb47629b567685b907" in text
    assert "phase-2-moe" in text

    phase_report = (repo_root / "docs" / "phases" / "phase-2-moe.md").read_text()
    assert "## Approval status\n\n`APPROVED`" in phase_report
    assert "resolved by the annotated `phase-2-moe` tag" in phase_report
