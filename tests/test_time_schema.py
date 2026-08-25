import csv

REQUIRED_COLUMNS = {
    "phase",
    "date",
    "start_time",
    "end_time",
    "engineering_hours",
    "self_review_hours",
    "independent_review_hours",
    "active_human_hours",
    "ai_assisted_engineering_hours",
    "gpu_hours",
    "cpu_data_processing_hours",
    "major_task",
    "commit_or_experiment_id",
    "outcome",
    "blocker_or_failure",
    "approximation_flag",
    "notes",
}


def test_phase_hours_csv_exists(repo_root):
    assert (repo_root / "docs" / "time" / "phase-hours.csv").is_file()


def test_phase_hours_csv_has_required_columns(repo_root):
    path = repo_root / "docs" / "time" / "phase-hours.csv"
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
    missing = set(REQUIRED_COLUMNS) - fieldnames
    assert not missing, f"missing columns: {missing}"


def test_phase_hours_csv_has_at_least_one_phase_0_row(repo_root):
    path = repo_root / "docs" / "time" / "phase-hours.csv"
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r["phase"] == "phase-0"]
    assert len(rows) >= 1


def test_phase_hours_csv_gpu_and_cpu_hours_are_zero_or_numeric_in_phase_0(repo_root):
    """Phase 0 does no training, so GPU hours should be 0 for every row --
    this is a sanity check against silently claiming GPU work happened."""
    path = repo_root / "docs" / "time" / "phase-hours.csv"
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r["phase"] == "phase-0"]
    for row in rows:
        assert float(row["gpu_hours"]) == 0.0, f"unexpected nonzero gpu_hours in Phase 0 row: {row}"
