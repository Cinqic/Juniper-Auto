# Phase N Report Template

Copy this file to `docs/phases/phase-N-<short-name>.md` and fill in every
section. Do not leave a section as prose-only when an executable check
exists for it -- link to the actual test, script output, or CI run instead
of asserting completion in words. See
[project-governance](../research/project-governance.md), rule 15
(executed-action truthfulness).

## Phase

Phase number and name.

## Objective

What this phase set out to accomplish, in 1-3 sentences.

## Starting commit

Commit hash this phase began from.

## Final commit

Commit hash this phase's candidate ends at. (Leave as
`CANDIDATE - PENDING INDEPENDENT REVIEW` status until an independent
reviewer approves this exact commit -- see the Approval status section.)

## Implementation summary

What was actually built, file by file or component by component. Link to
the files.

## Architecture / configuration IDs

Which `architecture_id`(s) this phase's work applies to or changes.

## Environment

Link to the environment specification used (see
`docs/architecture/environment-specification.md`), noting if this phase's
work required anything different.

## Artifacts

What this phase produced (docs, code, configs, data, checkpoints). For each,
state its status per `manifests/frozen-artifacts.yaml`
(`frozen`/`planned`/`not-yet-created`/`superseded`).

## Hashes

Link to the relevant artifact hash manifest entries
(`manifests/*-artifact-hashes.yaml`).

## Tests

What automated tests cover this phase's work, and their result. Link to the
actual test run output, not just the test file names.

## Evaluations

What evaluations were run (if any) and their results. `not-applicable` if
none were run this phase, stated explicitly rather than omitted.

## Ablations (where relevant)

What ablations were run (if any) and their results. `not-applicable` if
none.

## CI workflow / run

Workflow name, run ID, commit tested, conclusion.

## Recovery status

Was the clean-clone recovery exercise performed for this phase's changes?
Result.

## Engineering hours

From `docs/time/phase-hours.csv`, this phase's row(s).

## Independent review hours

From `docs/time/phase-hours.csv`. `0`/`pending` until an independent
reviewer actually reviews.

## GPU hours

From `docs/time/phase-hours.csv`.

## CPU / data-processing hours

From `docs/time/phase-hours.csv`.

## Project elapsed days

Calendar days since the project's initial commit.

## Known failures

Anything that did not work, stated plainly.

## Negative results

Any experiment in `experiments/registry.yaml` with
`status: negative-result` or `status: failed` relevant to this phase.

## Accepted limitations

What this phase knowingly leaves unresolved, and why that's acceptable for
now.

## Reproducibility procedure

Exact commands to reproduce this phase's validated state from a clean
clone. Link to `docs/recovery/`.

## Reviewer identity

Who performed the self-review (implementer) and who performed independent
review (if it has happened yet).

## Approval status

Exactly one of: `CANDIDATE - PENDING INDEPENDENT REVIEW`, `APPROVED`,
`APPROVED WITH ACCEPTED LIMITATIONS`, or `REJECTED / BLOCKED`. Identify the
reviewer and use an immutable approval tag to resolve the approved commit
when putting a commit's own hash or CI identity in that commit would require
impossible self-reference.
