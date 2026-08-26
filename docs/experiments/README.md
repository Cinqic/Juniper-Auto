# Experiment Registry

The canonical, version-controlled experiment registry is
`experiments/registry.yaml`. Every entry uses the schema below. This file
documents the schema and the conventions for filling it in; it is not
itself the data.

## Schema (per entry)

| Field | Meaning |
|---|---|
| `experiment_id` | Stable id, `exp-NNNN`, assigned in order, never reused. |
| `date` | `YYYY-MM-DD`. |
| `phase` | Which project phase this experiment belongs to. |
| `hypothesis` | What question this experiment is trying to answer. |
| `git_commit` | Commit the experiment was run at. |
| `architecture_id` | e.g. `ja150m-v0.1`, or `not-applicable`. |
| `tokenizer_id` | e.g. `not-yet-created` for Phase 0. |
| `dataset_id` | e.g. `not-yet-created` for Phase 0. |
| `sampler_id` | e.g. `not-applicable`. |
| `starting_checkpoint` | e.g. `not-applicable` (no checkpoints exist yet). |
| `configuration` | Path to the config file(s) used. |
| `seed` | Explicit integer seed used, or `not-applicable`. |
| `environment` | Short environment id (see `juniper_auto.util.environment`). |
| `token_budget` | Token budget for this run, or `not-applicable`. |
| `status` | `planned`, `running`, `completed`, `failed`, `negative-result`. |
| `result` | What was measured. |
| `conclusion` | What was concluded from the result. |
| `artifact_locations` | Where outputs live, or `not-applicable`. |

Canonical Phase 1 JSON result artifacts additionally record a stable result
identity, exact Git `HEAD`, clean/dirty worktree state, architecture-config
SHA-256, full command/run configuration, seed, and environment. The runner
refuses a dirty tree or an existing output by default. `--allow-dirty`
creates an explicitly non-canonical diagnostic artifact; `--overwrite` is
required to replace an existing path and must never be used to erase
historical evidence.

## Rules

- **No fabricated identities.** If a tokenizer, dataset, or checkpoint does
  not exist yet, the field says `not-yet-created`, never a plausible-looking
  placeholder name.
- **Negative and failed results stay in the registry.** `status:
  negative-result` and `status: failed` entries are never deleted (see
  [project-governance](../research/project-governance.md), rule 21).
- **Phase 0 entries are foundation-validation entries** -- they validate
  that the configuration/accounting/probe pipeline works, not that any
  model capability has been measured.
