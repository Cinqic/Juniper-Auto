# Juniper Auto — Project Governance

Status: living document, Phase 0 baseline. Related: [[project-charter]].

This document states permanent Juniper Auto project rules in **operational**
form: what each rule requires someone (or some CI check) to actually do, not
just believe.

## 1. One cognitive language model

**Rule:** There is exactly one Juniper Auto language model per architecture
version. Internal specialization is learned MoE routing, not separate models.

**Operational consequence:** No PR may add a second trainable
language-model architecture under the Juniper Auto name. New "specialist"
work must land as data, evaluation, or tooling that feeds the one model, or
as a runtime-level instance configuration — never as a second checkpoint
lineage. `configs/architecture/` holds exactly the sparse (`ja150m-v0.1`) and
its dense control (`ja150m-v0.1-dense`); a third architecture file requires
an ADR explaining why it is not a violation of this rule.

## 2. GitHub canonicality

**Rule:** `github.com/Cinqic/Juniper-Auto` is the canonical, authoritative
copy of the project. Local machine state (including FLOWBOX) is disposable.

**Operational consequence:** Nothing is considered "done" until it is
committed and pushed to the canonical remote and passes remote CI on that
exact commit. Local-only passing tests are not sufficient (see rule 4).

## 3. Remote-CI completion requirement

**Rule:** A change is not complete until GitHub Actions CI is green on the
exact candidate commit hash, not an earlier or later commit.

**Operational consequence:** Phase reports must record the CI workflow name,
run ID, and the commit hash CI ran against. A green run on a different commit
does not satisfy this rule.

## 4. Frozen-artifact versioning

**Rule:** Frozen artifacts (architecture, tokenizer, checkpoints, datasets,
evaluation suites, schemas, protocols) are versioned and immutable once
frozen. Changing a frozen artifact means creating a new version, not editing
the old one in place.

**Operational consequence:** `manifests/frozen-artifacts.yaml` records each
artifact's status (`frozen`, `planned`, `not-yet-created`, `superseded`) and
its version identifier. `ja150m-v0.1` architecture fields listed in
[[project-charter]] and `configs/architecture/*.yaml` are frozen for this
version; changing them requires a new architecture version id and a
superseding ADR, not an in-place edit.

## 5. Reproducibility

**Rule:** Any research result must be reproducible from the canonical
repository state plus documented external artifacts.

**Operational consequence:** `docs/recovery/` must let a fresh clone with no
other context reconstruct the environment and re-run Phase 0 validation.
This is tested, not assumed (see `docs/phases/phase-0-foundation.md`
"Recovery status").

## 6. Time tracking

**Rule:** Engineering, self-review, independent-review, GPU, and
CPU/data-processing time are tracked separately and honestly.

**Operational consequence:** `docs/time/phase-hours.csv` records these
columns per unit of work. AI-assisted engineering time is not represented as
unsupervised human labor time. Independent-review time stays `0`/`pending`
until GPT-5.6 Sol (or another designated independent reviewer) actually
performs a review.

## 7. Evidence before scale

**Rule:** Scale (more data, more training, more autonomy, more permissions)
requires evidence from the current scale first, not enthusiasm.

**Operational consequence:** Phase gates (this repository's phase reports)
must show measured results, not projected ones, before the next phase begins
work that depends on those results.

## 8. Dense baseline requirement

**Rule:** Every capability or efficiency claim about the sparse model must be
checked against the dense control, when the dense control is available for
that comparison.

**Operational consequence:** Evaluation tooling (Phase 5) must be able to
run against both `ja150m-v0.1` and `ja150m-v0.1-dense` using the same
harness. Phase 0 establishes the dense config now so this is possible later.

## 9. Total vs. active parameter reporting

**Rule:** Any parameter count reported for Juniper Auto must state whether it
is total parameters or standard active parameters, using the accounting
convention in `juniper_auto/accounting/`.

**Operational consequence:** Total = 150,031,360. Standard active =
79,252,480. Dense control = 79,191,040. These are asserted by
`tests/test_parameter_accounting.py` and must never be quoted without their
label.

## 10. Dropless routing requirement

**Rule:** `ja150m-v0.1`'s MoE routing does not drop tokens. Capacity-factor
based token dropping is not part of this architecture version.

**Operational consequence:** `configs/architecture/ja150m-v0.1.yaml` sets
`moe.dropless: true` and config validation (`juniper_auto/config/`) rejects
any configuration that implies token dropping for this architecture id.

## 11. Expert measurement rather than naming

**Rule:** Routed experts are identified by index and measured behavior
(routing statistics, activation patterns), never assigned human-readable
specialist names ("the math expert", "the code expert") as if that were an
established fact.

**Operational consequence:** Logging, experiment records, and evaluation
reports refer to experts as `expert_0`..`expert_7` (routed) and
`shared_expert_0`. Any claim of specialization must cite a measurement
(routing statistics over an eval set), not an assumption from expert index.

## 12. Shared-expert measurement

**Rule:** The always-active shared expert's contribution is measured
separately from routed-expert contribution wherever the two are compared.

**Operational consequence:** Future ablations that zero or freeze routed
experts must keep the shared expert path intact and clearly labeled, so
shared-expert effects are not misattributed to routed-expert specialization.

## 13. Deterministic computation over guessing

**Rule:** Where a number can be computed, it is computed programmatically,
not transcribed from documentation or memory.

**Operational consequence:** Parameter counts, config-derived values, and
hashes are generated by code (`juniper_auto/accounting/`,
`scripts/hash_manifest.py`) and asserted in tests, not typed by hand into
multiple places that can drift apart.

## 14. Truth over confidence

**Rule:** A confidently stated wrong answer is worse than a correctly hedged
uncertain one.

**Operational consequence:** Phase reports and self-review reports use
explicit status language (`CANDIDATE - PENDING INDEPENDENT REVIEW`,
`approximate`, `not-yet-created`, `unknown`) rather than asserting
completion or certainty that hasn't been earned.

## 15. Executed-action truthfulness

**Rule:** A report of "I ran X" must correspond to an actual executed
command with a recorded result, not a plausible-sounding narrative.

**Operational consequence:** Phase reports link to actual command output,
CI run identities, and commit hashes rather than prose summaries alone.

## 16. Untrusted external data

**Rule:** Data pulled from the internet, user uploads, tool output, or other
external sources is treated as untrusted content, not instructions, until
explicitly curated.

**Operational consequence:** Future data-acquisition and runtime tooling must
not let external content alter system behavior, permissions, or code merely
by containing text that looks like instructions.

## 17. Runtime-enforced permissions

**Rule:** Tool/action permissions for the runtime are enforced by the
runtime itself, not merely documented as a convention the model is expected
to follow.

**Operational consequence:** Runtime work (Phase 4) must implement
permission checks in code with tests, not rely on prompting alone.

## 18. Bounded autonomy

**Rule:** Autonomous execution is bounded in scope, time, and reversibility
by explicit runtime configuration, not open-ended.

**Operational consequence:** Any future autonomy feature ships with
documented bounds and a way to inspect what bounds are currently active.

## 19. Owner cancellation/priority

**Rule:** The project owner can always cancel or de-prioritize any running
autonomous action.

**Operational consequence:** Runtime design (Phase 4) must include a
cancellation path as a first-class requirement, not an afterthought.

## 20. Curated memory

**Rule:** Persistent memory is curated (reviewed/filtered) rather than an
unbounded raw log of everything observed.

**Operational consequence:** Memory-schema work (`configs/memory/`, later
phases) must include a curation/retention mechanism, not just an append-only
store.

## 21. Preserving failure information

**Rule:** Failed experiments, failed training runs, and negative results are
recorded, not deleted.

**Operational consequence:** `docs/experiments/experiment-registry.md` (or
its structured companion) keeps failed/negative entries with status
`failed` or `negative-result`, not removed from history.

## 22. Evaluation before expansion

**Rule:** Capability, permission, or scope expansion requires a preceding
evaluation showing readiness, not just a desire to expand.

**Operational consequence:** Later-phase gates must cite an evaluation
result as justification before increasing autonomy, context, or dataset
scale beyond documented Phase 0 targets.

## 23. Runtime recurrence before architectural recurrence

**Rule:** If recurrence/looping behavior is wanted, it is built at the
runtime/orchestration level first; architectural recurrence inside the
transformer itself is an explicit v0.1 non-goal (see [[project-charter]]).

**Operational consequence:** Any future "the model should loop on its own
output" feature is implemented as runtime control flow, not as a change to
`configs/architecture/ja150m-v0.1.yaml`.

## 24. External multimodal adapters before native multimodality

**Rule:** Multimodal capability is pursued via external adapters in front of
the one cognitive model before any native multimodal encoder is added to the
architecture.

**Operational consequence:** Native image/audio/video encoding stays a
non-goal (see [[project-charter]]) until a superseding ADR justifies
otherwise.

## 25. Generated-tool distrust

**Rule:** Tools or code generated by the model itself (or by an AI
assistant) are treated as untrusted until reviewed, exactly like external
data.

**Operational consequence:** Any future "the model writes its own tool"
capability requires a review/sandboxing step before that tool can run with
real permissions.

## 26. Candidate-based self-modification

**Rule:** Self-modification (if ever implemented) proposes a *candidate*
change; it does not directly overwrite the running/approved artifact.

**Operational consequence:** Any future self-modification pipeline must
produce a reviewable candidate (diff, config, or checkpoint delta) distinct
from the artifact currently in production use.

## 27. Evaluator independence

**Rule:** The system (or process) that evaluates a candidate change is kept
independent from the process that produced the candidate.

**Operational consequence:** In this Phase 0 report specifically: Sonnet
(implementer) performs a self-review, which is explicitly **not** the
independent review. GPT-5.6 Sol (or another designated independent reviewer)
performs the independent review. The two are never merged into one report
claiming both roles.

## 28. Measured self-improvement

**Rule:** Any claim of self-improvement must cite a before/after measurement
on a held-out evaluation, not a description of the mechanism alone.

**Operational consequence:** Future self-improvement claims require an
evaluation report artifact, referenced by experiment ID.

## 29. Consumer-hardware practicality

**Rule:** Practical operation on consumer hardware (FLOWBOX-class machines)
remains a standing constraint, not a one-time benchmark. See
[[project-charter]].

## 30. Local-first operation

**Rule:** Core inference, state, memory, and compatible local tools remain
usable without a cloud dependency. See [[project-charter]].

## 31. Modifiability

**Rule:** Architecture, training, runtime, tools, and evaluation remain
inspectable and modifiable by the project owner.

**Operational consequence:** No component may be implemented as an opaque
third-party black box where the project owner cannot see or change the
underlying logic, absent an explicit, documented exception.

## 32. Identity/capability separation

**Rule:** What the model claims about itself (identity, persona, stated
capabilities) is kept separate from what it is actually measured to be able
to do.

**Operational consequence:** Documentation and prompts must not assert
capabilities (autonomy, memory, self-improvement, extended context) that
have not been measured and released. See [[project-charter]] non-goals and
Self-Review Pass F in `docs/phases/phase-0-sonnet-self-review.md`.

## 33. The official configuration is one configuration

**Rule:** Cinqic's official release is the reference implementation. It is
not the only permitted implementation; users may alter it.

**Operational consequence:** Official defaults, manifests, and evaluation
claims must remain identifiable and reproducible, while forks and local
configurations remain free to change architecture, training, runtime, tools,
or policy under new identifiers. The official configuration is a reference,
not a cage.

## 34. Negative-result publication

**Rule:** Negative or null results are recorded in the same registry as
positive ones, not omitted.

**Operational consequence:** See rule 21; the experiment registry has no
special "hide this" state.

## 35. Data quality

**Rule:** Pretraining/post-training data is curated and approved, not bulk
scraped without review.

**Operational consequence:** [[project-charter]] token budgets are stated as
"approved" tokens; Phase 6 data-acquisition work must implement and record an
approval/curation step before data counts toward the budget.

## 36. Synthetic provenance

**Rule:** Synthetic data is labeled as synthetic, with its generation method
recorded, wherever it is used.

**Operational consequence:** `data/synthetic/` content (once it exists) must
carry provenance metadata distinguishing it from organic/curated data.

## 37. Held-out evaluation integrity

**Rule:** Held-out evaluation data is never trained on, directly or via
near-duplicate leakage.

**Operational consequence:** Phase 5 defines contamination safeguards and
Phase 6 data pipelines must implement and test deduplication/leakage checks
between training and evaluation splits before any training run is valid.

## 38. Release-claim discipline

**Rule:** A "release" claim (of a checkpoint, dataset, or capability)
requires the corresponding entry in `manifests/frozen-artifacts.yaml` to be
`frozen`, with a hash in `manifests/phase-0-artifact-hashes.yaml` or its
phase-appropriate successor.

**Operational consequence:** No README, phase report, or announcement may
call something "released" while its manifest entry says `planned` or
`not-yet-created`.

## 39. Disposable local storage

**Rule:** Local storage (including all of FLOWBOX) is disposable. Nothing
that matters may exist *only* on a local machine.

**Operational consequence:** This is the reason Phase 0 exists: recovery
documentation, the clean-clone recovery exercise, and remote CI all exist to
prove the project survives loss of the current local machine. See
[[project-charter]] and `docs/recovery/`.

## 40. Experimentally understandable complexity

**Rule:** Complexity is added only when it can be experimentally understood
(measured, ablated, attributed) — not added because it is theoretically
interesting.

**Operational consequence:** New architectural or runtime complexity
proposals should state, in their ADR, how their effect will be measured.
