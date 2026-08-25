# Phase 0 Independent Review: GPT-5.6 Sol Medium

## Identity

- Reviewer and authorized fixer: GPT-5.6 Sol Medium
- Primary engineer and self-reviewer: Claude Sonnet 5 High
- Project owner and human overseer: Blessom
- Phase: Phase 0 — Research Charter, Governance, Recovery, and Project Foundation

## Review starting state

- Repository: `https://github.com/Cinqic/Juniper-Auto`
- Branch: `main`
- Sonnet substantive candidate: `88b4ed452dd477849b4041137130f52cc814fb5b`
- Sonnet metadata/handoff HEAD and Sol review start:
  `ca49784e5da913c639d387cc37d672b947c91718`
- Original handoff CI: `Phase 0 Validation` run `32876335758`, success on
  `ca49784e5da913c639d387cc37d672b947c91718`
- Starting remote state independently confirmed: `origin/main` matched the
  local HEAD and no Phase 0 approval tag existed.

The review-start time was recorded in `docs/time/phase-hours.csv` before
substantive repository review.

## Specification reconciliation

Sonnet accurately documented that the complete end-to-end specification was
unavailable in its session. Sol received the authoritative specification and
performed a semantic requirements diff, recorded in
`phase-0-requirements-traceability.md`.

- Secondary research questions: restored from 8 rewritten questions to the
  authoritative 14.
- Model scope: restored all natural-language, instruction, coding, math,
  research, synthesis, productivity, creativity, planning, tool,
  verification, recovery, state, memory, and autonomous-control targets.
- Runtime scope: restored all 14 system targets and explicitly distinguished
  defined scope from implemented functionality.
- Governance: restored the omitted official-configuration principle and
  renumbered the complete authoritative set to 40 rules.
- Roadmap: versioned the authoritative Phase 0-15 map and corrected dataset,
  evaluation, runtime, and contamination references.
- Architecture: made implicit activation, shared-expert, combination,
  normalization, rotary-fraction, and expert-bias semantics explicit.
- Router jitter: replaced hard enablement with an experiment-only permission,
  null magnitude, and disabled evaluation/inference behavior; ADR-0007 records
  the interpretation.
- Precision: added a frozen versioned policy covering the complete training
  and inference baseline without claiming Phase 0 implementation.

## Independent verification

### Parameter accounting

Independent arithmetic, performed without importing the repository accounting
module, produced:

| Component | Sparse count |
|---|---:|
| Embeddings | 18,874,368 |
| Attention | 13,107,200 |
| Five dense FFNs | 11,796,480 |
| Routed experts | 94,371,840 |
| Shared experts | 11,796,480 |
| Routers | 61,440 |
| Q/K norms | 2,560 |
| Block norms | 20,480 |
| Final norm | 512 |
| **Sparse total** | **150,031,360** |
| **Sparse standard active** | **79,252,480** |
| **Dense control total/active** | **79,191,040** |

Standard active parameters remain explicitly documented as an accounting
convention, not actual FLOPs. Controlled configuration mutations verified
that top-k and expert-bias changes alter derived counts.

### Configuration, tests, and validator

- Strict schema and cross-field validation exercised with missing, duplicate,
  overlapping, out-of-range, dimensionally invalid, malformed-ID, invalid
  top-k/expert, token-dropping, wrong-vocabulary, and shared-expert mutations.
- Frozen validation now checks all sparse semantics and every shared dense
  semantic policy, not only count-affecting core dimensions.
- Local canonical validator: all 11 gates passed.
- Full pytest result: 97 passed, 0 failed, 0 skipped/xfail.
- Adversarial gate tests verify missing locked dependencies, incomplete
  manifests, and stale hashes fail closed.
- SHA-256: all 11 protected artifacts matched; the manifest does not hash
  itself.

### Dependencies, integrity, and recovery

- Python requirement: 3.12.x; local evidence used 3.12.3.
- Lock: 46 exact hash-pinned distributions cover all 6 direct/dev
  declarations; `pip check` passes inside validation.
- Integrity: 63 tracked files scanned at the substantive reviewed state; no
  prohibited artifacts, files over 1 MB, credential-pattern hits, or
  unjustified host paths.
- Independent recovery: a new remote clone at substantive reviewed commit
  `b3deb56cd0056d21f73cb64fd1e2c5afb0a532a9`, a fresh venv, lock-only
  dependency installation, editable package installation without dependency
  re-resolution, and the canonical validator all succeeded. Result: 11/11
  gates and 97/97 tests passed.
- Clean Linux CI on the same substantive commit: `Phase 0 Validation` run
  `32879054618`, conclusion `success`.

The final approval commit is resolved by the annotated
`phase-0-foundation` tag. Because a commit cannot contain its own hash or CI
run identity, the exact tag-target CI and final recovery identity are recorded
in the tag annotation and the external completion report rather than through
an impossible self-referential metadata loop.

## Defects and repairs

| Defect | Severity | Root cause | Repair and regression coverage |
|---|---|---|---|
| 8 rewritten questions replaced the required 14 | blocking | Full specification unavailable to Sonnet | Restored exact program; ordered-count and semantic tests |
| Model/runtime scope compressed and incomplete | major | Shortened source interpretation | Restored explicit targets; structural tests |
| Governance contained 39 rules | blocking | Omitted official-configuration principle | Restored rule 33; exact 1-40 test |
| Stale Phase 3/4/5 mappings | major | Full roadmap unavailable | Versioned roadmap, corrected references, stale-marker test |
| Training router jitter frozen `true` | major | Permission conflated with experiment enablement | Experiment-only/null policy; ADR-0007; frozen tests |
| Precision policy incomplete | major | Only three fields preserved in architecture YAML | Frozen complete precision-policy artifact and test |
| Frozen checker covered only a subset of semantics | major | Parameter totals used as a proxy for semantic fidelity | Full sparse/shared-dense constant checks and mutation tests |
| Expert accounting hard-coded bias-free behavior | moderate | Accounting assumption not derived from config | Derive expert bias from config; mutation test |
| Validator under-checked lock, manifest, time rows, secrets, and large files | moderate | Important checks split between prose/tests/CI | Added fail-closed canonical gates and adversarial tests |
| Logging lacked experiment context and probe run/config IDs | minor | Initial event context was incomplete | Added experiment-capable context and end-to-end assertions |
| Phase report called a pre-review commit “Final” and retained an obsolete limitation | moderate/documentation | Candidate bookkeeping predated independent review | Canonical report now distinguishes candidate, handoff, review, and tag identity |
| CI actions targeted deprecated Node.js 20 | minor/maintenance | Stale action majors | Updated official actions to current v7 majors before approval |

## Time

The final approximate review/repair split, GPU time, and CPU/data-processing
time are in `docs/time/phase-hours.csv`. Blessom's active human time remains
`PENDING`; it was not observable and was not fabricated.

## Remaining limitations

- Recovery evidence is a fresh remote clone and fresh venv on FLOWBOX plus a
  separate clean GitHub-hosted Linux runner, not a literal physical OS
  reinstall.
- CUDA bitwise determinism is not claimed; the Phase 0 deterministic claim is
  limited to the exercised CPU probe and environment.
- The eventual model-weight/dataset license remains an owner-level future
  decision; the existing repository code license is MIT.
- The hash-pinned PyTorch resolution includes large CUDA runtime packages even
  for CPU validation. This is reproducible but inefficient and may be revisited
  through a future dependency ADR.

These limitations do not invalidate the recoverable Phase 0 foundation.

## Final decision

`APPROVED WITH ACCEPTED LIMITATIONS`

This decision becomes canonical only when the annotated
`phase-0-foundation` tag points to the exact independently reviewed commit
whose remote CI and final recovery pass. No approval is inferred from this
document alone if that tag is absent.
