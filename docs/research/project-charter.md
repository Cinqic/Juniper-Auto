# Juniper Auto — Research Charter

Status: living document, Phase 0 baseline
Architecture target: `ja150m-v0.1`

## Primary research question

How capable, efficient, autonomous, persistent, adaptable, reliable, and
meaningfully self-improving can one approximately 150-million-parameter open
sparse Mixture-of-Experts language model become while remaining practical on
consumer hardware and genuinely customizable?

## Secondary research questions

1. Does learned MoE routing at this scale produce measurable internal
   specialization (by domain, task type, or modality of reasoning), and can
   that specialization be observed, measured, and reported rather than
   assumed?
2. At matched (or near-matched) active-parameter budgets, does the sparse
   `ja150m-v0.1` architecture outperform the dense control model
   (`ja150m-v0.1-dense`, 79,191,040 parameters) on held-out evaluation, and by
   how much, and on which capability axes?
3. What is the actual measured relationship between reported active-parameter
   count and realized FLOPs/latency/throughput on consumer hardware (FLOWBOX
   and comparable machines) — i.e., where does the parameter-accounting
   convention diverge from delivered compute?
4. How much of a small model's practical capability, autonomy, and perceived
   reliability can be delivered through runtime scaffolding (memory, tools,
   bounded autonomy, evaluation-gated self-modification) rather than raw
   parameter count?
5. Can a single ~150M-parameter checkpoint, run as multiple temporary
   instances with different runtime configuration, cover the practical range
   of tasks that would otherwise motivate separate specialist models --
   without those instances becoming permanent separate models?
6. What pretraining and post-training data composition, at the stated token
   budgets (approximately 4-5B unique approved pretraining tokens, ~6B
   effective exposures, ~100-300M curated post-training tokens), produces the
   best measured capability-per-token on this architecture?
7. What does genuine reproducibility cost and require for a project of this
   size — i.e., can the project survive the loss of the current local
   machine, and what does Phase 0 have to build to make that true?
8. Where do consumer-hardware constraints (6 GB VRAM, 16 GB system RAM)
   force real architectural or training-methodology tradeoffs, and how are
   those tradeoffs documented rather than silently absorbed?

## One-model philosophy

Juniper Auto has **one cognitive language model**. Internal specialization
occurs through learned Mixture-of-Experts routing inside that one model, not
through separate models for separate domains.

- There is no separate math model, code model, research model, productivity
  model, or creativity model as permanent language-model components of the
  architecture.
- Former or parallel specialist-model projects (internal or external) may
  contribute research findings, data-curation methodology, tooling,
  evaluation suites, lessons learned, and failure analyses to Juniper Auto.
  They are inputs to the research process, not architecture components.
  [[project-governance]]
- Multiple temporary *instances* of the same Juniper Auto checkpoint (e.g.
  different runtime configuration, different tool access, different system
  prompts, different in-context memory) may be run and studied concurrently
  without this becoming a specialist-model architecture. The underlying
  weights remain one model.

## Consumer-hardware constraint

The initial engineering baseline machine is **FLOWBOX**:

| Component | Spec |
|---|---|
| CPU | AMD Ryzen 7 5700G |
| GPU | NVIDIA RTX 2060, 6 GB VRAM |
| RAM | 16 GB (system-reported ~14 GiB usable) |
| Primary storage | 256 GB NVMe |
| Secondary storage | 500 GB HDD |

FLOWBOX is the baseline that keeps the project honest about consumer-hardware
practicality — it is not a claim that every future training run must occur
exclusively on FLOWBOX. Larger or cloud compute may be used later for
specific runs. What must not happen is Juniper Auto quietly turning into a
system that only works with datacenter-class inference hardware; core
inference on consumer hardware is a standing design constraint, not a
one-time benchmark.

## Local-first objective

Core inference, conversation state, memory, file access, and compatible
local tools should ultimately remain useful without a cloud dependency.
Cloud or remote services may be used as optional accelerants (e.g. faster
training, larger evaluation runs) but are not permitted to become a hard
requirement for using the model at all.

## Open and customizable objective

Architecture, training methodology, runtime, tools, evaluations, and
customization mechanisms are intended to remain inspectable and modifiable
by the project owner.

**License status:** an MIT `LICENSE` file already exists in this repository
(added prior to Phase 0 engineering). Phase 0 preserves it as-is. Phase 0
does **not** make a new legal licensing decision, does not add an additional
model-weights license, and does not assert that full open-source licensing
of the eventual trained artifacts (weights, tokenizer, datasets) has been
decided. That remains an open governance item — see
[[project-governance]] and [[frozen-artifact-registry]].

## Sparse-vs-dense requirement

A dense control model is a mandatory, first-class research requirement, not
an optional ablation. `ja150m-v0.1-dense` (79,191,040 parameters, all 20
layers using the 1536-dimensional dense SwiGLU FFN) exists specifically to
answer: does the sparse MoE architecture actually earn its complexity at a
comparable active-parameter budget?

**Active parameter count is a parameter-accounting convention, not a
measurement of FLOPs.** Standard active accounting for the sparse model
(79,252,480) counts embeddings, all attention parameters, the 5 dense-anchor
FFNs, 2 routed experts + 1 shared expert per MoE layer, router parameters,
QK norms, block norms, and the final norm. It does not itself account for
routing overhead, kernel efficiency, memory bandwidth, or dropless-routing
load imbalance. Any claim comparing sparse and dense models on efficiency
must state whether it is a parameter-accounting comparison or a measured
FLOPs/latency/throughput comparison, and must not conflate the two. See
[[parameter-accounting]].

## Token budgets (research targets, not completed artifacts)

- **Pretraining:** approximately 4-5B unique approved tokens.
- **Effective training exposure:** approximately 6B effective token
  exposures (implying some repetition across the unique-token pool).
- **Post-training:** approximately 100-300M curated tokens.

As of Phase 0, no pretraining or post-training corpus has been acquired,
curated, or approved. These figures are planning targets for Phase 3+ data
work, recorded here so later phases are held to them rather than drifting.
The [[frozen-artifact-registry]] records dataset status as `not-yet-created`.

## v0.1 model scope

`ja150m-v0.1` targets:

- A single decoder-only causal sparse MoE transformer, 150,031,360 total
  parameters, 79,252,480 standard active parameters (see
  [[parameter-accounting]]).
- 4,096-token training/inference context, with 16,384 as a documented future
  target that is explicitly **not** currently advertised as supported.
- FP16 mixed-precision training where training is eventually implemented.
- A dense control model at matched depth/width conventions
  (`ja150m-v0.1-dense`, 79,191,040 parameters) for comparison.
- A 36,864-token vocabulary target (tokenizer itself is out of scope for
  Phase 0; see [[project-governance]] and Phase 3 scope).

## v0.1 runtime scope

The eventual v0.1 runtime is expected to support: local inference on
consumer hardware, structured tool use under an explicit permission model,
curated/bounded memory, and bounded autonomous execution under owner
priority and cancellation. None of this runtime exists yet; Phase 0 builds
only the configuration, logging, and experiment-tracking foundations that
later runtime work will build on.

## Explicit v0.1 non-goals

The following are explicitly out of scope for `ja150m-v0.1` and must not be
silently absorbed into the architecture or runtime during any phase without
a superseding ADR:

- Native image encoding
- Native audio encoding
- Native speech synthesis
- Native video understanding
- Architectural recurrence
- Adaptive neural halting
- Hierarchical MoE
- Dynamic expert creation during ordinary inference
- Unrestricted self-modification
- Automatic self-promotion (a candidate change promoting itself without
  evaluator/owner approval)
- Permanent specialist-model collections (see One-model philosophy, above)

Where external multimodal capability is wanted, the project's stated
approach is external adapters in front of the one cognitive model, not native
multimodal encoders inside `ja150m-v0.1` itself. See
[[project-governance]].

## Relationship to governance and architecture documents

This charter states *why* the project exists and what it is trying to learn.
Operational rules that follow from the charter are recorded in
[[project-governance]]. The frozen numeric architecture is recorded in
`configs/architecture/ja150m-v0.1.yaml` and
`configs/architecture/ja150m-v0.1-dense.yaml`, independently verified by
`juniper_auto/accounting/` and covered by `tests/test_parameter_accounting.py`.
