# Juniper Auto — Research Charter

Status: living document, Phase 0 baseline
Architecture target: `ja150m-v0.1`

## Primary research question

How capable, efficient, autonomous, persistent, adaptable, reliable, and
meaningfully self-improving can one approximately 150-million-parameter open
sparse Mixture-of-Experts language model become while remaining practical on
consumer hardware and genuinely customizable?

## Secondary research questions

1. Whether approximately 150M total / approximately 79M active parameters can
   support useful competence across general language, coding, mathematics,
   research, productivity, creativity, planning, tools, and verification.
2. Whether the 150M MoE measurably improves over the approximately 79.19M
   dense control under equivalent data and optimization.
3. Whether routed experts learn useful specialization rather than
   token-frequency artifacts or redundant representations.
4. Whether the shared expert improves robustness while routed experts
   specialize.
5. Whether a 4,096-token working context plus external memory can outperform
   indiscriminately increasing context for persistent work.
6. Whether the same language model can function as assistant, planner, tool
   user, verifier, critic, researcher, programmer, and autonomous controller
   without permanent specialist language models.
7. Whether tool-use training can teach when a tool is required, which tool to
   use, how to invoke it, how to interpret its result, how to recover, and
   when not to use one.
8. Whether persistent state and memory can preserve long objectives without
   introducing stale, irrelevant, poisoned, or contradictory context.
9. Whether runtime recurrence -- reason, act, observe, verify, revise, and
   continue -- provides useful reasoning depth without architectural
   recurrence.
10. Whether increasingly long objectives can be completed with less human
    intervention, false completion, looping, and unrecovered failure.
11. Whether self-play and internally generated curricula produce measurable
    improvement rather than additional synthetic text alone.
12. Whether Juniper Auto can inspect routing, evaluations, tools, datasets,
    and runtime sufficiently to propose useful improvements.
13. Whether those proposals can be evaluated through isolated candidate
    experiments without granting the running model authority to silently
    modify or promote itself.
14. Whether the full system can remain realistically local, inspectable,
    forkable, retrainable, and customizable.

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
curated, or approved. These figures are planning targets for Phase 6 unified
pretraining-dataset work and Phases 10-11 post-training work, recorded here
so later phases are held to them rather than drifting.
The [[frozen-artifact-registry]] records dataset status as `not-yet-created`.

## v0.1 model scope

The following are future capability targets for `ja150m-v0.1`, not Phase 0
implementation claims. The model should eventually learn useful competence
in:

- natural language and instruction following;
- coding and mathematics;
- research-oriented reasoning and structured information synthesis;
- productivity and creativity;
- planning;
- tool selection, tool calls, and tool-result interpretation;
- verification and failure recovery;
- state interpretation and memory-use decisions; and
- autonomous-control decisions.

The corresponding frozen architecture targets are:

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

The eventual v0.1 runtime scope explicitly includes:

- objective management;
- persistent state and memory;
- permissions and a tool registry;
- sandboxed execution;
- scheduling and event handling;
- resource limits and process supervision;
- checkpoint/resume;
- action logging;
- interruption; and
- rollback.

These are defined system targets, not implemented functionality. The runtime
should ultimately provide local inference on consumer hardware and bounded
autonomous execution under owner priority. None of this runtime exists yet;
Phase 0 builds only configuration, logging, and experiment-tracking
foundations. Runtime implementation belongs to Phase 4.

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
- Automatic promotion of self-generated checkpoints
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
