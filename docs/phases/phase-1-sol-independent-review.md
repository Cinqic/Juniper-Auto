# Phase 1 GPT-5.6 Sol Independent Review

## Verdict

**PHASE 1 APPROVED**, with the accepted limitations listed below. Approval
becomes canonical through the annotated `phase-1-architecture` tag after
the approval-metadata commit itself passes both required remote workflows.

## Repository identities

- Review start / Sonnet final metadata candidate:
  `6a7410130e4b5ae6039a794f66f9360b675fd945` on `main`.
- Approved Phase 0 baseline: annotated tag `phase-0-foundation`, target
  `f6f8b5397b41d31081f55cdd60cb6363ec052be4`.
- Sol repair baseline used for canonical reruns:
  `e25b5478d1c72503286dc4d831f752f028e1b56e`.
- Substantive reviewed candidate:
  `9555bbcb43d7b4f63762a5f11c2cea13e11fa7c8`.
- Final approval metadata commit: this report's commit; resolved without
  self-reference by the annotated `phase-1-architecture` tag.

The remote baseline had not moved from the owner-provided `6a741013...`.
The frozen sparse/dense YAML files have no diff from `phase-0-foundation`.

## Review date and authority

- Date: 2026-08-26
- Reviewer and sole Phase 1 approval authority: GPT-5.6 Sol
- Implementer record reviewed: Claude Sonnet 5
- Scope: Phase 1 only; no Phase 2 work was performed.

## Independent methodology

The review began from a fetched, clean `main`, compared Phase 1 to the
Phase 0 approval tag, ran the existing validators, then read the model,
loss, checkpoint, state, profiling, experiment, test, CI, manifest, and
reporting code directly. Existing green tests were treated as claims to
challenge. Independent/manual references, controlled expert stubs,
fault-injection tests, invalid-shape cases, live CUDA FP16 execution,
checkpoint destruction/reconstruction, clean-tree provenance tests, and a
fresh clone were used where shared helpers or presence-only assertions
were insufficient.

## Architecture and parameter findings

Both official architectures implement the frozen contract. Independent
config equations and actual `Parameter` traversal agree:

| Category | Sparse total | Sparse active convention | Dense total |
|---|---:|---:|---:|
| Embedding / tied head | 18,874,368 | 18,874,368 | 18,874,368 |
| Attention projections | 13,107,200 | 13,107,200 | 13,107,200 |
| Dense FFNs | 11,796,480 | 11,796,480 | 47,185,920 |
| Routed experts | 94,371,840 | 23,592,960 | 0 |
| Shared experts | 11,796,480 | 11,796,480 | 0 |
| Routers | 61,440 | 61,440 | 0 |
| QK norms | 2,560 | 2,560 | 2,560 |
| Block norms | 20,480 | 20,480 | 20,480 |
| Final norm | 512 | 512 | 512 |
| **Total** | **150,031,360** | **79,252,480** | **79,191,040** |

The active count is explicitly a parameter-accounting convention, not a
FLOPs claim. The LM head is the same `Parameter` object as the embedding
and is counted once. Layer placement is dense at 1/5/10/15/20 for sparse,
MoE elsewhere, and all dense for the control.

## Mathematical and numerical findings

- GQA is 8 query heads / 2 KV heads with contiguous 4:1 KV sharing and
  exact scale 0.125; a manual masked-attention reference agrees.
- Q and K are RMS-normalized independently over each 64-dimensional head
  in FP32 before full-dimensional RoPE (`theta=100000`).
- Causal and key-padding masks prevent future/padding-key influence;
  scattered masks and fully padded rows remain finite.
- SwiGLU is exactly `down(SiLU(gate(x)) * up(x))`.
- MoE is dropless top-2 with eight routed experts plus one unconditional,
  ungated shared expert; selected probabilities renormalize to one; exact
  stub outputs prove there is no hidden averaging and token order is
  reconstructed correctly.
- Router logits/softmax remain FP32 under real RTX 2060 FP16 autocast;
  selected/router/shared gradients are finite.
- Load-balance and Z-loss formulas agree with hand-computed tensors and
  exclude padding. Coefficients are applied once after layer averaging.
- Causal CE shifts position `i` to target `i+1`; an adversarial example
  fails under the unshifted equation.
- Initialization reaches every Q/K/V, gate/up, output/down, router, and
  embedding weight exactly once with the intended standard deviation;
  norm scales start at one and tied weights are not initialized twice.

## Defects discovered and repaired

| Finding | Severity | Evidence | Repair | Regression proof |
|---|---|---|---|---|
| `sequence_curriculum_state` was saved but omitted from restore output | High / approval-blocking | Direct reproduction returned only step/token/sampler keys | Return and restore curriculum state; preserve an explicitly empty dict | Checkpoint unit test and exp-0012 exact resume |
| Experiment results could attribute dirty post-commit code to clean `HEAD` | High / evidence integrity | Dirty-tree probe still recorded `6a741013...` with no dirty marker | Clean-tree refusal, diagnostic override marking, config hashes, result ID, command/config/seed, overwrite refusal | Provenance tests; exp-0009..0014 all `canonical_result=true` at clean `e25b547...` |
| Explicit `build_model(seed=...)` consumed ambient RNG during overwritten constructor initialization | Moderate / reproducibility | 2,485 RNG-state bytes changed | Protect constructor work with CPU RNG fork before local-generator frozen init | Ambient RNG equality regression test |
| Supported 4,096-token context was not enforced | Moderate / capability truthfulness | Forward accepted lengths beyond the frozen supported boundary | Store/enforce `context_length`; explicit error says 16K target is unvalidated | Boundary test accepts 4,096 policy equivalent and rejects +1 |
| All-ignored/no-next-target causal batches could return NaN | Moderate / numerical | Existing test accepted NaN or zero | Fail clearly when no non-ignored shifted target exists | All-ignored and one-token labeled tests require `ValueError` |
| Invalid auxiliary/mask shapes could broadcast or fail cryptically | Moderate / correctness | Model/attention/MoE lacked exact shape validation | Exact shape/device validation at public boundaries | Adversarial attention/model/MoE shape tests |
| Sampler restore under-validated pool/batch/cursor/order identity | Moderate / resume integrity | Only seed/vocab/length were compared | Validate all construction fields, cursor range, and full permutation | Mismatch/corruption tests plus next-batch identity in exp-0012 |
| Non-null scheduler and GradScaler restoration were unproven | Moderate / test gap | Existing payload tests used both as `None` | Add real state round trips and live CUDA FP16 GradScaler step/restore | CPU scheduler/scaler and RTX 2060 CUDA tests |
| Profile labels/method omitted important training-path operations | Moderate / measurement integrity | Full-sequence work called generic throughput; `ru_maxrss` called RSS; no unscale/clipping | Label prefill and lifetime peak RSS; time synchronized unscale/clipping; check numerical finiteness | Profiling tests and exp-0013/0014 |
| Several config semantics were silently assumed | Moderate | Shared-always-active, jitter, norm reduction/bias, and rotary fraction were not enforced | Fail loudly or validate cross-field consistency | Config-assumption regression tests |
| Phase 1 records depended on a private final handoff and symbolic `HEAD` | Moderate / governance | Report, traceability, self-review, and time row contained those references | Preserve historical wording with concrete identities and add repository-contained Sol record | Documentation tests and final recovery |

## Checkpoint and resume findings

The v1 payload contains model, optimizer, scheduler, GradScaler, Python/
NumPy/CPU/CUDA RNG, sampler, counters, curriculum placeholder, full
architecture/training config, Git commit, and dataset/tokenizer identity.
Meaningful scheduler and scaler state restores. Exp-0012 destroys and
reconstructs the official sparse harness and obtains exact tail losses,
parameters (`max abs diff=0`), optimizer tensors/scalars, global step,
token count, next batch, and curriculum state. Negative controls without
checkpoint/sampler restore diverge.

## Experiment provenance and hardware findings

Sonnet's exp-0003..0008 remain historical. Their numeric results are not
deleted, but their `241295a...` attribution cannot prove the later dirty
self-review reruns. Approval relies on the clean, hash-protected Sol
reruns exp-0009..0014 at `e25b547...`.

| Experiment | Result |
|---|---|
| exp-0009 | Exact parameter gates and config hashes pass |
| exp-0010 dense overfit | LM 10.603769 → 0.001306; 100%; 1.672 GB peak; 29.73s |
| exp-0011 sparse overfit | LM 10.614758 → 0.000287; 100%; 3.087 GB peak; 120.93s |
| exp-0012 resume | Bit-exact across all inspectable state |
| exp-0013 dense profile | FP16 prefill 28,917 tok/s; checkpointed training 7,510 tok/s at 2.127 GB; 4,096 succeeds |
| exp-0014 sparse profile | FP16 prefill 5,167 tok/s; checkpointed training 1,782 tok/s at 3.377 GB; 4,096 succeeds |

FLOWBOX was directly identified as Ryzen 7 5700G, RTX 2060 6 GB, 16 GB
nominal RAM, 256 GB NVMe plus 500 GB disk, Linux. Profiles use untimed
warmup and CUDA synchronization. Prefill is not decode. Host metrics are
lifetime process peak RSS, not instantaneous stage-local memory.

## Test-suite and fault-injection findings

Final substantive suite: **289 passed** locally. CUDA-capable local runs
emit one expected PyTorch warning that memory-efficient attention backward
is nondeterministic. CI is CPU-only and skips CUDA-only tests. Existing
and new mutation checks detect future leakage, top-1/top-3 routing,
dropped assignments, missing shared expert, hidden averaging, broken
renormalization, FP16 router math, wrong RoPE theta, wrong placement,
untied head, bias, unshifted loss, padding-stat contamination, missing
sampler restoration, dirty provenance, and corrupt sampler identity.

## Reproducibility and recovery

A new clone from GitHub at substantive SHA `9555bbcb...` was created in a
new temporary directory with a fresh Python 3.12 venv, upgraded pip,
`requirements-lock.txt`, and editable `--no-deps` install. From the clone
directory:

```text
python scripts/validate_repo.py --all    -> exit 0, 289 passed
python scripts/validate_phase1.py --all  -> exit 0, 289 passed, 33 Phase 1 hashes
```

An initial absolute-path invocation from `/tmp` also passed but made the
standalone foundation log say `git_commit=unknown`; rerunning exactly as
documented from inside the clone logged `9555bbcb...`. The warning is
preserved as methodology evidence, not hidden.

## Remote CI

Substantive reviewed SHA `9555bbcb43d7b4f63762a5f11c2cea13e11fa7c8`:

- Phase 1 Validation run `32931520019`: `success`.
- Phase 0 Validation run `32931520056`: `success`.

The approval-metadata commit is pushed after this report is written and
must itself pass both workflows. Because a commit cannot contain its own
SHA or later CI run IDs, those final identities are stored in the
annotated `phase-1-architecture` tag message, which points at that exact
green metadata commit.

## Commands executed

Key commands included `git fetch --all --tags --prune`, Phase 0 diff and
frozen-config inspection, `python scripts/validate_repo.py --all`,
`python scripts/validate_phase1.py --all`, `pytest tests/ -q`, targeted
architecture/checkpoint/provenance/CUDA tests, exp-0009..0014 commands,
manifest regeneration, GitHub Actions inspection, and the fresh-clone
procedure above. All approval-critical commands used the repository venv
or the fresh recovery venv explicitly; the initial bare `python` attempt
failed because this shell exposes only `python3` outside the venv.

## Negative results preserved

- Sonnet's original sparse profile hit OOM because the profiler retained a
  prior model/optimizer; the failure and repair remain in exp-0008.
- The correctness-first sparse implementation is substantially slower
  than dense (including 1,782 vs 7,510 training tok/s in the corrected
  checkpointed profiles). No favorable benchmark was cherry-picked.
- CUDA attention backward emits a nondeterminism warning; CUDA bitwise
  determinism is not claimed.
- The first Sonnet recovery failed its own absolute-path integrity gate and
  remains documented.
- The first Sol recovery invocation logged unknown Git identity due shell
  working directory; the correct prescribed rerun passed with exact SHA.

## Accepted limitations

- No CUDA bitwise-determinism guarantee.
- Python-loop sparse dispatch remains unoptimized; Phase 2 may study it.
- No tokenizer, real corpus, pretrained weights, intelligence, autonomy,
  expert specialization, or evaluation suite exists yet.
- Only 4,096 context is supported and validated; 16,384 remains an
  unadvertised future target and is rejected by the Phase 1 forward path.
- Sequence curriculum is not implemented; its explicit placeholder now
  survives resume.
- Hardware profiles are single-machine reference measurements, not broad
  performance generalizations.

## Approval statement

Phase 1 satisfies its blocking architecture, mathematical, numerical,
training, checkpoint, provenance, hardware, recovery, repository, and CI
requirements. The exact approved repository state is reproducible. The
approval tag is created only after the final metadata commit's exact
remote CI succeeds. Phase 2 may begin after that tag is fetched and
verified; this approval proves an engineering foundation, not a trained or
intelligent model.
