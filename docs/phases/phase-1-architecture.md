# Phase 1 Report: Reference Model and Dense Control Architecture

## Phase

Phase 1: Reference Model and Dense Control Architecture.

## Objective

Implement the complete, executable reference architecture for
`ja150m-v0.1` (150,031,360-parameter sparse MoE) and
`ja150m-v0.1-dense` (79,191,040-parameter mandatory dense control) on one
shared, inspectable PyTorch stack, and prove -- with real executed
evidence, not documentation claims -- that the frozen architecture is
correct software: forward propagation, causal LM loss, backward
propagation, optimizer updates, FP16 mixed-precision execution on real
CUDA hardware, tiny-batch overfitting, checkpoint/save/restore,
interrupted/resumed training, and hardware profiling on FLOWBOX.

## Starting commit

`f6f8b5397b41d31081f55cdd60cb6363ec052be4` (tag `phase-0-foundation`,
verified against `origin/main` before any Phase 1 work began; Phase 0 CI
run `32879488449` was green at this exact commit).

## Final commit

The Sonnet final metadata candidate was
`6a7410130e4b5ae6039a794f66f9360b675fd945`. GPT-5.6 Sol's substantive
reviewed candidate is `9555bbcb43d7b4f63762a5f11c2cea13e11fa7c8`.
The final approval-metadata commit is resolved by the annotated
`phase-1-architecture` tag, avoiding impossible commit self-reference.
See `phase-1-sol-independent-review.md`.

## Implementation summary

- **`juniper_auto/model/`** (new package): `norm.py` (RMSNorm, reused for
  per-head QK-Norm), `rope.py` (rotary positional encoding), `attention.py`
  (causal GQA with a unified causal+padding mask), `ffn.py` (shared
  bias-free SwiGLU), `moe.py` (FP32 router, dropless top-2 dispatch,
  always-active ungated shared expert), `losses.py` (shifted causal CE,
  load-balance and router Z-loss per ADR-0008), `block.py` (Pre-Norm
  Dense/MoE blocks), `model.py` (`JuniperAutoModel`, `build_model`,
  frozen-policy initialization), `inspection.py` (independent Method B
  parameter accounting and structural audits). See
  `docs/architecture/reference-model-implementation.md` for the full
  module-by-module description.
- **`juniper_auto/training/`** (new package): `state.py` (RNG state
  capture/restore, `SyntheticSequenceStream`), `checkpoint.py` (versioned
  checkpoint format), `tiny_overfit.py` (`TinyOverfitHarness`,
  `run_tiny_overfit`), `profiling.py` (FLOWBOX inference/training-step/
  checkpoint-I/O profiling).
- **`docs/adr/0008-moe-auxiliary-loss-semantics.md`**: defines the
  load-balancing and router Z-loss formulas (not specified numerically by
  the frozen coefficients alone).
- **`scripts/run_phase1_experiment.py`**: reproducible CLI driving every
  Phase 1 experiment against the real official architectures.
- **`scripts/validate_phase1.py`**: the Phase 1 CPU-safe validation
  entrypoint (see CI workflow section).
- **`scripts/hash_manifest.py`** (extended, `--phase 0|1`) and
  **`scripts/generate_phase1_test_manifest.py`** (new): Phase 1
  artifact-hash tooling.
- **289 automated tests in the final substantive suite**, including
  deliberate fault injection and independent provenance, checkpoint,
  sampler, context, invalid-shape, profiling, and live-CUDA checks.

## Architecture / configuration IDs

`ja150m-v0.1` (sparse), `ja150m-v0.1-dense` (dense control). Neither
frozen architecture config was modified; Phase 1 implements them, it does
not redefine them.

## Environment

See `docs/architecture/environment-specification.md` for the general
Phase 0 baseline. Phase 1 GPU work ran on the actual FLOWBOX machine:
Ryzen 7 5700G, NVIDIA GeForce RTX 2060 (6 GB VRAM, driver 13.0), 16 GB
RAM, Linux, Python 3.12.3, `torch==2.13.0+cu130` (CUDA available and used
directly, not simulated). CPU-only work (the deterministic
checkpoint/resume comparison, and all CI) used the same machine's CPU
path with `CUDA_VISIBLE_DEVICES` unset but `device="cpu"` explicitly
requested.

## Artifacts

| Artifact | Status (per `manifests/frozen-artifacts.yaml`) |
|---|---|
| `ja150m-v0.1` / `ja150m-v0.1-dense` configs | `frozen` (unchanged from Phase 0) |
| Reference model implementation (`juniper_auto/model/`, `juniper_auto/training/`) | `frozen` (new Phase 1 entry, hash-tracked) |
| Checkpoint format v1 | `frozen` (new Phase 1 entry) |
| Tokenizer, pretraining/post-training datasets, base/instruction checkpoints, runtime protocol, evaluation suite | `not-yet-created` (unchanged -- Phase 1 does not create any of these) |

## Hashes

`manifests/phase-1-artifact-hashes.yaml` (model/training source, ADR-0008,
`reference-model-implementation.md`, the two Phase 1 scripts, and the
Phase 1 test manifest) and `manifests/phase-1-test-manifest.yaml` (every
Phase 1 test file). `manifests/phase-0-artifact-hashes.yaml` is untouched.

## Tests

`pytest tests/ -q`: 289 passed on the substantive reviewed candidate,
locally and in clean remote CI. CUDA-capable local execution emits the
accepted PyTorch nondeterministic-attention warning.

## Evaluations

`not-applicable`. Phase 1 does not implement or run the evaluation suite
(Phase 5 scope) -- this phase produces no language-capability claims.

## Ablations

`not-applicable`. No ablation study was performed; Phase 1's tiny-overfit
gates are plumbing-correctness checks, not a research comparison.

## Experiments

Twelve Phase 1 experiments are registered. Sonnet's exp-0003..0008 are
preserved as historical evidence. Approval relies on Sol's clean-tree,
config-hashed reruns exp-0009..0014; see the independent review report.

| ID | Experiment | Key result |
|---|---|---|
| exp-0003 | Parameter verification (Method A vs Method B) | Exact match: sparse 150,031,360 / 79,252,480 active, dense 79,191,040 |
| exp-0004 | Dense tiny-batch overfit | LM loss 10.60 -> 0.0012, 100% token accuracy, 0 non-finite events |
| exp-0005 | Sparse tiny-batch overfit | LM loss 10.61 -> 0.0003, 100% token accuracy, 0 non-finite events, aux losses bounded |
| exp-0006 | Checkpoint/resume equivalence (CPU, exact) | Bit-exact match (losses, parameters, AdamW state) to uninterrupted training |
| exp-0007 | FLOWBOX dense hardware profile | FP16 inference 27,574 tok/s; 4,096-token batch-1 inference succeeds at 1.09 GB peak VRAM |
| exp-0008 | FLOWBOX sparse hardware profile | FP16 inference 5,113 tok/s (near-FP32-parity, see Known limitations); 4,096-token batch-1 inference succeeds at 1.23 GB peak VRAM |

Independent approval reruns:

| ID | Experiment | Key result |
|---|---|---|
| exp-0009 | Parameter verification | Exact counts, tied identity, clean provenance |
| exp-0010 | Dense tiny overfit | Loss 10.6038 → 0.001306; 100%; gate passed |
| exp-0011 | Sparse tiny overfit | Loss 10.6148 → 0.000287; 100%; gate passed |
| exp-0012 | Resume equivalence | Bit-exact parameters/losses/optimizer/counters/next batch |
| exp-0013 | Corrected dense profile | Checkpointed training 7,510 tok/s; 4,096 succeeds |
| exp-0014 | Corrected sparse profile | Checkpointed training 1,782 tok/s; 4,096 succeeds |

## CI workflow / run

`.github/workflows/phase-1-validation.yml`, job `validate-phase1`, running
`python scripts/validate_phase1.py --all` (which itself runs the Phase 0
baseline first) on a clean `ubuntu-latest` GitHub Actions runner, no GPU.
Substantive reviewed SHA `9555bbcb43d7b4f63762a5f11c2cea13e11fa7c8`:
Phase 1 run `32931520019` success; Phase 0 run `32931520056` success.
Final metadata-commit CI identities are recorded in the annotated approval
tag because they cannot be embedded in their own commit.

## Recovery status

Performed as an actual fresh-clone exercise (not merely described) after
the final candidate commit was pushed -- see `docs/recovery/phase-1.md`
for the exact commands and output.

## Engineering hours

See `docs/time/phase-hours.csv`, `phase-1` row(s). Approximately 1.4
engineering hours (AI-assisted) at time of writing this report section,
recorded as `PENDING`/in-progress until this phase's remaining
documentation and self-review work concludes and the row is closed with a
final `end_time`.

## Self-review hours

See `docs/time/phase-hours.csv`. Recorded as a separate row from
engineering, per governance rule 6, once the dedicated self-review pass
(`docs/phases/phase-1-sonnet-self-review.md`) begins.

## Independent review hours

Completed by GPT-5.6 Sol; see `docs/time/phase-hours.csv` and
`phase-1-sol-independent-review.md`.

## GPU hours

See `docs/time/phase-hours.csv`. Approximately 0.6 hours, covering the six
executed experiments (most on the RTX 2060; exp-0006 ran on CPU by
design).

## CPU / data-processing hours

See `docs/time/phase-hours.csv`. Approximately 0.1 hours (no dataset
processing occurred -- Phase 1 uses only synthetic engineering data).

## Project elapsed days

0 days since Phase 0's approval commit (`f6f8b539`, 2026-08-25) -- Phase 1
engineering began the same day.

## Known failures

- An early run of the FLOWBOX sparse hardware profile (exp-0008) hit a
  CUDA out-of-memory error during the checkpoint-I/O stage. Root cause:
  `scripts/run_phase1_experiment.py` did not release the prior
  training-step model/optimizer's GPU memory before constructing a second
  full model for the checkpoint probe. This was a profiling-script bug,
  not an architecture or hardware limitation; fixed with explicit
  `del` + `torch.cuda.empty_cache()` between profiling stages and rerun
  to a clean result. See exp-0008's `result` field in
  `experiments/registry.yaml` for the full account -- the failed run is
  documented, not deleted.

## Negative results / limitations found through testing

- **The reference (Phase 1) MoE dispatch shows almost no FP16 speedup
  over FP32 at batch=1** (5,011 vs 5,113 tokens/sec, exp-0008), unlike the
  dense model (17,191 vs 27,574 tokens/sec, exp-0007). The reference
  dispatch loops over 8 experts per layer in Python rather than using a
  batched/fused kernel, so at small batch sizes it is kernel-launch/
  overhead-bound rather than compute-bound -- FP16 reduces per-op compute
  time but not the fixed per-launch overhead that dominates here. This is
  an accurate, expected property of a correctness-first reference
  implementation; optimizing MoE dispatch throughput is explicitly Phase 2
  scope (see instructions section 16), not something this phase attempts
  or claims to have solved.

## Accepted limitations

- **CUDA determinism is not claimed.** Per `juniper_auto/util/seed.py`
  (inherited from Phase 0) and confirmed by a `UserWarning` PyTorch itself
  emits during GPU training (`torch.autograd.graph`: "Memory Efficient
  attention defaults to a non-deterministic algorithm"), bitwise
  reproducibility on CUDA is not guaranteed. The checkpoint/resume
  equivalence experiment (exp-0006) therefore ran on CPU, where exact
  reproducibility is expected and was confirmed bit-for-bit.
- **No sequence-length curriculum is implemented.** The checkpoint format
  has a `sequence_curriculum_state` field, honestly recorded as
  `{"status": "not-implemented-phase-1"}` rather than fabricated.
- **16K context is not validated.** Only the frozen 4,096-token
  `context_length` was exercised (successfully, batch=1, both
  architectures). `future_context_target: 16384` remains
  `future_context_advertised: false`, unchanged from Phase 0.
- **The reference MoE dispatch is not optimized.** See Negative results
  above.
- **Tokenizer, pretraining corpus, and any trained checkpoint remain
  not-yet-created**, per `manifests/frozen-artifacts.yaml`. All Phase 1
  training uses honestly-labeled synthetic engineering data
  (`SyntheticSequenceStream`), never a fabricated dataset identity.

## Reproducibility procedure

```bash
git clone https://github.com/Cinqic/Juniper-Auto.git
cd Juniper-Auto
git checkout <candidate-commit>
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
pip install -e . --no-deps
python scripts/validate_repo.py --all      # Phase 0 baseline
python scripts/validate_phase1.py --all    # Phase 1 (includes the Phase 0 baseline + full pytest suite)
```

GPU-only experiments (tiny-overfit, hardware profiling) are not re-run by
`validate_phase1.py` (CI has no GPU); to reproduce them on hardware with a
CUDA-capable GPU:

```bash
python scripts/run_phase1_experiment.py param-verification --output /tmp/exp3.json
python scripts/run_phase1_experiment.py dense-overfit --output /tmp/exp4.json
python scripts/run_phase1_experiment.py sparse-overfit --output /tmp/exp5.json
python scripts/run_phase1_experiment.py resume-equivalence --architecture sparse --device cpu --output /tmp/exp6.json
python scripts/run_phase1_experiment.py profile --architecture dense --output /tmp/exp7.json
python scripts/run_phase1_experiment.py profile --architecture sparse --output /tmp/exp8.json
```

See `docs/recovery/phase-1.md` for the actual executed fresh-clone
recovery output.

## Reviewer identity

Self-review: Claude Sonnet 5 (implementer). Independent reviewer and Phase
1 approval authority: GPT-5.6 Sol.

## Approval status

`APPROVED WITH ACCEPTED LIMITATIONS BY GPT-5.6 SOL`

Canonical approval identity is the annotated `phase-1-architecture` tag
after exact metadata-commit remote CI succeeds.
