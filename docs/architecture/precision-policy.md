# Juniper Auto v0.1 Precision Policy

Status: frozen Phase 0 implementation requirement; training and inference are
not implemented in Phase 0.

This policy preserves the numeric baseline that later phases must implement.
It does not claim the policy has been exercised by a model training run.

## Training baseline

- FP16 mixed precision.
- FP32 optimizer state.
- FP32 master parameter state where required.
- FP32 gradient accumulation where practical.
- FP32 RMSNorm reductions.
- FP32 per-head QK-Norm reductions.
- FP32 router logits.
- FP32 router softmax.
- FP32 loss/logit accumulation.
- Dynamic gradient scaling.
- Global gradient clipping; the clipping threshold remains a future
  training-configuration decision and is not invented in Phase 0.

Training-only router jitter is permitted only as a documented Phase 8
experiment. The frozen architecture does not enable it or assign a magnitude.
Evaluation and inference router jitter are disabled.

## Inference baseline

- FP16 reference inference.
- INT8 is an optional later qualification target.
- Weight-only 4-bit is optional only after the reference release is frozen.

Any change to this policy requires a superseding ADR and versioned artifact.
