# Phase 1 Requirements Traceability

Maps every Phase 1 blocking requirement (per the Phase 1 instructions,
section 68 "Blocking Done Criteria") to the actual implementation file,
test, experiment, or CI gate that proves it -- not to a prose claim.
"Evidence" links to something executable; where a requirement is proven by
more than one thing, all are listed.

| # | Requirement | Implementation | Test(s) | Experiment / CI |
|---|---|---|---|---|
| 1 | Complete sparse reference architecture implemented | `juniper_auto/model/*.py` | `tests/test_model_official_architecture.py` | exp-0003 |
| 2 | Complete dense control implemented in the common stack | `juniper_auto/model/model.py` (`JuniperAutoModel`, shared for both) | `tests/test_model_official_architecture.py::test_dense_control_layer_placement_is_all_dense` | exp-0003 |
| 3 | Sparse instantiated parameter count exactly 150,031,360 | `juniper_auto/model/model.py` | `tests/test_model_official_architecture.py::test_sparse_total_parameters_exact_match_pytorch_and_config` | exp-0003 |
| 4 | Sparse active accounting exactly 79,252,480 | `juniper_auto/accounting/parameter_count.py` (Phase 0, unmodified) | `tests/test_model_official_architecture.py::test_sparse_standard_active_parameters_config_derived` | exp-0003 |
| 5 | Dense instantiated parameter count exactly 79,191,040 | `juniper_auto/model/model.py` | `tests/test_model_official_architecture.py::test_dense_total_parameters_exact_match_pytorch_and_config` | exp-0003 |
| 6 | Weight tying is real parameter sharing | `juniper_auto/model/model.py` (`self.lm_head.weight = self.embedding.weight`) | `tests/test_model_official_architecture.py::test_weight_tying_is_real_object_identity`, `tests/test_model_serialization.py` (survives save/load), `tests/test_model_fault_injection.py::test_untied_lm_head_is_detected` | exp-0003 |
| 7 | GQA is correct | `juniper_auto/model/attention.py` (`repeat_kv`) | `tests/test_model_attention.py` (head mapping, manual reference) | -- |
| 8 | QK-Norm is correct and FP32-reduced | `juniper_auto/model/norm.py`, `attention.py` | `tests/test_model_norm.py`, `tests/test_model_official_architecture.py::test_qk_norm_parameter_count_is_exactly_2560` | -- |
| 9 | RoPE is correct | `juniper_auto/model/rope.py` | `tests/test_model_rope.py` (theta, dims, position dependence, norm preservation) | -- |
| 10 | Causality test passes | `juniper_auto/model/attention.py` (`build_attention_mask`) | `tests/test_model_attention.py::test_causality_...`, `tests/test_model_causality_padding.py::test_full_model_causality_via_future_token_mutation` | -- |
| 11 | Padding/masking tests pass | `attention.py`, `model.py`, `moe.py`, `losses.py` | `tests/test_model_causality_padding.py` (full stack), `tests/test_model_attention.py`, `tests/test_model_moe.py` (padding excluded from stats) | -- |
| 12 | SwiGLU tests pass | `juniper_auto/model/ffn.py` | `tests/test_model_ffn.py` | -- |
| 13 | Exact dense/MoE layer positions pass | `juniper_auto/model/model.py` (layer_kind from `core.dense_layers`/`moe_layers`) | `tests/test_model_official_architecture.py::test_sparse_layer_placement_matches_frozen_dense_moe_positions` | -- |
| 14 | 8 routed + 1 shared expert per MoE layer verified | `juniper_auto/model/moe.py` (`MoELayer`) | `tests/test_model_official_architecture.py` (layer_placement_report), `tests/test_model_moe.py::test_expert_counts_match_config` | -- |
| 15 | Every valid MoE token gets exactly two routed assignments | `moe.py` (dropless dispatch loop) | `tests/test_model_moe.py::test_dropless_assignment_count_equals_valid_tokens_times_top_k`, `test_model_official_architecture.py` (official-model smoke) | -- |
| 16 | Shared expert always active | `moe.py` (`output = self.shared_expert(flat_x)` unconditional) | `tests/test_model_moe.py::test_shared_expert_is_never_gated` | -- |
| 17 | Top-2 weights renormalize to one | `moe.py` (`topk_weights = topk_probs / denom`) | `tests/test_model_moe.py::test_topk_weights_renormalize_to_one`, official-model smoke test | -- |
| 18 | No hidden MoE averaging | `moe.py` (sum combination, no divisor) | `tests/test_model_moe.py::test_shared_plus_topk_routed_combination_with_no_hidden_averaging`, `tests/test_model_fault_injection.py::test_hidden_averaging_of_moe_output_is_detected` | -- |
| 19 | Router logits are FP32 | `moe.py` (`torch.autocast(..., enabled=False)` block) | `tests/test_model_moe.py::test_router_logits_and_softmax_are_fp32_under_fp16_input` and `..._under_cpu_autocast_bf16`, `test_model_fault_injection.py::test_router_fp16_execution_is_detected` | Verified live under real CUDA FP16 autocast during exp-0005 |
| 20 | Router softmax is FP32 | same as above | same as above | same |
| 21 | Load-balancing implementation passes manual-reference tests | `juniper_auto/model/losses.py`, ADR-0008 | `tests/test_model_losses.py::test_load_balance_loss_hand_calculated`, `..._excludes_padding` | -- |
| 22 | Router Z-loss implementation passes manual-reference tests | `losses.py`, ADR-0008 | `tests/test_model_losses.py::test_router_z_loss_hand_calculated`, `..._excludes_padding` | -- |
| 23 | Causal CE passes manual-reference tests | `losses.py::causal_lm_loss` | `tests/test_model_losses.py::test_causal_lm_loss_exact_shift_two_token_example`, `..._off_by_one_shift_is_load_bearing` | -- |
| 24 | Forward passes succeed | `model.py::JuniperAutoModel.forward` | `tests/test_model_shapes.py`, `tests/test_model_official_architecture.py` | exp-0003 through exp-0008 (all forward-dependent) |
| 25 | Backward passes succeed | autograd (no custom backward) | `tests/test_model_gradients.py`, `tests/test_model_official_architecture.py::test_*_forward_backward_smoke_...` | exp-0004, exp-0005 (real backward on GPU) |
| 26 | Expected gradients are finite | -- | `tests/test_model_gradients.py` (`torch.isfinite` on every populated grad) | exp-0004, exp-0005 (`any_nonfinite_event: false`) |
| 27 | Optimizer update succeeds | `torch.optim.AdamW` (standard) | `tests/test_model_optimizer.py` | exp-0004, exp-0005 |
| 28 | Initialization tests pass | `model.py::initialize_weights` | `tests/test_model_initialization.py` (seed reproducibility, statistical std, ones-init norms) | -- |
| 29 | Serialization/deserialization passes | `torch.nn.Module.state_dict`/`load_state_dict` (standard) | `tests/test_model_serialization.py` | -- |
| 30 | Dense full-model tiny-batch overfit passes its predefined gate | `juniper_auto/training/tiny_overfit.py` | -- | exp-0004: LM loss 10.60 -> 0.0012, 100% token accuracy, 0 non-finite events |
| 31 | Sparse full-model tiny-batch overfit passes its predefined gate | `tiny_overfit.py` | -- | exp-0005: LM loss 10.61 -> 0.0003, 100% token accuracy, 0 non-finite events |
| 32 | Checkpoint contains every required state | `juniper_auto/training/checkpoint.py` (`REQUIRED_CHECKPOINT_FIELDS`) | `tests/test_training_checkpoint.py` (validation rejects any missing field) | exp-0006, exp-0007, exp-0008 (real checkpoints written/read) |
| 33 | Interrupted/resumed training comparison passes defined criteria | `tiny_overfit.py::TinyOverfitHarness` + `checkpoint.py` | `tests/test_training_resume_equivalence.py` (bit-exact, two negative controls) | exp-0006: exact match on full 150M sparse model, CPU |
| 34 | FLOWBOX dense profile completed | `juniper_auto/training/profiling.py` | -- | exp-0007 |
| 35 | FLOWBOX sparse profile completed | `profiling.py` | -- | exp-0008 |
| 36 | Practical microbatch/accumulation path demonstrated | `profiling.py::profile_training_step` | -- | exp-0007, exp-0008 (microbatch=2, seq=512, grad_accum=4, both fit comfortably under 6GB) |
| 37 | Any OOM/negative hardware results preserved | -- | -- | exp-0008's `result` field documents the earlier OOM and its (script, not architecture) root cause; not deleted |
| 38 | Phase 0 validation remains green | `scripts/validate_repo.py` unmodified | -- | `scripts/validate_phase1.py`'s first gate re-runs it; CI |
| 39 | Phase 1 validation is green | `scripts/validate_phase1.py` | -- | `.github/workflows/phase-1-validation.yml` |
| 40 | Full CPU-safe test suite is green | -- | `pytest tests/ -q` | Both CI workflows |
| 41 | Phase 1 requirements traceability complete | this document | -- | -- |
| 42 | Phase 1 experiment registry complete | `experiments/registry.yaml` (exp-0003..exp-0008) | `tests/test_experiment_registry.py` | -- |
| 43 | Phase 1 time accounting complete | `docs/time/phase-hours.csv` | `tests/test_time_schema.py` | -- |
| 44 | Phase 1 artifact hashes complete | `manifests/phase-1-artifact-hashes.yaml`, `manifests/phase-1-test-manifest.yaml` | `scripts/validate_phase1.py::gate_phase1_artifact_hashes` | -- |
| 45 | Phase 1 recovery documentation complete | `docs/recovery/phase-1.md` | -- | actual fresh-clone exercise, see that document |
| 46 | Remote fresh-clone recovery passes | -- | -- | `docs/recovery/phase-1.md` |
| 47 | README accurately reflects candidate status | `README.md` | -- | -- |
| 48 | Sonnet self-review completed | `docs/phases/phase-1-sonnet-self-review.md` | -- | -- |
| 49 | Every self-review blocking/major defect repaired | same | -- | -- |
| 50 | Final working tree clean | -- | -- | `git status` in recovery doc |
| 51 | Exact final Sonnet candidate pushed | -- | -- | final handoff message |
| 52 | Exact final Sonnet candidate passes remote CI | -- | -- | final handoff message |
| 53 | Phase report says `CANDIDATE - PENDING INDEPENDENT REVIEW` | `docs/phases/phase-1-architecture.md` | -- | -- |
| 54 | No `phase-1-architecture` approval tag created by Sonnet | -- | -- | verified in final handoff |

## Requirements satisfied only by prose (none found as of this writing)

Every row above links to an executable check. If a future edit adds a
claim without a corresponding test/experiment/gate, that is itself a
Phase 1 self-review defect (see self-review Pass A).

## Explicitly out of scope (not tracked as unmet requirements)

Per the Phase 1 instructions' scope boundaries: tokenizer training,
dataset acquisition, production pretraining, Phase 2 expert-specialization
research, exotic/optimized MoE kernels, distributed expert parallelism,
runtime autonomy, tools, memory, persistent autonomous state, evaluation
suite (Phase 5), 6B-token training, instruction tuning, quantization,
native multimodality, architectural recurrence, specialist language
models, hierarchical MoE, dynamic expert creation, self-modification. None
of these were attempted, and none are represented as complete anywhere in
this repository.
