# Phase 2 Requirements Traceability

Maps every Phase 2 blocking requirement (per the Phase 2 instructions,
section 42 "Blocking Gates", plus the fault-injection checklist in section
19) to the actual implementation file, test, experiment, or CI gate that
proves it -- not to a prose claim.

| # | Requirement | Implementation | Test(s) | Experiment / Validator |
|---|---|---|---|---|
| 1 | Phase 0 still passes | `scripts/validate_repo.py` (unmodified) | -- | `validate_phase2.py`'s `gate_phase1_baseline` chains to it |
| 2 | Phase 1 still passes | `scripts/validate_phase1.py` (unmodified) | -- | `validate_phase2.py::gate_phase1_baseline` |
| 3 | Frozen architecture unchanged | `configs/architecture/*.yaml` (unmodified) | -- | `validate_phase2.py::gate_frozen_architecture_unchanged` |
| 4 | Parameter counts unchanged | `juniper_auto/config/frozen.py` (unmodified) | -- | `gate_frozen_architecture_unchanged`; exp-0015/0019/0020/0021 all build the official model successfully |
| 5 | Reference MoE routing remains correct | `juniper_auto/model/moe_dispatch.py::reference_dispatch` | `tests/test_model_moe.py` (unmodified, passing), `tests/test_model_moe_dispatch.py::test_refactored_reference_dispatch_matches_phase1_golden_bit_for_bit` | -- |
| 6 | Optimized backend numerically equivalent | `moe_dispatch.py::optimized_dispatch` | `tests/test_model_moe_dispatch.py::test_reference_and_optimized_forward_agree`, `..._backward_agree`, `..._agree_under_real_cuda_fp16_autocast` | exp-0015; `validate_phase2.py::gate_reference_optimized_equivalence` |
| 7 | Every non-padding token gets exactly two unique routed experts | `routing.py::select_topk` (torch.topk over probabilities) | `tests/test_model_moe_property.py::test_dropless_invariants_hold_across_randomized_configurations`, `test_no_duplicate_expert_within_one_tokens_topk` | `validate_phase2.py::gate_dropless_invariants` |
| 8 | Shared expert per canonical semantics | `moe.py` (unconditional `shared_expert(flat_x)`) | `tests/test_model_moe.py::test_shared_expert_is_never_gated` (unmodified) | -- |
| 9 | Token dropping remains zero | `moe_dispatch.py` (no capacity logic in either backend) | `tests/test_model_moe_property.py` (assignment-count invariant) | `gate_dropless_invariants` |
| 10 | Token reconstruction is correct | `moe_dispatch.py` (`index_add` by original token index) | `tests/test_model_moe.py::test_token_order_is_preserved_in_reconstruction` (unmodified), `tests/test_model_fault_injection.py::test_reordered_token_reconstruction_is_detected` | -- |
| 11 | Padding excluded from expert execution, output, load, and losses | `moe.py` (valid-token compact dispatch), `losses.py`, `moe_diagnostics.py` | `tests/test_model_moe.py`, `tests/test_model_moe_property.py`, padded golden/equivalence cases | exp-0022/0023 |
| 12 | Full per-token route tracing works with every required audit field | `moe_diagnostics.py::build_token_trace`, `assemble_full_trace` | `tests/test_model_moe_trace.py` | exp-0023 (60/60 records, all 15 layers) |
| 13 | Routing entropy works | `moe_diagnostics.py::compute_entropy` | `tests/test_model_moe_diagnostics.py::test_entropy_matches_hand_computed_value_for_uniform_and_peaked_distributions`, `test_model_fault_injection.py::test_incorrect_entropy_calculation_is_detected` | -- |
| 14 | top-1/top-2 margin works | `moe_diagnostics.py::compute_topk_prob_margin`/`compute_topk_logit_margin` | `tests/test_model_moe_diagnostics.py::test_topk_margins_match_hand_computed_values`, `test_model_fault_injection.py::test_incorrect_topk_margin_is_detected` | -- |
| 15 | Expert load statistics work | `moe_diagnostics.py::RoutingWindowAccumulator` | `tests/test_model_moe_diagnostics.py::test_routing_window_accumulator_aggregates_counts_and_load_shares_across_batches` | -- |
| 16 | Expert-pair coactivation works | `moe_diagnostics.py::compute_expert_pair_coactivation` | `tests/test_model_moe_diagnostics.py::test_expert_pair_coactivation_matches_hand_computed_matrix`, `..._counts_each_pair_once_per_token_not_twice`, `test_model_fault_injection.py::test_incorrect_pair_coactivation_accounting_is_detected` | -- |
| 17 | Contribution metrics work | `moe_diagnostics.py::compute_contribution_norms` | `tests/test_model_moe_diagnostics.py::test_contribution_norms_match_hand_computed_values_with_stub_experts` | -- |
| 18 | Router-logit metrics work | `moe_diagnostics.py::compute_router_logit_magnitude_stats` | `tests/test_model_moe_diagnostics.py::test_router_logit_magnitude_stats_match_hand_computed_values` | -- |
| 19 | Expert gradient norms work per layer/expert | `moe_diagnostics.py::collect_layer_expert_gradient_norms`, `collect_model_expert_gradient_norms` | `test_post_backward_gradient_collector_distinguishes_unselected_experts`; validator all-layer gradient gate | exp-0023 (15/15 layers) |
| 20 | Routing health detectors work on controlled cases with configurable thresholds | `RoutingHealthThresholds`; dead/starved/dominant/collapse/saturation/oscillation functions | `tests/test_model_moe_diagnostics.py` (healthy/pathological + override tests) | exp-0018/0023 |
| 21 | Context-sensitivity probe infrastructure and required categories work | `context_sensitivity.py` low-level/model harness plus `CANONICAL_CONTEXT_PROBE_TEMPLATES` | `tests/test_context_sensitivity.py` | exp-0017; `validate_phase2.py::gate_context_sensitivity_infrastructure` |
| 22 | Semantic specialization is not falsely claimed | Docstrings throughout `context_sensitivity.py` and `docs/architecture/moe-routing-diagnostics.md`'s "What this is not" section | `tests/test_context_sensitivity.py::test_untrained_official_model_probe_runs_and_is_explicitly_labeled` (asserts the labeling text is present) | exp-0017's `result` field states the limitation explicitly |
| 23 | All required ablation modes work, are eval-only, validated, and logged | `moe_ablations.py` (6 modes), `MoEDiagnostics.ablation_mode` | `tests/test_model_moe_ablations.py` (exact math, train-mode rejection, malformed/range rejection, selected/executed trace) | exp-0019 |
| 24 | Seeded random routing reproduces | `moe_ablations.py::resolve_router_override` (`random_router`, isolated `torch.Generator`) | `tests/test_model_moe_ablations.py::test_random_router_is_reproducible_given_the_same_seed`, `..._is_independent_of_ambient_global_rng_state`, `test_model_fault_injection.py::test_non_reproducible_seeded_random_routing_is_detected` | -- |
| 25 | Normal inference unaffected when ablations disabled | `moe.py::MoELayer.forward` (`ablation=None` default path) | `tests/test_model_moe_ablations.py::test_ablation_none_is_byte_identical_to_pre_ablation_forward`, `..._state_does_not_persist_across_calls`, `test_model_fault_injection.py::test_evaluation_ablation_leaking_into_normal_inference_is_detected` | exp-0019 (`does_not_leak_into_next_normal_call`) |
| 26 | FP32 router math survives FP16 AMP | `routing.py::compute_router_logits_and_probs` (unmodified logic) | `tests/test_model_moe.py` (unmodified), `tests/test_model_moe_dispatch.py::test_reference_and_optimized_agree_under_real_cuda_fp16_autocast` (real CUDA) | -- |
| 27 | Randomized routing invariant tests pass | -- | `tests/test_model_moe_property.py` (212 parametrized cases across seed/batch/seq/experts/top_k/padding) | -- |
| 28 | Fault-injection tests prove the suite is load-bearing | -- | See the 20-row fault-injection sub-table below | -- |
| 29 | FLOWBOX profiling executed and documented | `scripts/run_phase2_experiment.py flowbox-moe-profile` | -- | exp-0021 (real RTX 2060 numbers, 5 shapes, both backends) |
| 30 | Performance results reported honestly | ADR-0009 | -- | exp-0021's `result`/`conclusion` fields; no shape cherry-picked (all 5 profiled shapes reported) |
| 31 | Experiment provenance complete | `scripts/run_phase2_experiment.py::_write` (mirrors Phase 1's provenance fields) | -- | exp-0015 through exp-0021, each with git commit/clean-tree/environment/command recorded |
| 32 | Recovery succeeds from a fresh clone | -- | -- | `docs/recovery/phase-2.md` |
| 33 | Phase 2 validator passes | `scripts/validate_phase2.py` | -- | Run directly; also gated in CI |
| 34 | Full pytest suite passes | -- | `pytest tests/ -q` | `validate_phase2.py::gate_pytest`; CI |
| 35 | Required remote CI passes and cannot skip missing golden tag | `.github/workflows/phase-2-validation.yml` (full tagged checkout), golden test/validator hard failure | `tests/test_model_moe_dispatch.py` | final identities in `phase-2-moe` annotated tag |
| 36 | Manifests and hashes current | `manifests/phase-2-artifact-hashes.yaml`, `manifests/phase-2-test-manifest.yaml` | -- | `validate_phase2.py::gate_phase2_artifact_hashes` |
| 37 | Time accounting current | `docs/time/phase-hours.csv` | -- | `validate_phase2.py::gate_phase2_experiment_and_time_records` |
| 38 | README current | `README.md` | -- | manual read-through in self-review Pass L |
| 39 | Phase report current | `docs/phases/phase-2-moe.md` | -- | this document |
| 40 | Requirements traceability complete | -- | -- | this document |
| 41 | Self-review complete | `docs/phases/phase-2-sonnet-self-review.md` | -- | -- |
| 42 | Known failures/negative results preserved | -- | -- | phase-2-moe.md "Known failures"/"Negative results" |
| 43 | Working tree clean, all work pushed | -- | -- | recorded at push time |

## Fault-injection checklist (Phase 2 instructions section 19)

| # | Defect | Test |
|---|---|---|
| 1 | top-1 instead of top-2 | `tests/test_model_fault_injection.py::test_top1_routing_is_detected_by_assignment_count` (Phase 1, unmodified) |
| 2 | top-3 instead of top-2 | `..._top3_routing_is_detected_by_assignment_count` (Phase 1, unmodified) |
| 3 | duplicate expert selected twice | `..._duplicate_expert_selected_twice_is_detected` |
| 4 | dropped token assignment | `..._token_dropping_is_detected_by_assignment_count` (Phase 1, unmodified) |
| 5 | missing shared expert | `..._missing_shared_expert_is_detected` (Phase 1, unmodified) |
| 6 | gated shared expert | `..._gated_shared_expert_is_detected` |
| 7 | hidden output averaging | `..._hidden_averaging_of_moe_output_is_detected` (Phase 1, unmodified) |
| 8 | broken selected-weight renormalization | `..._broken_renormalization_is_detected` (Phase 1, unmodified) |
| 9 | reordered reconstructed tokens | `..._reordered_token_reconstruction_is_detected` |
| 10 | padding counted as valid expert load | `..._padding_included_in_router_statistics_is_detected` (Phase 1, unmodified) |
| 11 | padding affecting auxiliary losses | same test (checks both count and loss) |
| 12 | FP16 router logits | `..._router_fp16_execution_is_detected` (Phase 1, unmodified) |
| 13 | FP16 router softmax | same test |
| 14 | incorrect expert assignment count | tests 1/2/4 above cover this directly |
| 15 | lost token trace record | `..._lost_token_trace_record_is_detected` |
| 16 | corrupted token position in trace | `..._corrupted_token_position_in_trace_is_detected` |
| 17 | incorrect pair co-activation accounting | `..._incorrect_pair_coactivation_accounting_is_detected` |
| 18 | incorrect entropy calculation | `..._incorrect_entropy_calculation_is_detected` |
| 19 | incorrect top-1/top-2 margin | `..._incorrect_topk_margin_is_detected` |
| 20 | missing expert gradient statistics | `..._missing_expert_gradient_statistics_is_detected` |
| 21 | non-reproducible seeded random routing | `..._non_reproducible_seeded_random_routing_is_detected` |
| 22 | evaluation ablation leaking into normal inference | `..._evaluation_ablation_leaking_into_normal_inference_is_detected` |
| 23 | reference/optimized mismatch hidden by overly broad tolerance | `..._reference_optimized_mismatch_is_not_hidden_by_an_overly_broad_tolerance` |

All 23 rows have a concrete, locally-scoped broken variant (monkeypatch or
hand-constructed buggy function) proven to fail the intended check and to
pass for the real implementation, per the file's module docstring
convention.
