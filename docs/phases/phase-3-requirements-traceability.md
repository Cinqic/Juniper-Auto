# Phase 3 Requirements Traceability

Maps every Phase 3 master-specification requirement to its implementation,
test(s), experiment/evidence, and documentation. Anything unmapped is
unfinished.

Legend: **I** = implementation, **T** = test, **E** = experiment / executed
evidence, **D** = documentation.

## Identity & vocabulary contract

| Requirement | Mapping |
|---|---|
| ID `ja-tokenizer-v0.1` | I `constants.TOKENIZER_ID`; T `test_tokenizer_config_and_model`; D design §1 |
| Algorithm: UTF-8 byte-level BPE | I `bpe.py`, `bytelevel.py`; T `test_tokenizer_roundtrip`; D design §2; ADR-0010 |
| Total vocab exactly 36,864 | I `constants._self_check`, `tokenizer._validate_invariants`; T `test_tokenizer_vocab::test_exact_vocab_size`; E exp-0026; D design §8 |
| Vocab includes byte fallback + specials + reserved within 36,864 | I `constants` layout; T `test_tokenizer_vocab::test_id_space_is_contiguous_and_hole_free`, `::test_control_block_is_256_ids`; D design §8; ADR-0011 |
| Highest id ≤ model vocab (36,863 < 36,864) | I `constants.LEARNED_VOCAB_END`; T `test_tokenizer_vocab::test_highest_id_within_frozen_model_vocab` |
| No "36,864 BPE tokens then extra control tokens" bug | I control block at fixed `[512-256, 512-1]` offset, learned count = `36864-512`; T `test_tokenizer_vocab::test_learned_vocab_starts_at_512_and_has_exact_count`; D ADR-0011 alternatives |

## Losslessness

| Requirement | Mapping |
|---|---|
| Round-trip arbitrary valid UTF-8 losslessly | I `encode`/`decode`; T `test_tokenizer_roundtrip::test_property_random_full_unicode_roundtrip`; E exp-0025 (20k property cases, whole corpus, fixture); D ADR-0012 |
| Preserve spaces/tabs/newlines/indentation/repeated whitespace exactly | T `test_tokenizer_roundtrip` (whitespace, indentation, crlf), `test_tokenizer_normalization` |
| Preserve code punctuation & structured-data syntax exactly | T `test_tokenizer_roundtrip` (json/yaml/xml_html/python), `test_tokenizer_normalization::test_structured_syntax_not_destructively_normalized` |
| No lowercasing / trimming / whitespace collapse / destructive Unicode norm | I normalization = none; T `test_tokenizer_normalization` (case, whitespace, NFC/NFD, NFKC, BOM, path separators); D ADR-0012 |
| Identity normalizer chosen deliberately & documented | I `tokenizer_config.json` `"normalization":"none"`; D ADR-0012; config `normalization.rationale` |
| Byte fallback path (not `<unk>` swallowing) | I 256 base byte tokens, no `<unk>`; T `test_tokenizer_roundtrip::test_all_256_single_bytes_roundtrip`, `test_tokenizer_fault_injection::test_silent_unk_substitution_breaks_roundtrip`; E exp-0025 (`all_256_bytes_roundtrip_ok`, `property_byte_fallback_rate`); D design §5, §16 |
| Unknown/fallback behaviour measured & documented | E exp-0025 (`property_byte_fallback_rate`), exp-0026 (`byte_fallback_rate` per domain); D design §13 |
| Silent info loss on valid input = blocker | T `test_tokenizer_fault_injection::test_decode_information_loss_breaks_roundtrip`; validator `gate_roundtrip_smoke`, `gate_byte_fallback` |

## Determinism

| Requirement | Mapping |
|---|---|
| Deterministic training input ordering / corpus sampling | I `corpus.load_corpus_shards` (manifest order), `build_word_frequencies`; D design §6-7 |
| Deterministic vocabulary construction / merge ordering | I `bpe.train_bpe` (`argmax` by count then smallest pair); T `test_tokenizer_determinism::test_small_bpe_train_is_order_deterministic`; D design §2 |
| Deterministic special-token assignment | I explicit `constants.CORE_CONTROL_TOKENS` list; T `test_tokenizer_vocab`, `test_tokenizer_control_tokens::test_special_token_serialization_roundtrip` |
| Deterministic encode / decode | T `test_tokenizer_determinism::test_repeated_encode_is_identical`, `::test_repeated_decode_is_identical`, `::test_cross_process_encode_matches`; validator `gate_roundtrip_smoke` |
| Deterministic serialization / reload | T `test_tokenizer_determinism::test_save_reload_is_byte_identical`; validator `gate_serialization_reload` |
| Prove deterministic rebuild by hash comparison | I `train.py`; T `test_tokenizer_determinism::test_full_rebuild_from_committed_corpus_is_hash_identical`; E exp-0024 (2 independent rebuilds, all artifact hashes identical to canonical) |
| Nondeterminism must be disclosed, not hidden | D design §7, §16; no third-party nondeterminism (ADR-0010) |

## Core control tokens

| Requirement | Mapping |
|---|---|
| 15 named core tokens present | I `constants.CORE_CONTROL_TOKENS`; T `test_tokenizer_vocab::test_every_core_control_token_exists_with_expected_id`; E exp-0028 |
| Frozen spelling / numeric id / semantics / behaviour | I `constants` + `CORE_CONTROL_SEMANTICS`; D design §9; ADR-0011; T same |
| Every id unique | T `test_tokenizer_vocab::test_core_ids_are_unique`; E exp-0028 (`core_ids_unique`) |
| Ids stable after serialization/reload & rebuild | T `test_tokenizer_determinism::test_special_ids_stable_after_reload`; E exp-0024 |
| Not defined by dict insertion order | I explicit `(str,int)` list; `ControlToken.id` reads `CORE_CONTROL_STR_TO_ID`; D ADR-0011 |

## Reserved future-control range

| Requirement | Mapping |
|---|---|
| Explicit contiguous documented range of unused ids | I `constants.RESERVED_CONTROL_*` = `[271,511]`; D design §10; ADR-0011 |
| Core & reserved do not overlap | T `test_tokenizer_vocab::test_core_and_reserved_do_not_overlap`; validator `gate_vocab_invariants` |
| Reserved ids never produced by normal BPE | T `test_tokenizer_vocab::test_reserved_ids_never_produced_by_normal_tokenization`; E exp-0028 (`reserved_ids_produced_by_normal_tokenization: []`) |
| Reserved ids have stable numeric identity | I explicit; T `test_tokenizer_control_tokens::test_reserved_control_handles` |
| Reserved ids part of the 36,864 total | I layout; T `test_tokenizer_vocab::test_control_block_is_256_ids` |
| Ordinary text never maps to an unused control id | T `test_tokenizer_control_tokens::test_ordinary_text_with_control_looking_strings_never_yields_control_ids`; validator `gate_roundtrip_smoke` |
| Future protocols activate reserved ids without retraining | D ADR-0011 "Consequences"; design §17 |
| No permanent per-tool token | D ADR-0011 alternatives; design §10 |
| Documented rationale for chosen range | D ADR-0011; config `reserved_control_range.rationale` |

## Control-token safety contract

| Requirement | Mapping |
|---|---|
| Untrusted text with literal `<\|system\|>` gains no authority / no control id | I `encode` byte-only path; T `test_tokenizer_control_tokens::test_untrusted_text_path_and_control_path_are_distinct`, `test_tokenizer_fault_injection::test_special_token_injection_ambiguity_is_caught`; E exp-0028 (`literal_control_string_leak: false`) |
| Explicit API contract: ordinary vs deliberate control insertion | I `encode` vs `build_sequence`/`encode_control`; D design §11 |
| Adjacent control tokens / control next to arbitrary Unicode | T `test_tokenizer_control_tokens::test_adjacent_control_tokens`, `::test_control_tokens_next_to_arbitrary_unicode` |
| Not claimed as a security boundary | D design §11; ADR-0011 |
| Tested & documented for Phase 4 | T `test_tokenizer_control_tokens`; E exp-0028; D design §11 |

## Training corpus

| Requirement | Mapping |
|---|---|
| Tokenizer-training corpus, not Phase 6 pretraining corpus | I `corpus.py` docstring; `corpus-manifest.json` `purpose`; D design §6 |
| Representative samples of all required domains | I `REPO_SOURCES` + `STDLIB_SOURCE` + `SYNTHETIC_SOURCES` (12 categories); D design §6 table |
| Generic prose not dominant | I `synthetic-prose` capped ~0.36 MB of ~8.9 MB; D design §6 |
| Provenance per external source (name, identity, URL, revision, date, license, redistribution, hash, method, counts, category, transformation) | I `corpus-manifest.json` per-shard fields; validator `gate_corpus_manifest_and_shards`; D design §6 |
| Synthetic/control examples identified as synthetic | I `builder: synthetic`, `category: synthetic-*`; T validator gate; D design §6 (governance rule 36) |
| No private files / secrets / credentials | I curated globs; repo-integrity secret scan (Phase 0 gate, runs in baseline); D design §6 |
| External data is data not instructions | governance rule 16; corpus is inert text used only for BPE counting |
| Prefer immutable / revision-pinned sources | I stdlib (version-pinned), repo files, seeded synthetic; committed shards are canonical; D design §6 "Reproducibility note" |
| Commit the derived corpus if small & redistributable | I `data/tokenizer/corpus/` committed (24 shards < 480 KB each, all MIT/PSF); D design §6 |
| Held-out eval separate from training | I `CORPUS_EXCLUDE_GLOBS`; T `test_tokenizer_evaluation::test_eval_fixture_is_disjoint_from_training_corpus` |

## Dependency decision

| Requirement | Mapping |
|---|---|
| Decide implement-in-project vs adopt a library, on the full criteria list | D ADR-0010 (deterministic training, byte fallback, transparency, freezing, special-token control, portability, license, maintainability, dependency size, CPU perf, modifiability) |
| If adding a dependency: ADR, justify, pin, update pyproject, regen lock, test clean install, no runtime downloads | N/A — no dependency added; `pyproject.toml` / `requirements-lock.txt` unchanged; D ADR-0010 |
| Comparator-only deps not mandatory runtime deps | I `comparators.py` (gitignored cache, never imported at runtime); D ADR-0010, design §14 |
| Runtime tokenizer load needs no internet | I `load_canonical_tokenizer`; T `test_tokenizer_determinism::test_cross_process_encode_matches` (subprocess, no network); validator `gate_frozen_artifact_and_identity` |

## Repository integration

| Requirement | Mapping |
|---|---|
| `configs/tokenizer/`, `juniper_auto/tokenizer/`, `data/tokenizer/`, `docs/`, `manifests/`, `scripts/`, `tests/` equivalents | I all created; D phase report "Implementation summary" |
| Canonical config exposes everything to reconstruct | I `configs/tokenizer/ja-tokenizer-v0.1.yaml` (id, vocab, algorithm, byte-level, normalization, pre-tokenization, seed, corpus-manifest identity, control map, reserved range, bos/eos, artifact paths/hashes) |
| Model stays independent of tokenizer impl; consumes int ids | I no change to `juniper_auto/model/`; T existing model tests unchanged |
| Compatibility check: tokenizer.vocab_size == model vocab (sparse & dense) | I `assert_model_compatible`; T `test_tokenizer_config_and_model::test_tokenizer_vocab_equals_model_vocab`; validator `gate_model_compatibility` |

## Frozen artifacts

| Requirement | Mapping |
|---|---|
| Freeze model/serialization, vocab, merges, config, special-token map, reserved range/map, corpus manifest, provenance, hashes, identity, training config | I `tokenizer.save` writes all; `data/tokenizer/ja-tokenizer-v0.1/hashes.json`; `manifests/phase-3-artifact-hashes.yaml` |
| Human-auditable vocab + merges exported, not one opaque blob | I `vocab.json` + `merges.txt` (text); D design §12; ADR-0010 |
| Loadable locally without network | I `load_canonical_tokenizer(verify_hashes=True)` |
| Corruption / hash mismatch fails loudly | I `verify_artifact_hashes`, `_validate_invariants`; T `test_tokenizer_failure_handling` (12 cases); validator `gate_frozen_artifact_and_identity` |
| `frozen-artifacts.yaml`: `tokenizer` → frozen with id; `special_token_map` → frozen with version/location | I `manifests/frozen-artifacts.yaml` updated; T `test_manifests` (existing) still passes |
| Don't mark frozen until artifact exists & hashes generated | I done in the metadata-closure commit after `hash_manifest.py --phase 3` |

## Automated tests

| Requirement group | Mapping |
|---|---|
| Vocabulary invariants (size, uniqueness, range, holes, core ids, reserved exact, no overlap, model agreement) | T `test_tokenizer_vocab.py` (13 tests) |
| Round-trip correctness (empty, ascii, prose, whitespace variants, tabs, newlines, crlf, indentation, json/yaml/xml/shell/paths/urls/git/logs/math/scientific/emoji/accents/CJK/RTL/combining/zero-width/mixed-scripts/tool-traces/records) | T `test_tokenizer_roundtrip.py` + `tokenizer_fixtures.DOMAIN_SAMPLES` |
| Randomized/property valid-Unicode round-trip | T `test_tokenizer_roundtrip::test_property_random_*`; E exp-0025 |
| Normalization invariants | T `test_tokenizer_normalization.py` (9 tests) |
| Byte/fallback behaviour measured & asserted | T `test_tokenizer_roundtrip::test_all_256_single_bytes_roundtrip`, `test_tokenizer_evaluation`; E exp-0025 |
| Determinism (repeat, save/reload, cross-process, rebuild, hash stability, special-id stability) | T `test_tokenizer_determinism.py` (8 tests); E exp-0024 |
| Control-token behaviour (explicit insertion, look-alike text, accidental manufacture, adjacency, near-Unicode, serialization) | T `test_tokenizer_control_tokens.py` (10 tests); E exp-0028 |
| Failure handling (wrong size, missing special, dup id, reserved/core overlap, corrupt vocab/merges, invalid manifest, id mismatch, hash mismatch, arch mismatch) | T `test_tokenizer_failure_handling.py` (13 tests) |
| Fault injection proving gates load-bearing (destructive norm, whitespace strip, altered special id, missing fallback, nondeterministic ordering, reserved collision, wrong total, decode loss, silent `<unk>`, injection ambiguity, model mismatch) | T `test_tokenizer_fault_injection.py` (12 tests) |

## Held-out evaluation & required metrics

| Requirement | Mapping |
|---|---|
| Eval corpus separate from training examples | I `data/tokenizer/eval/held-out-eval-fixture.json`; T disjointness test |
| Difficult representative samples per domain | I 15 domains × difficult samples; E exp-0030 (manual-inspection artifact with token pieces/ids) |
| Version + hash the fixture | I `version: 1.0.0`; hash in `manifests/phase-3-artifact-hashes.yaml` + phase report |
| Measure chars/token, bytes/token, tokens/line (code), tokens/expression (math), structural fragmentation, unknown/fallback | I `evaluation.DomainMetrics`; E exp-0026 |
| `structural fragmentation` precisely defined | D `evaluation.py` docstring; design §13 |
| Manual inspection of difficult examples w/ token pieces/ids | E exp-0030 (`token_ids`, `token_pieces` per record) |
| No single blended average hiding bad domains | E exp-0026 reports per-domain; report tables per-domain |

## Baseline comparisons

| Requirement | Mapping |
|---|---|
| ≥1 general-purpose tokenizer comparator | I/E `utf8-bytes` (exp-0027) |
| ≥1 small-model tokenizer comparator | I/E GPT-2 124M (exp-0027) |
| Pin/document comparator name, version, source, artifact identity/hash, config, special-token handling | E exp-0027 (`gpt2.identity`: revision `607a30d…`, per-file SHA-256, pretokenizer note, "special tokens excluded on both sides") |
| No cherry-picking; report better / equal / worse | D design §14; report "Evaluations"; E exp-0027 all 15 domains |
| No universal-superiority claim | D design §14 last paragraph; report |

## Acceptance criteria

| Requirement | Mapping |
|---|---|
| Define acceptance criteria BEFORE final eval; record rationale; freeze in config/docs; then run | I `configs/tokenizer/ja-tokenizer-v0.1.yaml` `acceptance_criteria`; D design §13-14 |
| Hard gates (zero round-trip corruption, deterministic encode/decode, stable special ids, exact vocab, functional fallback, preserved structure, valid hashes, model compat) | I config `acceptance_criteria.hard_gates`; T across suite; E exp-0024/0025/0026/0028; validator gates |
| Defensible domain-level efficiency gates from baselines | I config `acceptance_criteria.efficiency_gates`; evidence exp-0026 vs exp-0027 |
| Failed candidate preserved, corpus improved, re-evaluated, evidence not overwritten | The frequency-1 tail-fill observation + `merges.txt` parser bug + corpus contamination fix are recorded in the self-review; no evidence overwritten |

## Consumer-hardware measurements

| Requirement | Mapping |
|---|---|
| Training wall time, CPU processing time, peak RAM, artifact size, encode throughput, decode throughput | E exp-0029 (`training.wall_seconds_*`, `training.peak_python_heap_bytes`, `artifact_size_bytes`, `encode.chars_per_second`, `decode.tokens_per_second`) |
| GPU-hours = 0 if no GPU work | E exp-0029 `gpu_hours: 0.0`; report "GPU hours"; time CSV |
| Practical to load & run locally | I `load_canonical_tokenizer`; E exp-0029; validator `gate_serialization_reload` |

## Experiment registry

| Requirement | Mapping |
|---|---|
| Continue existing registry from next available id | I `experiments/registry.yaml` exp-0024..exp-0030 (exp-0023 was the last Phase 2 entry) |
| Cover deterministic rebuild, round-trip/fallback, efficiency, baseline comparison, control/reserved, FLOWBOX perf, difficult examples | exp-0024 / 0025 / 0026 / 0027 / 0028 / 0029 / 0030 |
| Required provenance fields + actual results | I each entry has all 18 required fields; validator `gate_phase3_experiments_and_time`; T `test_experiment_registry` (existing) |
| Failed/negative results stay | governance rule 21; none for Phase 3 |

## Phase 3 validator

| Requirement | Mapping |
|---|---|
| `scripts/validate_phase3.py --all` following the Phase 1/2 pattern | I created |
| Includes/invokes prior-phase baselines + Phase 3 gates (regression, artifact presence, config, id/version, exact 36,864, special map, reserved range, byte fallback, deterministic round-trip smoke, model/tokenizer compat, serialization/reload, artifact hashes, corpus manifest/provenance, required experiments, required docs, required time rows, test-manifest consistency, full pytest, frozen architecture unchanged, approved tags resolvable, repo integrity) | I 17 gates in `validate_phase3.py` |
| CPU-safe; no NVIDIA GPU required for CI | I torch-free tokenizer path; CI `ubuntu-latest` |

## Test manifest & hashing

| Requirement | Mapping |
|---|---|
| `scripts/generate_phase3_test_manifest.py`, `manifests/phase-3-test-manifest.yaml`, `manifests/phase-3-artifact-hashes.yaml` | I all created |
| Explicit artifact/test lists in the hashing utility, not a broad glob | I `hashing.PHASE_3_HASHED_ARTIFACTS`, `PHASE_3_TEST_FILES` (explicit); corpus shards protected transitively via `corpus-manifest.json` hash + per-shard verification |
| Hash tokenizer artifacts, special-token map, merges/vocab, corpus manifest, impl, reports, experiments, tests, validation machinery | I `PHASE_3_HASHED_ARTIFACTS` list |
| Careful with earlier-phase manifests; follow immutable-tag/hash semantics; document regeneration | I phase-1 & phase-2 manifests regenerated for 3 shared evolving files; D hashing.py comment; validator `gate_earlier_manifests_current` |

## GitHub Actions

| Requirement | Mapping |
|---|---|
| `.github/workflows/phase-3-validation.yml` following the repaired Phase 2 pattern | I created |
| Full checkout history + fetch tags (do not reintroduce the shallow-checkout bug) | I `fetch-depth: 0`, `fetch-tags: true` with an explanatory comment |
| push/PR to `main` + manual dispatch; clean `ubuntu-latest`; Python 3.12; clean venv; locked install; `pip install -e . --no-deps`; run `validate_phase3.py --all` | I workflow steps |
| Phase 0/1/2 workflows still pass | phase-1/phase-2 manifests regenerated so their validators stay green; verified in CI |

## Documentation

| Requirement | Mapping |
|---|---|
| `phase-3-tokenizer.md`, `phase-3-requirements-traceability.md`, `phase-3-sonnet-self-review.md`, `docs/recovery/phase-3.md` | I all created |
| Detailed tokenizer design/report doc | I `docs/architecture/tokenizer-design.md` (17 sections covering objective, algorithm, impl/library decision, normalization, byte-level design, corpus, provenance, training method, vocab accounting, core ids, reserved ids, safety semantics, serialization, artifact layout, evaluation methodology, comparison methodology, performance, limitations, future compatibility) |
| ADRs for consequential decisions (impl/dependency, special-token/reserved layout, normalization/pre-tokenization, corpus/reproducibility) | I ADR-0010, ADR-0011, ADR-0012 (corpus reproducibility decision covered in ADR-0010 consequences + design §6) |
| Update ADR index | I `docs/adr/README.md` |
| README updated truthfully; no false capability claims | I `README.md` Phase 3 section |
| Phase report status `CANDIDATE - PENDING INDEPENDENT REVIEW`; reviewer identity Sonnet self / Sol pending; independent-review time zero/pending | I `phase-3-tokenizer.md`; time CSV `independent_review_hours: 0` |

## Completion-gate items

Every item in the master spec's §29 completion gate is covered by a row
above. The two `CANDIDATE`/no-tag items — status stays
`CANDIDATE - PENDING INDEPENDENT REVIEW`, and `phase-3-tokenizer` remains
uncreated — are enforced by `validate_phase3.gate_phase3_documentation`.
