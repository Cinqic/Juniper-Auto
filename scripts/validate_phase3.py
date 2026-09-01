#!/usr/bin/env python3
"""Single canonical Phase 3 validation entrypoint.

Usage:
    python scripts/validate_phase3.py --all

Runs every Phase 3 local validation gate in order, stopping at the first
failure, after first requiring the Phase 0/1/2 baseline
(scripts/validate_phase2.py --all) to pass. CPU-safe: no GPU is required
(the tokenizer is pure Python and torch-free; the CUDA-gated model tests in
the pytest suite run for real when CUDA is present and skip cleanly
otherwise).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


class GateFailure(Exception):
    pass


def _header(name: str) -> None:
    print(f"\n=== [{name}] ===")


def gate_phase2_baseline() -> None:
    _header("phase 2 baseline (includes phases 0 and 1)")
    result = subprocess.run([sys.executable, "scripts/validate_phase2.py", "--all"], cwd=REPO_ROOT)
    if result.returncode != 0:
        raise GateFailure(
            "scripts/validate_phase2.py --all did not pass -- Phase 3 cannot build on a failing baseline"
        )


def gate_phase3_imports() -> None:
    _header("phase 3 imports")
    import importlib

    for m in [
        "juniper_auto.tokenizer",
        "juniper_auto.tokenizer.constants",
        "juniper_auto.tokenizer.bytelevel",
        "juniper_auto.tokenizer.bpe",
        "juniper_auto.tokenizer.tokenizer",
        "juniper_auto.tokenizer.artifacts",
        "juniper_auto.tokenizer.corpus",
        "juniper_auto.tokenizer.train",
        "juniper_auto.tokenizer.evaluation",
        "juniper_auto.tokenizer.comparators",
        "juniper_auto.tokenizer.config",
    ]:
        importlib.import_module(m)
        print(f"import {m}: OK")


def gate_tokenizer_config() -> None:
    _header("tokenizer config validation")
    from juniper_auto.tokenizer.config import validate_tokenizer_config

    cfg = validate_tokenizer_config()
    print(f"configs/tokenizer/ja-tokenizer-v0.1.yaml valid ({cfg['tokenizer_id']}, {cfg['algorithm']})")


def gate_frozen_artifact_and_identity() -> None:
    _header("tokenizer artifact presence, id/version, hashes")
    from juniper_auto.tokenizer import constants as C
    from juniper_auto.tokenizer.artifacts import CANONICAL_ARTIFACT_DIR, verify_artifact_hashes
    from juniper_auto.tokenizer.tokenizer import ARTIFACT_FILES

    missing = [f for f in (*ARTIFACT_FILES, "hashes.json") if not (CANONICAL_ARTIFACT_DIR / f).is_file()]
    if missing:
        raise GateFailure(f"frozen tokenizer artifact missing files: {missing}")
    verify_artifact_hashes(CANONICAL_ARTIFACT_DIR)
    if C.TOKENIZER_ID != "ja-tokenizer-v0.1":
        raise GateFailure(f"tokenizer id drifted: {C.TOKENIZER_ID}")
    print(f"artifact present and hash-verified: {C.TOKENIZER_ID}")


def _load_tok():
    from juniper_auto.tokenizer.artifacts import load_canonical_tokenizer

    return load_canonical_tokenizer(verify_hashes=True)


def gate_vocab_invariants() -> None:
    _header("exact vocabulary size, control ids, reserved range")
    from juniper_auto.tokenizer import constants as C

    tok = _load_tok()
    if tok.vocab_size != 36_864:
        raise GateFailure(f"vocab size {tok.vocab_size} != 36864")
    ids = set(tok._id_to_bytes) | set(tok._special_id_to_str)
    if ids != set(range(36_864)):
        raise GateFailure("id space has holes or overflow")
    for s, i in C.CORE_CONTROL_TOKENS:
        if tok._special_id_to_str.get(i) != s:
            raise GateFailure(f"core control token {s} lost id {i}")
    reserved = [i for _, i in C.RESERVED_CONTROL_TOKENS]
    if reserved != list(range(271, 512)):
        raise GateFailure("reserved-control range drifted from [271, 511]")
    core = {i for _, i in C.CORE_CONTROL_TOKENS}
    if core & set(reserved):
        raise GateFailure("core and reserved control ids overlap")
    if len(tok.merges) != 36_352:
        raise GateFailure(f"learned merge count {len(tok.merges)} != 36352")
    print("vocab=36864, 15 core ids stable, reserved=[271,511], 36352 merges OK")


def gate_byte_fallback() -> None:
    _header("byte fallback: every byte representable, lossless")
    tok = _load_tok()
    s = bytes(range(256)).decode("latin-1")
    if tok.decode(tok.encode(s)) != s:
        raise GateFailure("byte fallback failed to round-trip all 256 byte values")
    print("all 256 byte values round-trip via byte fallback OK")


def gate_roundtrip_smoke() -> None:
    _header("deterministic round-trip smoke (representative domains)")
    tok = _load_tok()
    samples = [
        "", "Hello,  World!\n\tindent  ", "def f(x):\n    return {'k': [1, 2]}\n",
        '{"a": [1, 2], "b": null}', "https://ex.com/a?b=1&c=2", "/usr/bin/env python3",
        "C:\\Users\\x\\y.txt", "e^{i*pi} + 1 = 0", "café 🦊 日本語 العربية",
        "<|system|> literal", "a\u200bb\u0301 combining",
    ]
    for s in samples:
        if tok.decode(tok.encode(s)) != s:
            raise GateFailure(f"round-trip failed for {s!r}")
        if tok.encode(s) != tok.encode(s):
            raise GateFailure(f"non-deterministic encode for {s!r}")
    # control-block ids never emitted from ordinary text
    for s in samples:
        if any(256 <= i <= 511 for i in tok.encode(s)):
            raise GateFailure(f"control-block id leaked into ordinary text encoding of {s!r}")
    print(f"{len(samples)} domain samples round-trip exactly and deterministically OK")


def gate_serialization_reload() -> None:
    _header("tokenizer serialization / reload equivalence")
    import tempfile

    from juniper_auto.tokenizer.artifacts import write_hashes
    from juniper_auto.tokenizer.tokenizer import JuniperTokenizer

    tok = _load_tok()
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "tok"
        tok.save(d)
        write_hashes(d)
        reloaded = JuniperTokenizer.load(d)
        for s in ["hello world", "def f():\n    pass\n", '{"x":1}', "café 🦊"]:
            if reloaded.encode(s) != tok.encode(s):
                raise GateFailure("reloaded tokenizer encodes differently")
    print("save -> reload produces an identical encoder OK")


def gate_model_compatibility() -> None:
    _header("model / tokenizer vocab compatibility")
    from juniper_auto.config import load_architecture_config

    tok = _load_tok()
    for arch in ("ja150m-v0.1", "ja150m-v0.1-dense"):
        cfg = load_architecture_config(REPO_ROOT / "configs" / "architecture" / f"{arch}.yaml")
        tok.assert_model_compatible(cfg)
        if cfg.embeddings.vocab_size != 36_864:
            raise GateFailure(f"{arch} vocab_size != 36864")
    print("ja-tokenizer-v0.1.vocab_size == ja150m-v0.1 == ja150m-v0.1-dense == 36864 OK")


def gate_corpus_manifest_and_shards() -> None:
    _header("corpus manifest + per-shard hash verification + provenance")
    import json

    from juniper_auto.tokenizer.corpus import CORPUS_DIR, load_corpus_shards

    manifest = json.loads((CORPUS_DIR / "corpus-manifest.json").read_text())
    shards = load_corpus_shards(CORPUS_DIR)  # raises on any hash mismatch
    if len(shards) != manifest["shard_count"]:
        raise GateFailure("corpus shard count disagrees with the manifest")
    required_fields = {"shard", "category", "source", "license", "redistribution", "transformation", "bytes", "sha256"}
    for entry in manifest["shards"]:
        missing = required_fields - set(entry)
        if missing:
            raise GateFailure(f"corpus shard entry missing provenance fields: {missing}")
    if not any(e["category"].startswith("synthetic") for e in manifest["shards"]):
        raise GateFailure("corpus is missing labelled synthetic content")
    print(f"{len(shards)} corpus shards verified against manifest with full provenance OK")


def gate_frozen_architecture_unchanged() -> None:
    _header("frozen architecture and parameter counts unchanged")
    from juniper_auto.config import load_architecture_config
    from juniper_auto.config.frozen import FROZEN_STANDARD_ACTIVE_PARAMETERS, FROZEN_TOTAL_PARAMETERS
    from juniper_auto.model import build_model
    from juniper_auto.model.inspection import total_parameters
    from juniper_auto.accounting import standard_active_parameter_breakdown

    sparse_cfg = load_architecture_config(REPO_ROOT / "configs/architecture/ja150m-v0.1.yaml")
    dense_cfg = load_architecture_config(REPO_ROOT / "configs/architecture/ja150m-v0.1-dense.yaml")
    sparse = build_model(sparse_cfg, seed=0, device="cpu")
    dense = build_model(dense_cfg, seed=0, device="cpu")
    checks = [
        ("sparse total", total_parameters(sparse), 150_031_360),
        ("sparse standard active", standard_active_parameter_breakdown(sparse_cfg).total, 79_252_480),
        ("dense total", total_parameters(dense), 79_191_040),
    ]
    for label, actual, expected in checks:
        if actual != expected:
            raise GateFailure(f"{label}: {actual} != {expected} -- Phase 3 must not touch frozen architecture")
    if FROZEN_TOTAL_PARAMETERS["ja150m-v0.1"] != 150_031_360:
        raise GateFailure("frozen constant drift")
    print("frozen sparse/dense parameter counts unchanged OK")


def gate_approved_tags_resolvable() -> None:
    _header("approved Phase 0/1/2 tags still resolvable")
    # Every approved-phase tag must still dereference to a commit. Only
    # phase-1-architecture's commit is pinned here, because the Phase 2
    # baseline's golden moe.py comparison
    # (scripts/validate_phase2.py::gate_reference_backend_available) reads
    # that exact commit; phase-0/phase-2 just have to resolve.
    pinned = {"phase-1-architecture": "073acf46e04241ed35d00bc4b4c29ac463ee744d"}
    resolved = {}
    for tag in ("phase-0-foundation", "phase-1-architecture", "phase-2-moe"):
        out = subprocess.run(
            ["git", "rev-list", "-n", "1", tag], cwd=REPO_ROOT, capture_output=True, text=True
        )
        commit = out.stdout.strip()
        if out.returncode != 0 or not commit:
            raise GateFailure(f"approved tag {tag} is not resolvable")
        resolved[tag] = commit
        if tag in pinned and commit != pinned[tag]:
            raise GateFailure(f"approved tag {tag} moved: {commit} != {pinned[tag]}")
    print(
        "approved tags resolvable OK: "
        + ", ".join(f"{t}->{c[:10]}" for t, c in resolved.items())
    )


def gate_phase3_documentation() -> None:
    _header("phase 3 documentation")
    required = [
        "docs/phases/phase-3-tokenizer.md",
        "docs/phases/phase-3-requirements-traceability.md",
        "docs/phases/phase-3-sonnet-self-review.md",
        "docs/recovery/phase-3.md",
        "docs/architecture/tokenizer-design.md",
        "docs/adr/0010-tokenizer-implementation-choice.md",
        "docs/adr/0011-tokenizer-special-token-and-reserved-id-layout.md",
        "docs/adr/0012-tokenizer-normalization-and-pretokenization-policy.md",
    ]
    missing = [r for r in required if not (REPO_ROOT / r).is_file()]
    if missing:
        raise GateFailure(f"missing required Phase 3 documentation: {missing}")
    report = (REPO_ROOT / "docs/phases/phase-3-tokenizer.md").read_text()
    if "CANDIDATE - PENDING INDEPENDENT REVIEW" not in report:
        raise GateFailure("phase-3-tokenizer.md must remain CANDIDATE - PENDING INDEPENDENT REVIEW")
    if "phase-3-tokenizer" in _git_tags():
        raise GateFailure("phase-3-tokenizer approval tag must NOT exist during primary engineering")
    print(f"{len(required)} required Phase 3 docs present; status is candidate-pending; approval tag absent")


def _git_tags() -> str:
    return subprocess.run(["git", "tag", "-l"], cwd=REPO_ROOT, capture_output=True, text=True).stdout


def gate_phase3_experiments_and_time() -> None:
    _header("phase 3 experiment registry and time accounting")
    import csv

    import yaml

    entries = yaml.safe_load((REPO_ROOT / "experiments" / "registry.yaml").read_text())
    p3 = [e for e in entries if e.get("phase") == "phase-3"]
    if len(p3) < 7:
        raise GateFailure(f"expected >= 7 phase-3 experiment entries, found {len(p3)}")
    for e in p3:
        if e["tokenizer_id"] not in ("ja-tokenizer-v0.1",):
            raise GateFailure(f"{e['experiment_id']}: phase-3 experiments must reference tokenizer_id ja-tokenizer-v0.1")
    ids = [e["experiment_id"] for e in entries]
    if len(ids) != len(set(ids)):
        raise GateFailure("duplicate experiment ids")
    with (REPO_ROOT / "docs" / "time" / "phase-hours.csv").open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["phase"] == "phase-3"]
    if not rows:
        raise GateFailure("no phase-3 rows in docs/time/phase-hours.csv")
    for r in rows:
        if float(r["independent_review_hours"]) != 0.0:
            raise GateFailure("independent_review_hours must be 0 until Sol reviews")
    print(f"{len(p3)} phase-3 experiments, {len(rows)} phase-3 time rows OK")


def gate_phase3_artifact_hashes() -> None:
    _header("phase 3 artifact + test-manifest hashes")
    import yaml

    from juniper_auto.util.hashing import PHASE_3_HASHED_ARTIFACTS, PHASE_3_TEST_FILES, compute_hashes

    tm_path = REPO_ROOT / "manifests" / "phase-3-test-manifest.yaml"
    if not tm_path.is_file():
        raise GateFailure("missing manifests/phase-3-test-manifest.yaml")
    tm = yaml.safe_load(tm_path.read_text())
    if tm["sha256"] != compute_hashes(REPO_ROOT, PHASE_3_TEST_FILES):
        raise GateFailure("phase-3-test-manifest.yaml is stale relative to the test files")

    mpath = REPO_ROOT / "manifests" / "phase-3-artifact-hashes.yaml"
    if not mpath.is_file():
        raise GateFailure("missing manifests/phase-3-artifact-hashes.yaml")
    manifest = yaml.safe_load(mpath.read_text())
    recorded = manifest["sha256"]
    actual = compute_hashes(REPO_ROOT, PHASE_3_HASHED_ARTIFACTS)
    if set(recorded) != set(actual):
        raise GateFailure(
            "hash manifest list mismatch:\n"
            f"  manifest only: {sorted(set(recorded) - set(actual))}\n"
            f"  code only: {sorted(set(actual) - set(recorded))}"
        )
    stale = [k for k in recorded if recorded[k] != actual[k]]
    if stale:
        raise GateFailure(f"stale Phase 3 artifact hashes: {stale}")
    print(f"{len(actual)} Phase 3 artifact hashes + {len(tm['sha256'])} test-file hashes verified OK")


def gate_earlier_manifests_current() -> None:
    _header("earlier-phase manifests still current (globally-evolving files)")
    import yaml

    from juniper_auto.util.hashing import (
        PHASE_1_HASHED_ARTIFACTS,
        PHASE_2_HASHED_ARTIFACTS,
        compute_hashes,
    )

    for phase, artifacts in ((1, PHASE_1_HASHED_ARTIFACTS), (2, PHASE_2_HASHED_ARTIFACTS)):
        mpath = REPO_ROOT / "manifests" / f"phase-{phase}-artifact-hashes.yaml"
        recorded = yaml.safe_load(mpath.read_text())["sha256"]
        actual = compute_hashes(REPO_ROOT, artifacts)
        stale = [k for k in recorded if recorded.get(k) != actual.get(k)]
        if stale:
            raise GateFailure(
                f"phase-{phase}-artifact-hashes.yaml is stale after Phase 3 edits to shared files: {stale} "
                f"-- regenerate with: python scripts/hash_manifest.py --phase {phase}"
            )
    print("phase-1 and phase-2 artifact manifests are current OK")


def gate_pytest() -> None:
    _header("pytest suite (full)")
    result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"], cwd=REPO_ROOT)
    if result.returncode != 0:
        raise GateFailure(f"pytest exited with code {result.returncode}")


GATES = [
    ("phase 2 baseline", gate_phase2_baseline),
    ("phase 3 imports", gate_phase3_imports),
    ("tokenizer config", gate_tokenizer_config),
    ("frozen artifact + identity", gate_frozen_artifact_and_identity),
    ("vocab invariants", gate_vocab_invariants),
    ("byte fallback", gate_byte_fallback),
    ("round-trip smoke", gate_roundtrip_smoke),
    ("serialization/reload", gate_serialization_reload),
    ("model compatibility", gate_model_compatibility),
    ("corpus manifest + shards", gate_corpus_manifest_and_shards),
    ("frozen architecture unchanged", gate_frozen_architecture_unchanged),
    ("approved tags resolvable", gate_approved_tags_resolvable),
    ("phase 3 documentation", gate_phase3_documentation),
    ("phase 3 experiments + time", gate_phase3_experiments_and_time),
    ("phase 3 artifact hashes", gate_phase3_artifact_hashes),
    ("earlier manifests current", gate_earlier_manifests_current),
    ("pytest suite", gate_pytest),
]


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="run all Phase 3 validation gates")
    args = parser.parse_args()
    if not args.all:
        parser.print_help()
        print("\nNo gates selected. Pass --all to run the full Phase 3 validation suite.")
        return 2

    print(textwrap.dedent(f"""\
        Juniper Auto Phase 3 repository validation
        Repo root: {REPO_ROOT}
        Gates: {len(GATES)}
    """))

    for name, fn in GATES:
        try:
            fn()
        except GateFailure as e:
            print(f"\nFAILED at gate '{name}': {e}", file=sys.stderr)
            return 1
        except Exception as e:  # noqa: BLE001
            print(f"\nFAILED at gate '{name}' with unexpected error: {e!r}", file=sys.stderr)
            return 1

    print("\nAll Phase 3 validation gates passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
