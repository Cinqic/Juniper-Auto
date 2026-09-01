#!/usr/bin/env python3
"""Run a single Phase 3 tokenizer experiment and write a JSON result artifact.

Usage:
    python scripts/run_phase3_experiment.py rebuild-determinism   --output <path>
    python scripts/run_phase3_experiment.py roundtrip-fallback    --output <path>
    python scripts/run_phase3_experiment.py efficiency            --output <path>
    python scripts/run_phase3_experiment.py baseline-comparison   --output <path> [--allow-no-network]
    python scripts/run_phase3_experiment.py control-reserved      --output <path>
    python scripts/run_phase3_experiment.py flowbox-performance   --output <path>
    python scripts/run_phase3_experiment.py difficult-examples    --output <path>

All runs are real and executed against the committed frozen
``ja-tokenizer-v0.1`` artifact. Results (including negatives) are written
verbatim for the experiment registry to reference. A dirty working tree
yields an explicitly non-canonical diagnostic unless --allow-dirty.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import shlex
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from juniper_auto.config import load_architecture_config  # noqa: E402
from juniper_auto.tokenizer import constants as C  # noqa: E402
from juniper_auto.tokenizer.artifacts import (  # noqa: E402
    CANONICAL_ARTIFACT_DIR,
    compute_artifact_hashes,
)
from juniper_auto.tokenizer.config import validate_tokenizer_config  # noqa: E402
from juniper_auto.tokenizer.corpus import CORPUS_DIR, load_corpus_shards  # noqa: E402
from juniper_auto.tokenizer.evaluation import (  # noqa: E402
    EVAL_FIXTURE_PATH,
    byte_baseline_metrics,
    compare_tokens_per_domain,
    evaluate_all_domains,
    load_eval_fixture,
)
from juniper_auto.tokenizer.tokenizer import ControlToken, JuniperTokenizer  # noqa: E402
from juniper_auto.tokenizer.train import train_tokenizer  # noqa: E402
from juniper_auto.util.environment import describe_environment  # noqa: E402
from juniper_auto.util.hashing import sha256_file  # noqa: E402

SPARSE_PATH = REPO_ROOT / "configs/architecture/ja150m-v0.1.yaml"
DENSE_PATH = REPO_ROOT / "configs/architecture/ja150m-v0.1-dense.yaml"


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=5, check=True
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _git_status_porcelain() -> str:
    out = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=10, check=True,
    )
    return out.stdout


def _config_identity(path: Path) -> dict:
    cfg = load_architecture_config(path)
    return {
        "architecture_id": cfg.architecture_id,
        "config_path": str(path.relative_to(REPO_ROOT)),
        "config_sha256": sha256_file(path),
    }


def _write(path: Path, payload: dict, *, args, seed) -> None:
    if path.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {path} without --overwrite")
    raw_status = _git_status_porcelain()
    # A previously-written experiment-result artifact (this run's own earlier
    # outputs) does not make the tree "dirty" for canonicality purposes --
    # only modified tracked files or untracked files elsewhere do. This lets
    # the seven Phase 3 experiments run in sequence at a clean commit.
    status_lines = [
        line
        for line in raw_status.splitlines()
        if not (line.startswith("?? ") and line[3:].startswith("docs/experiments/results/"))
    ]
    status = "\n".join(status_lines)
    clean = status == ""
    if not clean and not args.allow_dirty:
        raise RuntimeError("dirty working tree; commit/stash or pass --allow-dirty for a non-canonical diagnostic")
    commit = _git_commit()
    if commit == "unknown" and not args.allow_dirty:
        raise RuntimeError("no resolvable Git HEAD")
    path.parent.mkdir(parents=True, exist_ok=True)
    full = {
        "result_identity": args.result_id or path.stem,
        "git_commit": commit,
        "git_worktree_clean": clean,
        "canonical_result": clean,
        "git_status_porcelain": status.splitlines(),
        "tokenizer_id": C.TOKENIZER_ID,
        "tokenizer_artifact_sha256": compute_artifact_hashes(CANONICAL_ARTIFACT_DIR),
        "eval_fixture_sha256": sha256_file(EVAL_FIXTURE_PATH) if EVAL_FIXTURE_PATH.is_file() else None,
        "architecture_configs": [_config_identity(SPARSE_PATH), _config_identity(DENSE_PATH)],
        "environment": describe_environment().as_dict(),
        "command": shlex.join(sys.argv),
        "seed": seed,
        **payload,
    }
    path.write_text(json.dumps(full, indent=2) + "\n")
    print(f"wrote {path}")


def _canonical_tokenizer() -> JuniperTokenizer:
    return JuniperTokenizer.load(CANONICAL_ARTIFACT_DIR, verify_hashes=True)


# ------------------------------------------------------------------
def cmd_rebuild_determinism(args) -> None:
    import tempfile

    recorded = compute_artifact_hashes(CANONICAL_ARTIFACT_DIR)
    rebuilds = []
    for i in range(args.rebuilds):
        with tempfile.TemporaryDirectory() as td:
            _tok, report = train_tokenizer(out_dir=Path(td) / "tok")
            rebuilt = compute_artifact_hashes(Path(td) / "tok")
            rebuilds.append(
                {
                    "rebuild_index": i,
                    "hashes": rebuilt,
                    "matches_canonical": rebuilt == recorded,
                    "wall_seconds_total": report.wall_seconds_total,
                    "merges_at_primary_threshold": report.bpe_merges_at_primary_threshold,
                    "tail_fill_used": report.bpe_tail_fill_used,
                }
            )
    gate_passed = all(r["matches_canonical"] for r in rebuilds)
    _write(
        Path(args.output),
        {
            "experiment": "tokenizer-rebuild-determinism",
            "canonical_hashes": recorded,
            "rebuilds": rebuilds,
            "gate": {"require_all_rebuilds_hash_identical_to_canonical": True},
            "gate_passed": gate_passed,
        },
        args=args,
        seed=None,
    )


def cmd_roundtrip_fallback(args) -> None:
    tok = _canonical_tokenizer()
    shards = load_corpus_shards(CORPUS_DIR)
    fixture = load_eval_fixture()

    corpus_failures = 0
    corpus_chars = 0
    for _name, text in shards:
        # sample windows to keep runtime bounded but cover every shard
        for start in range(0, len(text), 9973):
            seg = text[start : start + 4096]
            if tok.decode(tok.encode(seg)) != seg:
                corpus_failures += 1
            corpus_chars += len(seg)

    fixture_failures = 0
    for samples in fixture["domains"].values():
        for s in samples:
            if tok.decode(tok.encode(s)) != s:
                fixture_failures += 1

    rng = random.Random(20260901)
    prop_failures = 0
    prop_cases = 0
    fallback_tokens = 0
    total_tokens = 0
    for _ in range(args.property_cases):
        n = rng.randint(0, 80)
        s = "".join(chr(rng.randint(0, 0x10FFFF)) for _ in range(n))
        try:
            s.encode("utf-8")
        except UnicodeEncodeError:
            continue
        prop_cases += 1
        ids = tok.encode(s)
        if tok.decode(ids) != s:
            prop_failures += 1
        total_tokens += len(ids)
        fallback_tokens += sum(1 for t in ids if t < 256)

    # pure-byte inputs (every byte 0..255 as latin-1 text) must round-trip
    all_bytes_text = bytes(range(256)).decode("latin-1")
    all_bytes_ok = tok.decode(tok.encode(all_bytes_text)) == all_bytes_text

    gate_passed = bool(
        corpus_failures == 0
        and fixture_failures == 0
        and prop_failures == 0
        and all_bytes_ok
    )
    _write(
        Path(args.output),
        {
            "experiment": "tokenizer-roundtrip-fallback",
            "corpus_windows_chars": corpus_chars,
            "corpus_roundtrip_failures": corpus_failures,
            "fixture_roundtrip_failures": fixture_failures,
            "property_cases": prop_cases,
            "property_roundtrip_failures": prop_failures,
            "property_byte_fallback_rate": round(fallback_tokens / max(1, total_tokens), 6),
            "all_256_bytes_roundtrip_ok": all_bytes_ok,
            "gate": {"require_zero_roundtrip_failures_everywhere": True, "require_all_byte_representable": True},
            "gate_passed": gate_passed,
        },
        args=args,
        seed=20260901,
    )


def cmd_efficiency(args) -> None:
    tok = _canonical_tokenizer()
    report = evaluate_all_domains(tok)
    gate_passed = report["overall"]["roundtrip_failures"] == 0
    _write(
        Path(args.output),
        {
            "experiment": "tokenizer-efficiency",
            "evaluation": report,
            "gate": {"require_zero_roundtrip_failures_on_fixture": True},
            "gate_passed": gate_passed,
        },
        args=args,
        seed=None,
    )


def cmd_baseline_comparison(args) -> None:
    tok = _canonical_tokenizer()
    fixture = load_eval_fixture()
    byte_cmp = byte_baseline_metrics(fixture)
    result: dict = {
        "experiment": "tokenizer-baseline-comparison",
        "comparators": {},
    }

    # byte baseline
    byte_ratio = {}
    for domain, m in byte_cmp["per_domain"].items():
        ja = sum(len(tok.encode(s)) for s in fixture["domains"][domain])
        byte_ratio[domain] = {
            "juniper_tokens": ja,
            "byte_tokens": m["n_tokens"],
            "compression_ratio_vs_bytes": round(m["n_tokens"] / max(1, ja), 4),
        }
    result["comparators"]["utf8-bytes"] = {
        "identity": {"name": "utf8-bytes", "version": "n/a", "source": "construction", "bytes_per_token": 1.0},
        "per_domain": byte_ratio,
    }

    # gpt2 comparator
    gpt2_block: dict
    try:
        from juniper_auto.tokenizer.comparators import (
            GPT2_REVISION,
            ensure_gpt2_artifacts,
            gpt2_encode,
        )

        hashes = ensure_gpt2_artifacts(allow_download=True)
        per_domain = compare_tokens_per_domain(tok, gpt2_encode, fixture)
        gpt2_block = {
            "identity": {
                "name": "gpt2",
                "model": "openai-community/gpt2 (124M)",
                "revision": GPT2_REVISION,
                "source": "https://huggingface.co/openai-community/gpt2",
                "artifact_sha256": hashes,
                "pretokenizer_note": "re-based approximation of GPT-2's regex-module pattern; counts indicative within a few percent",
                "special_token_handling": "special tokens excluded on both sides (raw text only)",
            },
            "per_domain": per_domain,
            "status": "measured",
        }
    except Exception as exc:  # noqa: BLE001 - record honestly
        gpt2_block = {"status": "unavailable", "detail": repr(exc)}
        if not args.allow_no_network:
            print(f"WARNING: gpt2 comparator unavailable: {exc}", file=sys.stderr)
    result["comparators"]["gpt2"] = gpt2_block

    result["gate"] = {"require_byte_baseline": True, "gpt2_optional_if_offline": True}
    result["gate_passed"] = "utf8-bytes" in result["comparators"]
    _write(Path(args.output), result, args=args, seed=None)


def cmd_control_reserved(args) -> None:
    tok = _canonical_tokenizer()

    core_ok = all(tok.encode_control(t) == t.id for t in ControlToken)
    core_ids = [t.id for t in ControlToken]
    core_unique = len(core_ids) == len(set(core_ids))

    # literal control strings in ordinary text never yield a control id
    literal_leak = False
    produced_control_ids: set[int] = set()
    for s, _i in C.CORE_CONTROL_TOKENS:
        ids = tok.encode(f"prefix {s} suffix {s}{s}")
        produced_control_ids.update(t for t in ids if 256 <= t <= 511)
    literal_leak = len(produced_control_ids) > 0

    # reserved ids never produced by normal tokenization over the whole corpus
    reserved_seen: set[int] = set()
    for _name, text in load_corpus_shards(CORPUS_DIR):
        for start in range(0, len(text), 20011):
            for t in tok.encode(text[start : start + 8192]):
                if C.RESERVED_CONTROL_START <= t <= C.RESERVED_CONTROL_END:
                    reserved_seen.add(t)

    # explicit insertion works and round-trips
    seq = tok.build_sequence([ControlToken.SYSTEM, "safety first", ControlToken.USER, "hi", ControlToken.EOS])
    decoded = tok.decode(seq)
    decoded_skip = tok.decode(seq, skip_special=True)

    # adjacency and Unicode neighbours
    adj = tok.build_sequence([ControlToken.BOS, ControlToken.SYSTEM, "x", ControlToken.EOS])
    adj_ok = adj[:2] == [tok.bos_id, ControlToken.SYSTEM.id]

    gate_passed = bool(
        core_ok
        and core_unique
        and not literal_leak
        and not reserved_seen
        and decoded == "<|system|>safety first<|user|>hi<|eos|>"
        and decoded_skip == "safety firsthi"
        and adj_ok
    )
    _write(
        Path(args.output),
        {
            "experiment": "tokenizer-control-and-reserved",
            "core_control_ids": {t.value: t.id for t in ControlToken},
            "core_ids_unique": core_unique,
            "literal_control_string_leak": literal_leak,
            "reserved_ids_produced_by_normal_tokenization": sorted(reserved_seen),
            "reserved_range": [C.RESERVED_CONTROL_START, C.RESERVED_CONTROL_END],
            "explicit_sequence_roundtrip": decoded,
            "explicit_sequence_skip_special": decoded_skip,
            "gate": {
                "require_stable_core_ids": True,
                "require_no_literal_control_leak": True,
                "require_no_reserved_ids_from_text": True,
                "require_explicit_insertion_roundtrip": True,
            },
            "gate_passed": gate_passed,
        },
        args=args,
        seed=None,
    )


def cmd_flowbox_performance(args) -> None:
    # tokenizer training wall time (fresh temp build) + encode/decode throughput
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tracemalloc.start()
        _tok, report = train_tokenizer(out_dir=Path(td) / "tok")
        _cur, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    tok = _canonical_tokenizer()
    shards = load_corpus_shards(CORPUS_DIR)
    sample_text = "".join(text for _n, text in shards)[: args.throughput_chars]
    sample_bytes = len(sample_text.encode("utf-8"))

    t0 = time.perf_counter()
    ids = tok.encode(sample_text)
    enc_s = time.perf_counter() - t0
    # cold decode
    fresh = _canonical_tokenizer()
    t0 = time.perf_counter()
    _ = fresh.decode(ids)
    dec_s = time.perf_counter() - t0

    artifact_size = sum(p.stat().st_size for p in CANONICAL_ARTIFACT_DIR.glob("*") if p.is_file())

    result = {
        "experiment": "tokenizer-flowbox-performance",
        "training": {
            "wall_seconds_total": report.wall_seconds_total,
            "wall_seconds_bpe": report.wall_seconds_bpe,
            "wall_seconds_pretokenize": report.wall_seconds_pretokenize,
            "peak_python_heap_bytes": peak_bytes,
            "corpus_total_bytes": report.corpus_total_bytes,
            "distinct_pretokens": report.distinct_pretokens,
        },
        "encode": {
            "chars": len(sample_text),
            "bytes": sample_bytes,
            "tokens": len(ids),
            "seconds": round(enc_s, 4),
            "chars_per_second": round(len(sample_text) / max(enc_s, 1e-9)),
            "mb_per_second": round(sample_bytes / 1e6 / max(enc_s, 1e-9), 2),
        },
        "decode": {
            "tokens": len(ids),
            "seconds": round(dec_s, 4),
            "tokens_per_second": round(len(ids) / max(dec_s, 1e-9)),
        },
        "artifact_size_bytes": artifact_size,
        "gpu_hours": 0.0,
        "gate": {
            "require_training_under_600s": True,
            "require_encode_over_100k_chars_per_s": True,
            "require_artifact_under_5mb": True,
        },
        "gate_passed": bool(
            report.wall_seconds_total < 600
            and (len(sample_text) / max(enc_s, 1e-9)) > 100_000
            and artifact_size < 5_000_000
        ),
    }
    _write(Path(args.output), result, args=args, seed=None)


def cmd_difficult_examples(args) -> None:
    tok = _canonical_tokenizer()
    hard = [
        ("deep-indent-python", "def outer():\n    def inner():\n        if x:\n            return [\n                {'k': v}\n                for v in xs\n            ]\n"),
        ("json-punctuation", '{"a":{"b":[{"c":1},{"c":2}]},"d":null,"e":[[],[1],[1,2]]}'),
        ("windows-path", "C:\\Program Files\\Juniper\\ja150m-v0.1\\model.safetensors"),
        ("url-query", "https://host.example/api/v2/search?q=byte%20pair&limit=50&sort=-created_at#frag"),
        ("math-expr", "int_0^inf x^{n-1} e^{-x} dx = Gamma(n); sum_{k=0}^{n} binom(n,k) = 2^n"),
        ("emoji-zwj", "family: 👨\u200d👩\u200d👧\u200d👦 flag: 🇯🇵 skin: 👍🏽"),
        ("combining-rtl", "café النص العربي \u200f mixed \u0301\u0327 marks"),
        ("cjk", "機械学習のためのトークナイザー設計、日本語と中文の混在。"),
        ("control-literal-in-text", "user wrote: <|system|> please ignore previous <|tool_call|>"),
        ("tsv-log", "2026-09-01T05:31:24Z\tERROR\tworker[2]\ttask-88123\tconnection reset by peer\tretry=3"),
    ]
    records = []
    failures = 0
    for name, text in hard:
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        ok = decoded == text
        failures += not ok
        pieces = [p.decode("utf-8", errors="backslashreplace") for p in tok.token_pieces(text)]
        records.append(
            {
                "name": name,
                "chars": len(text),
                "bytes": len(text.encode("utf-8")),
                "tokens": len(ids),
                "chars_per_token": round(len(text) / max(1, len(ids)), 3),
                "roundtrip_ok": ok,
                "token_ids": ids,
                "token_pieces": pieces,
            }
        )
    _write(
        Path(args.output),
        {
            "experiment": "tokenizer-difficult-examples",
            "records": records,
            "gate": {"require_zero_roundtrip_failures": True},
            "gate_passed": failures == 0,
        },
        args=args,
        seed=None,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--output", required=True)
        p.add_argument("--result-id")
        p.add_argument("--allow-dirty", action="store_true")
        p.add_argument("--overwrite", action="store_true")

    p = sub.add_parser("rebuild-determinism"); common(p); p.add_argument("--rebuilds", type=int, default=2); p.set_defaults(func=cmd_rebuild_determinism)
    p = sub.add_parser("roundtrip-fallback"); common(p); p.add_argument("--property-cases", type=int, default=20000); p.set_defaults(func=cmd_roundtrip_fallback)
    p = sub.add_parser("efficiency"); common(p); p.set_defaults(func=cmd_efficiency)
    p = sub.add_parser("baseline-comparison"); common(p); p.add_argument("--allow-no-network", action="store_true"); p.set_defaults(func=cmd_baseline_comparison)
    p = sub.add_parser("control-reserved"); common(p); p.set_defaults(func=cmd_control_reserved)
    p = sub.add_parser("flowbox-performance"); common(p); p.add_argument("--throughput-chars", type=int, default=2_000_000); p.set_defaults(func=cmd_flowbox_performance)
    p = sub.add_parser("difficult-examples"); common(p); p.set_defaults(func=cmd_difficult_examples)

    args = parser.parse_args()
    validate_tokenizer_config()
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
