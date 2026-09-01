"""Tokenizer-training corpus assembly with full provenance.

This builds the ``ja-tokenizer-v0.1`` *training* corpus -- deliberately not
the Phase 6 pretraining corpus. The canonical training input is the set of
committed shards under ``data/tokenizer/corpus/shards/`` plus
``data/tokenizer/corpus/corpus-manifest.json`` (per-shard SHA-256, byte
counts, category, source, license, transformation). Rebuilding the
tokenizer reads those committed shards, so the artifact is reproducible from
a fresh clone with no network access and independent of whether the
original sources are still reachable.

Every source is redistributable:

* ``repo-*``      -- this repository's own files (MIT, see LICENSE);
* ``stdlib-python`` -- CPython standard-library modules (PSF License 2.0),
  present in every Python 3.12 install; the manifest records module name +
  ``python_version`` + per-shard hash;
* ``synthetic-*`` -- project-authored, deterministically generated from the
  seeded templates in this module; labelled synthetic per governance
  rule 36.

Generic prose is intentionally capped so code / structured-data / tool-trace
content is not drowned out.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import sysconfig
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = REPO_ROOT / "data" / "tokenizer" / "corpus"
SHARDS_DIR = CORPUS_DIR / "shards"
MANIFEST_PATH = CORPUS_DIR / "corpus-manifest.json"

SHARD_MAX_BYTES = 480_000  # keep every committed shard well under 1 MB
CORPUS_SEED = 20260901


@dataclass
class SourceSpec:
    category: str
    source: str
    license: str
    redistribution: str
    transformation: str
    builder: str  # "files" or "synthetic"
    target_bytes: int
    file_globs: list[str] = field(default_factory=list)
    file_root: str = "repo"  # "repo" or "stdlib"
    exclude_globs: list[str] = field(default_factory=list)


# Files kept OUT of the training corpus:
#  - held-out evaluation material (would contaminate the eval);
#  - Phase 3 artifacts that reference the tokenizer's own frozen hashes or
#    are still being edited during Phase 3 engineering (avoids a circular
#    dependency between the corpus and files derived from it).
CORPUS_EXCLUDE_GLOBS = [
    "scripts/build_tokenizer_eval_fixture.py",
    "data/tokenizer/eval/*.json",
    "tests/tokenizer_fixtures.py",
    "tests/test_tokenizer_*.py",
    "configs/tokenizer/*.yaml",
    "docs/architecture/tokenizer-design.md",
    "docs/phases/phase-3-tokenizer.md",
    "docs/phases/phase-3-requirements-traceability.md",
    "docs/phases/phase-3-sonnet-self-review.md",
    "docs/recovery/phase-3.md",
    "docs/adr/0010-tokenizer-implementation-choice.md",
    "docs/adr/0011-tokenizer-special-token-and-reserved-id-layout.md",
    "docs/adr/0012-tokenizer-normalization-and-pretokenization-policy.md",
    "docs/experiments/results/exp-0024-tokenizer-rebuild-determinism.json",
    "docs/experiments/results/exp-0025-tokenizer-roundtrip-fallback.json",
    "docs/experiments/results/exp-0026-tokenizer-efficiency.json",
    "docs/experiments/results/exp-0027-tokenizer-baseline-comparison.json",
    "docs/experiments/results/exp-0028-tokenizer-control-and-reserved.json",
    "docs/experiments/results/exp-0029-tokenizer-flowbox-performance.json",
    "docs/experiments/results/exp-0030-tokenizer-difficult-examples.json",
]

# Machine-specific absolute home paths are redacted from every corpus source
# so the tokenizer never learns one user's directory layout and no committed
# shard trips the repository's absolute-path hygiene checks.
_HOME_REDACTIONS = {str(Path.home()): "/home/user"}


# ------------------------------------------------------------------
# File-derived sources
# ------------------------------------------------------------------
REPO_SOURCES: list[SourceSpec] = [
    SourceSpec(
        category="repo-python",
        source="Cinqic/Juniper-Auto @ phase-3 start (juniper_auto/, scripts/, tests/)",
        license="MIT",
        redistribution="permitted (repository ships MIT LICENSE)",
        transformation="whole-file concatenation, sorted by path, UTF-8, LF newlines",
        builder="files",
        target_bytes=600_000,
        file_globs=["juniper_auto/**/*.py", "scripts/*.py", "tests/*.py"],
    ),
    SourceSpec(
        category="repo-docs",
        source="Cinqic/Juniper-Auto @ phase-3 start (docs/, README.md)",
        license="MIT",
        redistribution="permitted (repository ships MIT LICENSE)",
        transformation="whole-file concatenation, sorted by path",
        builder="files",
        target_bytes=380_000,
        file_globs=["docs/**/*.md", "README.md"],
    ),
    SourceSpec(
        category="repo-config",
        source="Cinqic/Juniper-Auto @ phase-3 start (configs/, manifests/, experiments/, pyproject.toml)",
        license="MIT",
        redistribution="permitted (repository ships MIT LICENSE)",
        transformation="whole-file concatenation, sorted by path",
        builder="files",
        target_bytes=180_000,
        file_globs=[
            "configs/**/*.yaml",
            "manifests/*.yaml",
            "experiments/*.yaml",
            "pyproject.toml",
            "docs/experiments/results/*.json",
        ],
    ),
]

# CPython stdlib source: a fixed, sorted set of top-level modules plus whole
# packages chosen for syntactic diversity (argparse/typing/dataclasses/json/
# asyncio/email/http/xml/unittest/...). Globs are expanded against the
# running interpreter's stdlib path and then sorted; the Python version is
# recorded in the manifest. The committed shards are canonical -- a
# different 3.12.x patch level may produce slightly different regenerated
# stdlib shards, which is why the shards (not the sources) are the frozen
# training input.
STDLIB_MODULE_GLOBS = [
    "argparse.py", "ast.py", "base64.py", "bdb.py", "bisect.py", "calendar.py",
    "cmd.py", "code.py", "codeop.py", "configparser.py", "contextlib.py",
    "copy.py", "csv.py", "dataclasses.py", "datetime.py", "decimal.py",
    "difflib.py", "dis.py", "enum.py", "fnmatch.py", "fractions.py",
    "functools.py", "gettext.py", "graphlib.py", "gzip.py", "hashlib.py",
    "heapq.py", "inspect.py", "io.py", "keyword.py", "linecache.py",
    "mailbox.py", "numbers.py", "operator.py", "pathlib.py", "pdb.py",
    "pickle.py", "pickletools.py", "pprint.py", "profile.py", "pstats.py",
    "queue.py", "random.py", "sched.py", "secrets.py", "selectors.py",
    "shlex.py", "shutil.py", "smtplib.py", "socket.py", "socketserver.py",
    "ssl.py", "statistics.py", "string.py", "stringprep.py", "struct.py",
    "subprocess.py", "tarfile.py", "telnetlib.py", "textwrap.py",
    "threading.py", "tokenize.py", "trace.py", "traceback.py", "tracemalloc.py",
    "typing.py", "warnings.py", "weakref.py",
    "collections/*.py", "concurrent/futures/*.py", "email/*.py",
    "http/*.py", "importlib/*.py", "importlib/metadata/*.py", "json/*.py",
    "logging/*.py", "multiprocessing/*.py", "re/*.py", "unittest/*.py",
    "urllib/*.py", "wsgiref/*.py", "xml/*.py", "xml/etree/*.py",
    "xml/dom/*.py", "xml/sax/*.py", "asyncio/*.py",
]

STDLIB_SOURCE = SourceSpec(
    category="stdlib-python",
    source="CPython standard library modules + packages",
    license="PSF License Agreement 2.0",
    redistribution="permitted (PSF license; ships with every CPython install)",
    transformation="whole-file concatenation of a fixed sorted module/package set",
    builder="files",
    target_bytes=7_000_000,
    file_root="stdlib",
    file_globs=STDLIB_MODULE_GLOBS,
)


# ------------------------------------------------------------------
# Synthetic sources (project-authored, deterministic)
# ------------------------------------------------------------------
_JS_SNIPPETS = [
    """export function debounce(fn, waitMs) {
  let handle = null;
  return (...args) => {
    if (handle !== null) clearTimeout(handle);
    handle = setTimeout(() => fn.apply(this, args), waitMs);
  };
}
""",
    """const routes = {
  "/": renderHome,
  "/users/:id": renderUser,
  "/search": renderSearch,
};

async function dispatch(path) {
  const handler = routes[path] ?? renderNotFound;
  try {
    return await handler(parseQuery(location.search));
  } catch (err) {
    console.error(`dispatch failed for ${path}:`, err);
    return renderError(err);
  }
}
""",
    """class RingBuffer {
  #items = [];
  #capacity;
  constructor(capacity) { this.#capacity = capacity; }
  push(x) {
    this.#items.push(x);
    if (this.#items.length > this.#capacity) this.#items.shift();
  }
  get length() { return this.#items.length; }
  *[Symbol.iterator]() { yield* this.#items; }
}
""",
    """import { useState, useEffect } from "react";

export function useFetch(url) {
  const [state, setState] = useState({ loading: true, data: null, error: null });
  useEffect(() => {
    let cancelled = false;
    fetch(url)
      .then((r) => r.json())
      .then((data) => { if (!cancelled) setState({ loading: false, data, error: null }); })
      .catch((error) => { if (!cancelled) setState({ loading: false, data: null, error }); });
    return () => { cancelled = true; };
  }, [url]);
  return state;
}
""",
]

_C_SNIPPETS = [
    """#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct node { int key; struct node *next; } node_t;

static node_t *list_prepend(node_t *head, int key) {
    node_t *n = malloc(sizeof *n);
    if (!n) { perror("malloc"); exit(EXIT_FAILURE); }
    n->key = key;
    n->next = head;
    return n;
}

int main(int argc, char **argv) {
    node_t *head = NULL;
    for (int i = 1; i < argc; i++)
        head = list_prepend(head, atoi(argv[i]));
    for (node_t *p = head; p; p = p->next)
        printf("%d\\n", p->key);
    return 0;
}
""",
    """#include <stdint.h>

uint32_t fnv1a(const uint8_t *data, size_t len) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < len; i++) {
        h ^= data[i];
        h *= 16777619u;
    }
    return h;
}
""",
]

_CPP_SNIPPETS = [
    """#include <algorithm>
#include <vector>
#include <string>
#include <unordered_map>

template <typename T>
class Pool {
public:
    T& acquire() {
        if (free_.empty()) return storage_.emplace_back();
        T& slot = *free_.back();
        free_.pop_back();
        return slot;
    }
    void release(T& obj) { free_.push_back(&obj); }
private:
    std::vector<T> storage_;
    std::vector<T*> free_;
};

int count_words(const std::string& text) {
    std::unordered_map<std::string, int> freq;
    std::string token;
    for (char c : text) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!token.empty()) { ++freq[token]; token.clear(); }
        } else {
            token.push_back(c);
        }
    }
    if (!token.empty()) ++freq[token];
    return static_cast<int>(freq.size());
}
""",
]

_MATH_SNIPPETS = [
    "The gradient of f(x, y) = x^2 + 3xy - y^3 is grad f = (2x + 3y, 3x - 3y^2).",
    "By the binomial theorem, (a + b)^n = sum_{k=0}^{n} C(n, k) a^{n-k} b^k.",
    "Euler's identity: e^{i*pi} + 1 = 0. More generally e^{i*theta} = cos(theta) + i*sin(theta).",
    "The sample mean is x_bar = (1/n) * sum_{i=1}^{n} x_i and Var(X) = E[X^2] - (E[X])^2.",
    "lim_{x -> 0} (sin x) / x = 1, and d/dx [tan x] = sec^2 x = 1 + tan^2 x.",
    "Matrix product: (AB)_{ij} = sum_k A_{ik} B_{kj}; det(AB) = det(A) det(B).",
    "Bayes: P(H | E) = P(E | H) P(H) / P(E), with P(E) = sum_h P(E | h) P(h).",
    "Big-O: 3n^3 + 12n^2 log n + 7 = O(n^3); the recurrence T(n) = 2 T(n/2) + O(n) gives T(n) = O(n log n).",
    "Scientific notation: 6.02214076e23 mol^-1, 1.602176634e-19 C, 2.99792458e8 m/s, -273.15 degC.",
    "Integral: int_0^1 x^2 dx = 1/3; int_{-inf}^{inf} e^{-x^2} dx = sqrt(pi).",
    "Vector norm ||v||_2 = sqrt(v_1^2 + ... + v_n^2); cosine similarity = (u . v) / (||u|| ||v||).",
    "Softmax_i(z) = exp(z_i) / sum_j exp(z_j); cross-entropy H(p, q) = - sum_i p_i log q_i.",
]

_SHELL_SNIPPETS = [
    """$ git status
On branch main
Your branch is ahead of 'origin/main' by 2 commits.
  (use "git push" to publish your local commits)

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
        modified:   juniper_auto/tokenizer/tokenizer.py
        modified:   tests/test_tokenizer_roundtrip.py

no changes added to commit (use "git add" and/or "git commit -a")
""",
    """$ ls -la /var/log
total 1284
drwxr-xr-x  12 root   root    4096 Sep  1 03:14 .
drwxr-xr-x  14 root   root    4096 Aug 20 11:02 ..
-rw-r-----   1 syslog adm   145238 Sep  1 05:31 auth.log
-rw-r--r--   1 root   root       0 Sep  1 00:00 boot.log
-rw-r-----   1 syslog adm    88213 Sep  1 05:30 syslog
""",
    """$ python -m pytest tests/ -q
........................................................................ [ 41%]
........................................................................ [ 83%]
..............................                                           [100%]
102 passed in 3.55s
""",
    """$ curl -sS https://api.example.com/v1/status | jq '.services[]'
{
  "name": "gateway",
  "healthy": true,
  "latency_ms": 12.4
}
{
  "name": "worker",
  "healthy": false,
  "latency_ms": null
}
""",
    """$ df -h
Filesystem      Size  Used Avail Use% Mounted on
/dev/nvme0n1p2  234G  148G   74G  67% /
tmpfs           7.8G  1.2M  7.8G   1% /run
$ uname -srm
Linux 7.0.0-30-generic x86_64
""",
    """2026-09-01T05:31:22.481Z INFO  scheduler        starting run id=run-3931 workers=4
2026-09-01T05:31:22.902Z DEBUG queue            dequeued task task-88123 attempt=1
2026-09-01T05:31:23.114Z WARN  worker[2]        retryable error: connection reset by peer
2026-09-01T05:31:24.550Z ERROR worker[2]        task task-88123 failed after 3 attempts
2026-09-01T05:31:24.551Z INFO  scheduler        run id=run-3931 finished ok=120 failed=1
""",
]

_PATH_URL_SNIPPETS = [
    "See /usr/local/lib/python3.12/site-packages/juniper_auto/tokenizer/tokenizer.py and ~/.config/juniper/settings.toml.",
    "Windows path: C:\\Users\\dev\\Documents\\Juniper Auto\\data\\tokenizer\\ja-tokenizer-v0.1\\vocab.json",
    "Relative import target: ../../juniper_auto/model/moe.py; build output at ./dist/juniper_auto-0.0.1-py3-none-any.whl",
    "URL: https://github.com/Cinqic/Juniper-Auto/blob/main/docs/architecture/tokenizer-design.md#section-7",
    "https://raw.githubusercontent.com/Cinqic/Juniper-Auto/05fc185/README.md?token=abc123&ref=main",
    "mailto:maintainers@juniper.example; ssh://git@github.com:22/Cinqic/Juniper-Auto.git",
    "file:///home/user/Documents/Juniper%20Auto/manifests/phase-3-artifact-hashes.yaml",
    "s3://juniper-artifacts/tokenizer/ja-tokenizer-v0.1/merges.txt  gs://juniper-backups/2026-09-01/",
    "Endpoint: https://api.example.com:8443/v2/models/ja150m-v0.1/generate?stream=true&max_tokens=256",
]

_STRUCTURED_SNIPPETS = [
    """{
  "objective_id": "obj-0007",
  "status": "in_progress",
  "steps": [
    {"id": 1, "action": "read_file", "path": "docs/plan.md", "done": true},
    {"id": 2, "action": "summarize", "done": false, "notes": "waiting on step 1"}
  ],
  "deadline": null,
  "tags": ["research", "phase-3"]
}
""",
    """services:
  gateway:
    image: registry.example.com/gateway:1.8.2
    ports:
      - "8080:8080"
    environment:
      LOG_LEVEL: info
      TIMEOUT_MS: "2500"
    depends_on: [redis, worker]
  worker:
    build: ./worker
    deploy:
      replicas: 3
      resources:
        limits: { cpus: "0.50", memory: 256M }
""",
    """<article class="post" data-id="88123">
  <h1>Tokenizer Notes</h1>
  <p>Round-trip must be <em>exact</em> for &lt;code&gt; &amp; whitespace.</p>
  <ul>
    <li>tabs\tpreserved</li>
    <li>trailing spaces preserved   </li>
  </ul>
</article>
""",
    """<?xml version="1.0" encoding="UTF-8"?>
<config xmlns:j="urn:juniper:tokenizer">
  <j:vocab size="36864" algorithm="utf8-byte-level-bpe"/>
  <j:special start="256" end="511"/>
</config>
""",
]

_TOOL_TRACE_SNIPPETS = [
    """<|objective|> {"id": "obj-12", "goal": "verify tokenizer round-trips on the eval fixture", "budget_steps": 6}
<|state|> {"step": 2, "open_files": ["juniper_auto/tokenizer/evaluation.py"], "last_error": null}
<|assistant|> I will run the evaluation harness on the held-out fixture.
<|tool_call|> {"name": "run_python", "arguments": {"module": "juniper_auto.tokenizer.evaluation", "args": ["--fixture", "held-out"]}}
<|tool_result|> {"exit_code": 0, "stdout": "chars_per_token=3.91 bytes_per_token=4.12 roundtrip_failures=0"}
<|assistant|> Round-trip failures are zero; efficiency is within the declared gate.
<|final|> The held-out fixture round-trips exactly at 3.91 characters/token.
""",
    """<|user|> Check whether the config file parses.
<|tool_call|> {"name": "read_file", "arguments": {"path": "configs/tokenizer/ja-tokenizer-v0.1.yaml"}}
<|tool_result|> {"bytes": 1843, "sha256": "b1a0...c4"}
<|tool_call|> {"name": "validate_yaml", "arguments": {"path": "configs/tokenizer/ja-tokenizer-v0.1.yaml"}}
<|tool_error|> {"error": "yaml.scanner.ScannerError", "message": "mapping values are not allowed here", "line": 12, "column": 9}
<|assistant|> The YAML has a syntax error on line 12; I will fix the indentation and re-validate.
""",
    """<|memory|> {"kind": "preference", "key": "review_style", "value": "adversarial", "confidence": 0.9, "source": "obj-04"}
<|observation|> exit_code=1 stderr="ModuleNotFoundError: No module named 'juniper_auto.tokenizer'"
<|action|> {"type": "install", "command": "pip install -e . --no-deps", "reversible": true}
<|observation|> exit_code=0 stdout="Successfully installed juniper-auto-0.0.1"
""",
]

_PROSE_SNIPPETS = [
    "A tokenizer is the boundary between human-readable text and the integer sequences a language model actually consumes. Its job is unglamorous but load-bearing: every downstream metric inherits whatever fragmentation, ambiguity, or information loss the tokenizer introduces.",
    "Byte-level fallback matters because natural inputs are messier than any training corpus. Filenames carry accents, logs carry control characters, and pasted code carries whatever encoding its author used. A tokenizer that leans on an unknown-token to absorb the unexpected is quietly lossy in exactly the situations where fidelity matters most.",
    "Determinism is a correctness property, not a nicety. If retraining the tokenizer on the same corpus and configuration produced a different vocabulary, then every hash, every frozen artifact, and every reproducibility claim downstream would rest on sand.",
    "Technical writing rewards precision over flourish. State the invariant, state how it is checked, and link to the check. Prefer a sentence that a reader can falsify to a paragraph that merely sounds authoritative.",
    "Educational material works best when it separates the mechanism from the motivation. Explain what the code does, then explain why anyone should care, and keep the two threads visibly distinct so a reader can follow either one.",
    "Structured data formats each make different trade-offs. JSON is ubiquitous and unforgiving about trailing commas; YAML is friendlier to write and treacherous about whitespace; XML is verbose but self-describing. A single tokenizer has to represent all three without turning their punctuation into confetti.",
    "The phrase 'works on my machine' is a bug report, not a defense. Reproducibility means a stranger with a clean checkout and the documented environment reaches the same result, and the only way to know that is to actually try it on a clean checkout.",
]


def _iter_files(spec: SourceSpec) -> list[Path]:
    if spec.file_root == "stdlib":
        root = Path(sysconfig.get_paths()["stdlib"])
        paths: list[Path] = []
        for g in spec.file_globs:
            if any(ch in g for ch in "*?["):
                paths.extend(root.glob(g))
            else:
                paths.append(root / g)
        return sorted({p for p in paths if p.is_file()})
    paths: list[Path] = []
    for g in spec.file_globs:
        paths.extend(sorted(REPO_ROOT.glob(g)))
    excluded: set[Path] = set()
    for g in list(spec.exclude_globs) + CORPUS_EXCLUDE_GLOBS:
        excluded.update(REPO_ROOT.glob(g))
    # Deduplicate + stable sort by repo-relative path.
    seen: set[Path] = set()
    out: list[Path] = []
    for p in sorted(set(paths)):
        if p.is_file() and p not in seen and p not in excluded:
            seen.add(p)
            out.append(p)
    return out


def _build_file_source(spec: SourceSpec) -> tuple[str, list[str]]:
    parts: list[str] = []
    used: list[str] = []
    total = 0
    files = _iter_files(spec)
    for p in files:
        if total >= spec.target_bytes:
            break
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if spec.file_root == "stdlib":
            label = f"# ---- CPython stdlib: {p.name} ----\n"
            used.append(str(p.relative_to(Path(sysconfig.get_paths()["stdlib"]).parent)))
        else:
            rel = p.relative_to(REPO_ROOT).as_posix()
            label = f"# ---- {rel} ----\n"
            used.append(rel)
        parts.append(label + text.rstrip("\n") + "\n\n")
        total += len(label) + len(text)
    return "".join(parts), used


def _synthetic_text(category: str, target_bytes: int, rng: random.Random) -> str:
    banks = {
        "synthetic-javascript": _JS_SNIPPETS,
        "synthetic-c": _C_SNIPPETS,
        "synthetic-cpp": _CPP_SNIPPETS,
        "synthetic-math": _MATH_SNIPPETS,
        "synthetic-shell-logs": _SHELL_SNIPPETS,
        "synthetic-paths-urls": _PATH_URL_SNIPPETS,
        "synthetic-structured": _STRUCTURED_SNIPPETS,
        "synthetic-tool-traces": _TOOL_TRACE_SNIPPETS,
        "synthetic-prose": _PROSE_SNIPPETS,
    }
    bank = banks[category]
    out: list[str] = []
    total = 0
    # Deterministic: a fixed number of shuffled passes over the bank.
    order = list(range(len(bank)))
    while total < target_bytes:
        rng.shuffle(order)
        for idx in order:
            piece = bank[idx]
            if category in ("synthetic-math", "synthetic-paths-urls"):
                piece = piece + "\n"
            out.append(piece)
            total += len(piece.encode("utf-8"))
            if total >= target_bytes:
                break
    return "".join(out)


SYNTHETIC_SOURCES: list[SourceSpec] = [
    SourceSpec("synthetic-javascript", "project-authored synthetic JavaScript", "MIT",
               "permitted (project-authored)", "deterministic seeded expansion of embedded templates",
               "synthetic", 330_000),
    SourceSpec("synthetic-c", "project-authored synthetic C", "MIT",
               "permitted (project-authored)", "deterministic seeded expansion of embedded templates",
               "synthetic", 210_000),
    SourceSpec("synthetic-cpp", "project-authored synthetic C++", "MIT",
               "permitted (project-authored)", "deterministic seeded expansion of embedded templates",
               "synthetic", 210_000),
    SourceSpec("synthetic-math", "project-authored synthetic mathematical / numeric notation", "MIT",
               "permitted (project-authored)", "deterministic seeded expansion of embedded templates",
               "synthetic", 300_000),
    SourceSpec("synthetic-shell-logs", "project-authored synthetic shell sessions / terminal output / logs / git output", "MIT",
               "permitted (project-authored)", "deterministic seeded expansion of embedded templates",
               "synthetic", 390_000),
    SourceSpec("synthetic-paths-urls", "project-authored synthetic filesystem paths and URLs", "MIT",
               "permitted (project-authored)", "deterministic seeded expansion of embedded templates",
               "synthetic", 225_000),
    SourceSpec("synthetic-structured", "project-authored synthetic JSON / YAML / XML / HTML", "MIT",
               "permitted (project-authored)", "deterministic seeded expansion of embedded templates",
               "synthetic", 330_000),
    SourceSpec("synthetic-tool-traces", "project-authored synthetic tool-call / result / error / objective / state / memory / observation / action records", "MIT",
               "permitted (project-authored)", "deterministic seeded expansion of embedded templates",
               "synthetic", 330_000),
    SourceSpec("synthetic-prose", "project-authored synthetic general / technical / educational prose", "MIT",
               "permitted (project-authored)", "deterministic seeded expansion of embedded templates",
               "synthetic", 360_000),
]

ALL_SOURCES: list[SourceSpec] = REPO_SOURCES + [STDLIB_SOURCE] + SYNTHETIC_SOURCES


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_corpus(*, seed: int = CORPUS_SEED, out_dir: Path = CORPUS_DIR) -> dict:
    """(Re)build the committed corpus shards + manifest. Deterministic given
    ``seed``, the repository content, and the running Python's stdlib."""
    out_dir = Path(out_dir)
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    for old in shards_dir.glob("*.txt"):
        old.unlink()

    rng = random.Random(seed)
    manifest_entries: list[dict] = []
    shard_index = 0
    total_bytes = 0

    for spec in ALL_SOURCES:
        if spec.builder == "files":
            text, used = _build_file_source(spec)
            source_detail = {"files": used, "file_count": len(used)}
            if spec.file_root == "stdlib":
                source_detail["python_version"] = sys.version.split()[0]
                source_detail["stdlib_relative_root"] = "lib/python3.12 of the running interpreter"
        else:
            text = _synthetic_text(spec.category, spec.target_bytes, rng)
            source_detail = {"generator": "juniper_auto.tokenizer.corpus._synthetic_text", "seed": seed}

        for needle, replacement in _HOME_REDACTIONS.items():
            text = text.replace(needle, replacement)
        data = text.encode("utf-8")
        # split into <SHARD_MAX_BYTES shards on a UTF-8-safe boundary
        offset = 0
        while offset < len(data):
            end = min(offset + SHARD_MAX_BYTES, len(data))
            while end < len(data) and (data[end] & 0xC0) == 0x80:
                end -= 1
            chunk = data[offset:end]
            shard_index += 1
            name = f"{shard_index:04d}-{spec.category}.txt"
            (shards_dir / name).write_bytes(chunk)
            manifest_entries.append(
                {
                    "shard": f"shards/{name}",
                    "category": spec.category,
                    "source": spec.source,
                    "license": spec.license,
                    "redistribution": spec.redistribution,
                    "transformation": spec.transformation,
                    "builder": spec.builder,
                    "bytes": len(chunk),
                    "sha256": _sha256_bytes(chunk),
                    "source_detail": source_detail,
                }
            )
            total_bytes += len(chunk)
            offset = end

    manifest = {
        "corpus_id": "ja-tokenizer-v0.1-training-corpus",
        "purpose": "tokenizer BPE training only (NOT the Phase 6 pretraining corpus)",
        "global_transformations": [
            "UTF-8, LF newlines",
            "the building machine's absolute home path is redacted to /home/user in every shard",
        ],
        "seed": seed,
        "shard_count": len(manifest_entries),
        "total_bytes": total_bytes,
        "categories": sorted({e["category"] for e in manifest_entries}),
        "shards": manifest_entries,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def corpus_manifest_sha256() -> str:
    return _sha256_bytes(MANIFEST_PATH.read_bytes())


def load_corpus_shards(out_dir: Path = CORPUS_DIR) -> list[tuple[str, str]]:
    """Return ``[(shard_name, text)]`` in canonical (manifest) order, after
    verifying every shard hash. Raises on any mismatch (fail loud)."""
    out_dir = Path(out_dir)
    manifest = json.loads((out_dir / "corpus-manifest.json").read_text(encoding="utf-8"))
    shards: list[tuple[str, str]] = []
    for entry in manifest["shards"]:
        p = out_dir / entry["shard"]
        raw = p.read_bytes()
        if _sha256_bytes(raw) != entry["sha256"]:
            raise RuntimeError(f"corpus shard hash mismatch: {entry['shard']}")
        shards.append((entry["shard"], raw.decode("utf-8")))
    return shards
