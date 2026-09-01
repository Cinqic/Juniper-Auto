"""Shared helpers for the Phase 3 tokenizer test suite."""

from __future__ import annotations

import functools

from juniper_auto.tokenizer import load_canonical_tokenizer
from juniper_auto.tokenizer.tokenizer import JuniperTokenizer


@functools.lru_cache(maxsize=1)
def canonical_tokenizer() -> JuniperTokenizer:
    return load_canonical_tokenizer(verify_hashes=True)


DOMAIN_SAMPLES: dict[str, list[str]] = {
    "empty": [""],
    "ascii": ["hello", "The quick brown fox.", "a b c 1 2 3"],
    "english": [
        "She said the meeting would run long, and it did.",
        "Reproducibility is a correctness property, not a nicety.",
    ],
    "whitespace": [
        " leading", "trailing ", "  two  spaces  ", "\ttab\tstart",
        "line1\nline2\n\n\nline5", "mixed \t \n \t end", "\r\n\r\ncrlf",
    ],
    "python": [
        "def f(x):\n    return x + 1\n",
        "class A:\n    def __init__(self):\n        self.xs = []\n\n    def add(self, x):\n        self.xs.append(x)\n",
        "if __name__ == '__main__':\n\tmain()\n",
    ],
    "nested_code": [
        "for i in range(10):\n    for j in range(10):\n        if i * j > 20:\n            grid[i][j] = {'k': [1, 2, 3]}\n",
    ],
    "json": [
        '{"a": 1, "b": [2, 3], "c": {"d": null, "e": true}}',
        '[{"id": 1}, {"id": 2, "tags": ["x", "y"]}]',
    ],
    "yaml": [
        "a:\n  b: 1\n  c:\n    - x\n    - y\nd: [1, 2, 3]\n",
    ],
    "xml_html": [
        '<div class="x" data-n="1"><span>text &amp; more</span></div>',
        "<?xml version=\"1.0\"?><root><child/></root>",
    ],
    "shell": [
        "$ ls -la | grep '\\.py$' | wc -l",
        "cd /tmp && rm -rf build/ && mkdir -p build && cd build",
    ],
    "paths": [
        "/usr/local/lib/python3.12/site-packages/pkg/mod.py",
        "C:\\Users\\dev\\AppData\\Local\\Temp\\x.txt",
        "../relative/../path/./here.md",
    ],
    "urls": [
        "https://example.com/a/b?c=1&d=2#frag",
        "postgres://u:p@h:5432/db?sslmode=require",
    ],
    "git": [
        "diff --git a/x.py b/x.py\n@@ -1,3 +1,4 @@\n-old\n+new\n",
        "commit 05fc185a573504fea4901845bc114d3fb79d8567\nAuthor: Cinqic\n",
    ],
    "logs": [
        "2026-09-01T05:31:24.551Z ERROR worker[2] task failed after 3 attempts",
    ],
    "math": [
        "e^{i*pi} + 1 = 0",
        "sum_{k=1}^{n} k = n(n+1)/2",
        "6.02214076e23, -273.15, 1.6e-19",
    ],
    "tool_traces": [
        '<|tool_call|> {"name": "read", "arguments": {"path": "x.py"}}',
        '<|tool_error|> {"type": "timeout", "after_ms": 5000}',
    ],
    "records": [
        '<|state|> {"step": 3, "cwd": "/home/x"}',
        '<|memory|> {"kind": "fact", "text": "hello"}',
    ],
    "unicode": [
        "café résumé naïve", "日本語 中文 한국어", "emoji 🦊🇯🇵👍🏽",
        "combining: e\u0301 a\u0327", "zero width\u200bjoin", "rtl العربية text",
        "math \u2211 \u222b \u2248", "\U0001F600\U0001F4A9",
    ],
}
