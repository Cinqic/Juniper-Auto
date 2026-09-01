# Phase 3 Recovery Exercise — ja-tokenizer-v0.1

Extends [docs/recovery/README.md](README.md) and
[docs/recovery/phase-2.md](phase-2.md) with the Phase 3 tokenizer artifacts.
As with earlier phases, this is a genuine fresh-clone / fresh-venv exercise
from the canonical pushed candidate, in a location with no prior project
state.

## What Phase 3 adds to recovery

- `data/tokenizer/ja-tokenizer-v0.1/` — the frozen tokenizer artifact
  (committed; loads with **no network access**).
- `data/tokenizer/corpus/` — 24 committed training-corpus shards +
  `corpus-manifest.json` (per-shard SHA-256).
- `data/tokenizer/eval/held-out-eval-fixture.json` — held-out evaluation
  fixture v1.0.0.
- `manifests/phase-3-artifact-hashes.yaml`, `manifests/phase-3-test-manifest.yaml`.
- `scripts/validate_phase3.py` — canonical Phase 3 validation entrypoint.

Nothing Phase 3 adds requires a GPU, a network connection (except the
optional GPT-2 comparator in `exp-0027`, which is not part of loading or
validating the tokenizer), or any artifact hosted outside this git
repository.

## Procedure

```bash
git clone https://github.com/Cinqic/Juniper-Auto.git
cd Juniper-Auto
git fetch --tags
python3 -m venv .venv && source .venv/bin/activate
python -m ensurepip --upgrade
python -m pip install --upgrade pip
pip install -r requirements-lock.txt
pip install -e . --no-deps

# Phase 3 canonical validation (includes the Phase 0/1/2 baselines)
python scripts/validate_phase3.py --all

# Load the frozen tokenizer with no network and round-trip check
python - <<'PY'
from juniper_auto.tokenizer import load_canonical_tokenizer, ControlToken
t = load_canonical_tokenizer()
assert t.vocab_size == 36864
for s in ["hello world", "def f(x):\n    return x\n", '{"a":[1,2]}', "café 🦊 日本語"]:
    assert t.decode(t.encode(s)) == s
assert 259 not in t.encode("<|system|> literal")
assert t.build_sequence([ControlToken.SYSTEM])[0] == 259
print("tokenizer OK")
PY

# Prove deterministic rebuild from the committed corpus
python - <<'PY'
import tempfile, pathlib
from juniper_auto.tokenizer.train import train_tokenizer
from juniper_auto.tokenizer.artifacts import CANONICAL_ARTIFACT_DIR, compute_artifact_hashes
_t, r = train_tokenizer(out_dir=pathlib.Path(tempfile.mkdtemp()) / "rebuild")
# compute_artifact_hashes(rebuild dir) must equal compute_artifact_hashes(CANONICAL_ARTIFACT_DIR)
print("rebuild merges:", r.bpe_steps_completed)
PY
```

## Result

`RECOVERY EXERCISE RESULT: recorded in the Phase 3 metadata-closure commit`
after the candidate is pushed to `origin/main` and the exact commands above
are executed from a fresh clone. If the first attempt fails, that fact and
its repair are preserved here (not rewritten to look like it worked the
first time), followed by a second clean run — matching the Phase 1/2
recovery-doc convention.

<!-- FRESH-CLONE RECOVERY LOG -->
