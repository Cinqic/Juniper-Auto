"""``ja-tokenizer-v0.1`` -- the one unified Juniper Auto tokenizer.

UTF-8 byte-level BPE, exactly 36,864 ids, byte fallback (no ``<unk>``),
identity normalization, an explicit frozen control-token map, and a
contiguous reserved future-control range. See
docs/architecture/tokenizer-design.md.
"""

from juniper_auto.tokenizer.constants import (
    ALGORITHM,
    CORE_CONTROL_TOKENS,
    NUM_MERGES,
    RESERVED_CONTROL_COUNT,
    RESERVED_CONTROL_END,
    RESERVED_CONTROL_START,
    TOKENIZER_ID,
    VOCAB_SIZE,
)
from juniper_auto.tokenizer.tokenizer import (
    ControlToken,
    JuniperTokenizer,
    ReservedControl,
    TokenizerArtifactError,
)

__all__ = [
    "ALGORITHM",
    "CORE_CONTROL_TOKENS",
    "ControlToken",
    "JuniperTokenizer",
    "NUM_MERGES",
    "RESERVED_CONTROL_COUNT",
    "RESERVED_CONTROL_END",
    "RESERVED_CONTROL_START",
    "ReservedControl",
    "TOKENIZER_ID",
    "TokenizerArtifactError",
    "VOCAB_SIZE",
    "load_canonical_tokenizer",
]


def load_canonical_tokenizer(*, verify_hashes: bool = True) -> JuniperTokenizer:
    """Load the committed canonical tokenizer artifact (no network access)."""
    from juniper_auto.tokenizer.artifacts import load_canonical_tokenizer as _load

    return _load(verify_hashes=verify_hashes)
