"""``JuniperTokenizer`` -- the canonical ``ja-tokenizer-v0.1`` runtime object.

Loading contract:

* the frozen artifact directory is self-contained and loads with no network
  access;
* every id produced by :meth:`encode` is an ordinary byte / learned-merge id
  -- control-block ids (256..511) are ONLY ever produced by the explicit
  :meth:`build_sequence` / :meth:`encode_control` path (see the control-token
  safety contract in docs/architecture/tokenizer-design.md section 7);
* :meth:`decode` of :meth:`encode` output reproduces the input byte-for-byte
  for every valid ``str``.
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass
from pathlib import Path

from juniper_auto.tokenizer import constants as C
from juniper_auto.tokenizer.bpe import Pair, adjacent_pairs, apply_merges
from juniper_auto.tokenizer.bytelevel import (
    bytes_to_token_string,
    pretokenize,
    token_string_to_bytes,
)

# Exact sentinel first line of merges.txt. Parsing keys off equality with
# this string, never a "#" prefix -- a learned merge piece can legitimately
# start with "#" (byte 0x23).
MERGES_HEADER = "#ja-tokenizer-v0.1 byte-level BPE merges; rank == 0-based line index below this header"

ARTIFACT_FILES = (
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "special_tokens.json",
    "tokenizer_config.json",
)


class ControlToken(enum.Enum):
    """Deliberate protocol tokens. The only way to get a control-block id
    into a sequence is to name one of these (or pass a raw int id to
    :meth:`build_sequence`)."""

    BOS = "<|bos|>"
    EOS = "<|eos|>"
    PAD = "<|pad|>"
    SYSTEM = "<|system|>"
    USER = "<|user|>"
    ASSISTANT = "<|assistant|>"
    OBJECTIVE = "<|objective|>"
    STATE = "<|state|>"
    MEMORY = "<|memory|>"
    TOOL_CALL = "<|tool_call|>"
    TOOL_RESULT = "<|tool_result|>"
    TOOL_ERROR = "<|tool_error|>"
    OBSERVATION = "<|observation|>"
    ACTION = "<|action|>"
    FINAL = "<|final|>"

    @property
    def id(self) -> int:
        return C.CORE_CONTROL_STR_TO_ID[self.value]


class TokenizerArtifactError(RuntimeError):
    """Raised when a frozen artifact is missing, malformed, inconsistent, or
    fails a hard invariant. Always fails closed -- never silently repairs."""


@dataclass(frozen=True)
class ReservedControl:
    """A handle to a reserved-control id (activated by a future protocol
    version). Present so callers can reference reserved ids by index without
    the tokenizer pretending they carry semantics yet."""

    index: int

    @property
    def id(self) -> int:
        return C.RESERVED_CONTROL_START + self.index

    @property
    def surface(self) -> str:
        return C.reserved_control_token_str(self.index)


class JuniperTokenizer:
    def __init__(
        self,
        *,
        merges: list[Pair],
        metadata: dict,
        vocab: dict[bytes, int] | None = None,
    ) -> None:
        if len(merges) != C.NUM_MERGES:
            raise TokenizerArtifactError(
                f"expected exactly {C.NUM_MERGES} merges, got {len(merges)}"
            )
        self.merges: list[Pair] = list(merges)
        self.merge_ranks: dict[Pair, int] = {p: r for r, p in enumerate(self.merges)}
        self.metadata = dict(metadata)

        # id <-> bytes for the byte + learned-merge id space.
        self._id_to_bytes: dict[int, bytes] = {b: bytes([b]) for b in range(256)}
        self._bytes_to_id: dict[bytes, int] = {bytes([b]): b for b in range(256)}
        for rank, (a, b) in enumerate(self.merges):
            merged = a + b
            tid = C.LEARNED_VOCAB_START + rank
            if merged in self._bytes_to_id:
                raise TokenizerArtifactError(
                    f"merge {rank} produces byte string already owned by id "
                    f"{self._bytes_to_id[merged]} -- learned vocab must be collision free"
                )
            self._id_to_bytes[tid] = merged
            self._bytes_to_id[merged] = tid

        if vocab is not None and vocab != self._bytes_to_id:
            raise TokenizerArtifactError(
                "vocab.json is inconsistent with a merge replay of merges.txt"
            )

        self._special_id_to_str = dict(C.SPECIAL_ID_TO_STR)
        self._special_str_to_id = dict(C.SPECIAL_STR_TO_ID)
        self._encode_cache: dict[str, tuple[int, ...]] = {}

        self._validate_invariants()

    # -- invariants ----------------------------------------------------
    def _validate_invariants(self) -> None:
        total = len(self._id_to_bytes) + len(self._special_id_to_str)
        if total != C.VOCAB_SIZE:
            raise TokenizerArtifactError(
                f"vocabulary size is {total}, must be exactly {C.VOCAB_SIZE}"
            )
        all_ids = set(self._id_to_bytes) | set(self._special_id_to_str)
        if all_ids != set(range(C.VOCAB_SIZE)):
            missing = sorted(set(range(C.VOCAB_SIZE)) - all_ids)[:8]
            raise TokenizerArtifactError(f"id space has holes, e.g. {missing}")
        if set(self._id_to_bytes) & set(self._special_id_to_str):
            raise TokenizerArtifactError("byte/learned ids overlap special ids")
        for s, i in C.CORE_CONTROL_TOKENS:
            if self._special_id_to_str.get(i) != s:
                raise TokenizerArtifactError(f"core control token {s!r} lost id {i}")
        if len(C.RESERVED_CONTROL_TOKENS) != C.RESERVED_CONTROL_COUNT:
            raise TokenizerArtifactError("reserved-control range size drifted")

    # -- properties --------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return C.VOCAB_SIZE

    @property
    def tokenizer_id(self) -> str:
        return C.TOKENIZER_ID

    @property
    def bos_id(self) -> int:
        return ControlToken.BOS.id

    @property
    def eos_id(self) -> int:
        return ControlToken.EOS.id

    @property
    def pad_id(self) -> int:
        return ControlToken.PAD.id

    # -- ordinary (untrusted) text path ------------------------------
    def encode(
        self, text: str, *, add_bos: bool = False, add_eos: bool = False
    ) -> list[int]:
        """Encode ``text`` as ordinary, untrusted content.

        Control-block ids are never produced here, even if ``text`` contains
        the literal string ``<|system|>`` (those characters are encoded as
        ordinary bytes). Use :meth:`build_sequence` to insert protocol
        tokens deliberately.
        """
        if not isinstance(text, str):
            raise TypeError("encode() takes str; decode integer ids with decode()")
        out: list[int] = []
        if add_bos:
            out.append(self.bos_id)
        for chunk in pretokenize(text):
            cached = self._encode_cache.get(chunk)
            if cached is None:
                cached = self._encode_chunk(chunk)
                self._encode_cache[chunk] = cached
            out.extend(cached)
        if add_eos:
            out.append(self.eos_id)
        return out

    encode_ordinary = encode

    def _encode_chunk(self, chunk: str) -> tuple[int, ...]:
        raw = chunk.encode("utf-8")
        symbols: tuple[bytes, ...] = tuple(bytes([b]) for b in raw)
        merged = apply_merges(symbols, self.merge_ranks)
        return tuple(self._bytes_to_id[piece] for piece in merged)

    # -- deliberate protocol path -----------------------------------
    def encode_control(self, token: ControlToken | ReservedControl | str) -> int:
        if isinstance(token, ControlToken):
            return token.id
        if isinstance(token, ReservedControl):
            return token.id
        if isinstance(token, str):
            if token in self._special_str_to_id:
                return self._special_str_to_id[token]
            raise KeyError(f"{token!r} is not a registered special token")
        raise TypeError(f"unsupported control token: {token!r}")

    def build_sequence(
        self, parts: list[str | ControlToken | ReservedControl | int]
    ) -> list[int]:
        """Assemble a token id sequence from an explicit ordered mix of
        ordinary text (``str``), protocol tokens (``ControlToken`` /
        ``ReservedControl``), and raw ids (``int``)."""
        out: list[int] = []
        for part in parts:
            if isinstance(part, str):
                out.extend(self.encode(part))
            elif isinstance(part, (ControlToken, ReservedControl)):
                out.append(self.encode_control(part))
            elif isinstance(part, int):
                if not 0 <= part < C.VOCAB_SIZE:
                    raise ValueError(f"token id {part} out of range")
                out.append(part)
            else:
                raise TypeError(f"unsupported sequence part: {part!r}")
        return out

    # -- decode ------------------------------------------------------
    def decode(
        self,
        ids: list[int],
        *,
        skip_special: bool = False,
        errors: str = "strict",
    ) -> str:
        buf = bytearray()
        for tid in ids:
            if tid in self._special_id_to_str:
                if skip_special:
                    continue
                buf.extend(self._special_id_to_str[tid].encode("utf-8"))
            else:
                piece = self._id_to_bytes.get(tid)
                if piece is None:
                    raise TokenizerArtifactError(f"unknown token id {tid}")
                buf.extend(piece)
        return bytes(buf).decode("utf-8", errors=errors)

    def id_to_token_bytes(self, tid: int) -> bytes | None:
        return self._id_to_bytes.get(tid)

    def token_pieces(self, text: str) -> list[bytes]:
        """The byte pieces (not ids) ``encode`` would emit -- for manual
        inspection of fragmentation on difficult examples."""
        pieces: list[bytes] = []
        for chunk in pretokenize(text):
            raw = chunk.encode("utf-8")
            symbols = tuple(bytes([b]) for b in raw)
            pieces.extend(apply_merges(symbols, self.merge_ranks))
        return pieces

    def is_special_id(self, tid: int) -> bool:
        return tid in self._special_id_to_str

    # -- model compatibility ---------------------------------------
    def assert_model_compatible(self, arch_config) -> None:
        model_vocab = arch_config.embeddings.vocab_size
        if model_vocab != C.VOCAB_SIZE:
            raise TokenizerArtifactError(
                f"tokenizer vocab {C.VOCAB_SIZE} != model {arch_config.architecture_id} "
                f"vocab {model_vocab}"
            )

    # -- serialization --------------------------------------------
    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        vocab_out = {
            bytes_to_token_string(piece): tid
            for piece, tid in sorted(self._bytes_to_id.items(), key=lambda kv: kv[1])
        }
        (directory / "vocab.json").write_text(
            json.dumps(vocab_out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )

        merge_lines = [MERGES_HEADER]
        for a, b in self.merges:
            merge_lines.append(f"{bytes_to_token_string(a)} {bytes_to_token_string(b)}")
        (directory / "merges.txt").write_text("\n".join(merge_lines) + "\n", encoding="utf-8")

        (directory / "special_tokens.json").write_text(
            json.dumps(
                {
                    "core_control": {s: i for s, i in C.CORE_CONTROL_TOKENS},
                    "core_control_semantics": C.CORE_CONTROL_SEMANTICS,
                    "reserved_control_range": {
                        "start": C.RESERVED_CONTROL_START,
                        "end": C.RESERVED_CONTROL_END,
                        "count": C.RESERVED_CONTROL_COUNT,
                        "surface_pattern": "<|reserved_{index}|>",
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        (directory / "tokenizer_config.json").write_text(
            json.dumps(
                {
                    "tokenizer_id": C.TOKENIZER_ID,
                    "algorithm": C.ALGORITHM,
                    "normalization": "none",
                    "pretokenizer_pattern": C.PRETOKEN_PATTERN,
                    "byte_fallback": True,
                    "unk_token": None,
                    "add_bos_by_default": False,
                    "add_eos_by_default": False,
                    "bos_token": {"str": "<|bos|>", "id": ControlToken.BOS.id},
                    "eos_token": {"str": "<|eos|>", "id": ControlToken.EOS.id},
                    "pad_token": {"str": "<|pad|>", "id": ControlToken.PAD.id},
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        tok = {
            "tokenizer_id": C.TOKENIZER_ID,
            "algorithm": C.ALGORITHM,
            "vocab_size": C.VOCAB_SIZE,
            "layout": {
                "byte_tokens": [0, 255],
                "control_block": [C.CONTROL_BLOCK_START, C.CONTROL_BLOCK_END],
                "core_control": [256, 270],
                "reserved_control": [C.RESERVED_CONTROL_START, C.RESERVED_CONTROL_END],
                "learned_vocab": [C.LEARNED_VOCAB_START, C.LEARNED_VOCAB_END],
            },
            "merges_count": len(self.merges),
            "special_tokens": dict(C.SPECIAL_STR_TO_ID),
            "training": self.metadata,
        }
        (directory / "tokenizer.json").write_text(
            json.dumps(tok, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return directory

    @classmethod
    def load(cls, directory: str | Path, *, verify_hashes: bool = True) -> "JuniperTokenizer":
        directory = Path(directory)
        missing = [f for f in ARTIFACT_FILES if not (directory / f).is_file()]
        if missing:
            raise TokenizerArtifactError(f"frozen tokenizer artifact missing files: {missing}")

        if verify_hashes:
            from juniper_auto.tokenizer.artifacts import verify_artifact_hashes

            verify_artifact_hashes(directory)

        tok = json.loads((directory / "tokenizer.json").read_text(encoding="utf-8"))
        if tok.get("tokenizer_id") != C.TOKENIZER_ID:
            raise TokenizerArtifactError(
                f"tokenizer.json id {tok.get('tokenizer_id')!r} != expected {C.TOKENIZER_ID!r}"
            )
        if tok.get("vocab_size") != C.VOCAB_SIZE:
            raise TokenizerArtifactError("tokenizer.json vocab_size drifted")
        if tok.get("special_tokens") != dict(C.SPECIAL_STR_TO_ID):
            raise TokenizerArtifactError("tokenizer.json special-token map drifted")

        merges: list[Pair] = []
        raw_lines = (directory / "merges.txt").read_text(encoding="utf-8").split("\n")
        if not raw_lines or raw_lines[0] != MERGES_HEADER:
            raise TokenizerArtifactError("merges.txt is missing its exact header sentinel")
        for line in raw_lines[1:]:
            if line == "":
                continue
            parts = line.split(" ")
            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise TokenizerArtifactError(f"malformed merge line: {line!r}")
            merges.append((token_string_to_bytes(parts[0]), token_string_to_bytes(parts[1])))

        raw_vocab = json.loads((directory / "vocab.json").read_text(encoding="utf-8"))
        vocab = {token_string_to_bytes(k): v for k, v in raw_vocab.items()}

        return cls(merges=merges, metadata=tok.get("training", {}), vocab=vocab)
