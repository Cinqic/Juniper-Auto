"""RNG state capture/restore and a deterministic, resumable synthetic
token-sequence stream used as Phase 1 engineering training data.

There is no production tokenizer or pretraining corpus yet (see
manifests/frozen-artifacts.yaml) -- `SyntheticSequenceStream` generates
fixed-vocabulary integer token id sequences from an explicit seed and is
labeled honestly as Phase 1 engineering data everywhere it is used
(checkpoint `dataset_identity`, experiment registry entries).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class RNGState:
    python_random: tuple
    numpy_state: tuple
    torch_cpu: torch.Tensor
    torch_cuda: list[torch.Tensor] | None  # one per visible CUDA device, or None if no CUDA


def capture_rng_state() -> RNGState:
    return RNGState(
        python_random=random.getstate(),
        numpy_state=np.random.get_state(),
        torch_cpu=torch.get_rng_state(),
        torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )


def restore_rng_state(state: RNGState) -> None:
    random.setstate(state.python_random)
    np.random.set_state(state.numpy_state)
    torch.set_rng_state(state.torch_cpu)
    if state.torch_cuda is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state.torch_cuda)


def rng_state_to_dict(state: RNGState) -> dict:
    return {
        "python_random": state.python_random,
        "numpy_state": state.numpy_state,
        "torch_cpu": state.torch_cpu,
        "torch_cuda": state.torch_cuda,
    }


def rng_state_from_dict(d: dict) -> RNGState:
    return RNGState(
        python_random=d["python_random"],
        numpy_state=d["numpy_state"],
        torch_cpu=d["torch_cpu"],
        torch_cuda=d["torch_cuda"],
    )


class SyntheticSequenceStream:
    """A fixed pool of `n_sequences` deterministic random token-id sequences
    (generated once from `seed`), served in shuffled batches with a
    separate, independently checkpointable shuffle-generator so that
    `next_batch()` after `load_state_dict(state_dict())` reproduces exactly
    the same batches an uninterrupted run would have produced next.

    This is the Phase 1 "sampler/data-stream" whose state must survive a
    checkpoint/resume cycle (see docs/phases/phase-1-architecture.md,
    interrupted/resumed equivalence).
    """

    def __init__(self, *, seed: int, vocab_size: int, seq_len: int, n_sequences: int, batch_size: int):
        if batch_size > n_sequences:
            raise ValueError("batch_size cannot exceed n_sequences for this reference stream")
        self.seed = seed
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_sequences = n_sequences
        self.batch_size = batch_size

        pool_generator = torch.Generator().manual_seed(seed)
        self.sequences = torch.randint(0, vocab_size, (n_sequences, seq_len), generator=pool_generator)

        self._epoch_generator = torch.Generator().manual_seed(seed + 1)
        self._order = torch.randperm(n_sequences, generator=self._epoch_generator)
        self._cursor = 0
        self.step = 0

    def next_batch(self) -> torch.Tensor:
        indices = []
        for _ in range(self.batch_size):
            if self._cursor >= self._order.numel():
                self._order = torch.randperm(self.n_sequences, generator=self._epoch_generator)
                self._cursor = 0
            indices.append(int(self._order[self._cursor].item()))
            self._cursor += 1
        self.step += 1
        return self.sequences[torch.tensor(indices)]

    def state_dict(self) -> dict:
        return {
            "seed": self.seed,
            "vocab_size": self.vocab_size,
            "seq_len": self.seq_len,
            "n_sequences": self.n_sequences,
            "batch_size": self.batch_size,
            "cursor": self._cursor,
            "step": self.step,
            "order": self._order.clone(),
            "epoch_generator_state": self._epoch_generator.get_state().clone(),
        }

    def load_state_dict(self, state: dict) -> None:
        if state["seed"] != self.seed or state["vocab_size"] != self.vocab_size or state["seq_len"] != self.seq_len:
            raise ValueError("checkpointed sampler state does not match this stream's construction parameters")
        self._cursor = state["cursor"]
        self.step = state["step"]
        self._order = state["order"].clone()
        self._epoch_generator.set_state(state["epoch_generator_state"].clone())
