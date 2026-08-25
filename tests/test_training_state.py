"""RNG state capture/restore and the synthetic sequence stream's
determinism and checkpoint-resume behavior."""

from __future__ import annotations

import random

import numpy as np
import torch

from juniper_auto.training.state import (
    SyntheticSequenceStream,
    capture_rng_state,
    restore_rng_state,
    rng_state_from_dict,
    rng_state_to_dict,
)


def test_rng_state_round_trip_reproduces_subsequent_draws():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    random.random()
    np.random.rand()
    torch.rand(3)

    state = capture_rng_state()
    expected_python = [random.random() for _ in range(3)]
    expected_numpy = np.random.rand(3).tolist()
    expected_torch = torch.rand(3).clone()

    restore_rng_state(state)
    actual_python = [random.random() for _ in range(3)]
    actual_numpy = np.random.rand(3).tolist()
    actual_torch = torch.rand(3)

    assert actual_python == expected_python
    assert actual_numpy == expected_numpy
    assert torch.equal(actual_torch, expected_torch)


def test_rng_state_dict_round_trip():
    torch.manual_seed(0)
    state = capture_rng_state()
    d = rng_state_to_dict(state)
    restored = rng_state_from_dict(d)
    assert torch.equal(state.torch_cpu, restored.torch_cpu)
    assert state.python_random == restored.python_random


def test_synthetic_stream_deterministic_given_same_seed():
    a = SyntheticSequenceStream(seed=1, vocab_size=50, seq_len=8, n_sequences=6, batch_size=2)
    b = SyntheticSequenceStream(seed=1, vocab_size=50, seq_len=8, n_sequences=6, batch_size=2)
    assert torch.equal(a.sequences, b.sequences)
    for _ in range(5):
        assert torch.equal(a.next_batch(), b.next_batch())


def test_synthetic_stream_different_seed_differs():
    a = SyntheticSequenceStream(seed=1, vocab_size=50, seq_len=8, n_sequences=6, batch_size=2)
    b = SyntheticSequenceStream(seed=2, vocab_size=50, seq_len=8, n_sequences=6, batch_size=2)
    assert not torch.equal(a.sequences, b.sequences)


def test_synthetic_stream_checkpoint_resume_reproduces_exact_next_batches():
    reference = SyntheticSequenceStream(seed=3, vocab_size=40, seq_len=6, n_sequences=5, batch_size=2)
    for _ in range(3):
        reference.next_batch()
    saved_state = reference.state_dict()
    expected_next_batches = [reference.next_batch().clone() for _ in range(6)]  # forces at least one reshuffle

    resumed = SyntheticSequenceStream(seed=3, vocab_size=40, seq_len=6, n_sequences=5, batch_size=2)
    resumed.load_state_dict(saved_state)
    actual_next_batches = [resumed.next_batch() for _ in range(6)]

    for expected, actual in zip(expected_next_batches, actual_next_batches):
        assert torch.equal(expected, actual)


def test_synthetic_stream_state_dict_rejects_mismatched_stream():
    a = SyntheticSequenceStream(seed=1, vocab_size=50, seq_len=8, n_sequences=6, batch_size=2)
    b = SyntheticSequenceStream(seed=2, vocab_size=50, seq_len=8, n_sequences=6, batch_size=2)
    try:
        b.load_state_dict(a.state_dict())
        assert False, "expected ValueError for seed mismatch"
    except ValueError:
        pass


def test_synthetic_stream_rejects_batch_size_larger_than_pool():
    try:
        SyntheticSequenceStream(seed=1, vocab_size=10, seq_len=4, n_sequences=3, batch_size=4)
        assert False, "expected ValueError"
    except ValueError:
        pass
