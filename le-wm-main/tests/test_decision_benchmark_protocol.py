import numpy as np

from experiments.decision_benchmark.generate_candidate_bank import (
    _bank_hash,
    _candidate_sequences,
    _split_for_index,
)


def test_candidate_bank_has_equal_duration_and_is_deterministic():
    source = np.asarray([0.4, 2.0])
    goal = np.asarray([1.2, 1.2])
    first = _candidate_sequences(source, goal, count=16, steps=20, seed=7000)
    second = _candidate_sequences(source, goal, count=16, steps=20, seed=7000)
    assert first.shape == (16, 20, 2)
    assert np.array_equal(first, second)
    assert _bank_hash(first) == _bank_hash(second)


def test_candidate_bank_changes_sequences_not_duration_and_split_is_root_level():
    first = _candidate_sequences(np.asarray([0.4, 2.0]), np.asarray([1.2, 1.2]), 16, 20, 1)
    second = _candidate_sequences(np.asarray([1.0, 1.0]), np.asarray([1.2, 1.2]), 16, 20, 2)
    assert first.shape[1] == second.shape[1] == 20
    assert not np.array_equal(first, second)
    assert [_split_for_index(i) for i in range(5)] == ["train", "train", "train", "calibration", "outcome"]
    assert [_split_for_index(i, "distance_balanced_v1") for i in range(5)] == [
        "outcome", "train", "train", "calibration", "train"
    ]
