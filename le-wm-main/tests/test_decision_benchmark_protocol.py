import numpy as np
import pytest

from experiments.decision_benchmark.generate_candidate_bank import (
    _bank_hash,
    _candidate_sequences,
    _checkpoint_protocol_split_map,
    _hash_random_split_map,
    _to_goal_frame,
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


def test_hash_random_split_is_exact_deterministic_and_order_invariant():
    roots = [f"root_{index:03d}" for index in range(100)]
    first = _hash_random_split_map(roots)
    second = _hash_random_split_map(list(reversed(roots)))
    assert first == second
    assert list(first.values()).count("train") == 60
    assert list(first.values()).count("calibration") == 20
    assert list(first.values()).count("outcome") == 20


def test_checkpoint_protocol_split_preserves_model_provenance():
    checkpoint = {
        "protocol": {
            "train_episode_ids": [1, 2],
            "calibration_episode_ids": [3],
            "outcome_episode_ids": [4, 5],
        }
    }
    split = _checkpoint_protocol_split_map(checkpoint, ["1", "3", "5"])
    assert split == {"1": "train", "3": "calibration", "5": "outcome"}


def test_checkpoint_protocol_split_rejects_unassigned_episode():
    checkpoint = {
        "protocol": {
            "train_episode_ids": [1],
            "calibration_episode_ids": [2],
            "outcome_episode_ids": [3],
        }
    }
    with pytest.raises(ValueError, match="does not assign"):
        _checkpoint_protocol_split_map(checkpoint, ["4"])


def test_canonical_action_bank_is_shared_in_goal_relative_coordinates():
    goal = np.asarray([1.2, 1.2])
    first_source = np.asarray([0.4, 2.0])
    second_source = np.asarray([1.0, 1.0])
    first = _candidate_sequences(first_source, goal, 16, 20, 7000, canonical_action_frame=True)
    second = _candidate_sequences(second_source, goal, 16, 20, 7000, canonical_action_frame=True)
    assert np.allclose(_to_goal_frame(first, first_source, goal), _to_goal_frame(second, second_source, goal))
    assert np.array_equal(
        _bank_hash(np.round(_to_goal_frame(first, first_source, goal), 8)),
        _bank_hash(np.round(_to_goal_frame(second, second_source, goal), 8)),
    )
    assert float(np.max(np.abs(first))) <= 1.0 + 1.0e-12
    assert float(np.max(np.abs(second))) <= 1.0 + 1.0e-12


def test_shared_world_action_bank_is_identical_but_not_goal_aligned():
    goal = np.asarray([1.2, 1.2])
    first_source = np.asarray([0.4, 2.0])
    second_source = np.asarray([1.0, 1.0])
    first = _candidate_sequences(
        first_source, goal, 16, 20, 7000, world_action_frame=True,
    )
    second = _candidate_sequences(
        second_source, goal, 16, 20, 7000, world_action_frame=True,
    )
    assert np.array_equal(first, second)
    assert not np.allclose(
        _to_goal_frame(first, first_source, goal),
        _to_goal_frame(second, second_source, goal),
    )
    assert float(np.max(np.abs(first))) <= 1.0 + 1.0e-12


def test_candidate_bank_frames_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _candidate_sequences(
            np.asarray([0.0, 0.0]), np.asarray([1.0, 1.0]), 16, 20, 7000,
            canonical_action_frame=True, world_action_frame=True,
        )
