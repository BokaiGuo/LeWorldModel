import numpy as np
import pytest

from experiments.spectral_reachability.pc_ap_observation_memory import ObservationMemory


def test_exact_match_target_and_adaptive_horizon():
    trajectory = np.arange(12, dtype=float)[:, None]
    memory = ObservationMemory({"memory": trajectory}, ["query"], block=2)
    result = memory.retrieve(np.array([2.]), np.array([8.]), 6, 0)
    assert (result.episode_id, result.start, result.horizon) == ("memory", 2, 6)
    np.testing.assert_array_equal(result.target, [4.])
    assert result.squared_distance == 0
    assert memory.retrieve(np.array([2.]), np.array([4.]), 6, 99).horizon == 2


def test_exclusion_and_insufficient_horizon_fail_closed():
    x = np.arange(8, dtype=float)[:, None]
    with pytest.raises(ValueError, match="query episode"):
        ObservationMemory({"query": x}, ["query"], block=2)
    memory = ObservationMemory({"memory": x}, ["query"], block=2)
    with pytest.raises(ValueError, match="no eligible"):
        memory.retrieve(np.array([0.]), np.array([1.]), 8, 0)


def test_no_cross_episode_records_and_inputs_are_immutable():
    x = np.arange(4, dtype=float)[:, None]
    memory = ObservationMemory({"a": x, "b": x + 100}, ["q"], block=2)
    x[:] = 900
    result = memory.retrieve(np.array([1.]), np.array([3.]), 2, 0)
    assert (result.episode_id, result.start) == ("a", 1)
    np.testing.assert_array_equal(result.target, [3.])
    result.target[:] = -10
    np.testing.assert_array_equal(memory.retrieve(np.array([1.]), np.array([3.]), 2, 0).target, [3.])


def test_rejects_invalid_features_and_queries():
    with pytest.raises(ValueError):
        ObservationMemory({"a": np.full((8, 1), np.nan)}, ["q"])
    memory = ObservationMemory({"a": np.zeros((8, 1))}, ["q"])
    with pytest.raises(ValueError):
        memory.retrieve(np.array([np.nan]), np.array([0.]), 5, 0)
