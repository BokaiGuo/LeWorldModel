import numpy as np
import pytest

from experiments.pc_tworoom_predicted_stop_dev_20261107_codex import choose, action_hash


def test_choice_uses_candidate_and_time_axes_and_stable_ties():
    costs = np.full((64, 5), 10.0)
    costs[7, 4] = 2
    costs[3, 1] = 1
    costs[4, 0] = 1
    assert choose(costs) == {"final25": (7, 25), "min25": (3, 25), "min_stop": (3, 10)}


def test_rejects_invalid_prediction():
    with pytest.raises(ValueError):
        choose(np.zeros((5, 64)))
    costs = np.zeros((64, 5))
    costs[0, 0] = np.nan
    with pytest.raises(ValueError):
        choose(costs)


def test_action_hash_binds_executed_length_and_values():
    actions = np.zeros((25, 2), np.float32)
    assert action_hash(actions[:5]) != action_hash(actions)
    actions[0, 0] = 0.5
    assert action_hash(actions[:5]) != action_hash(np.zeros((5, 2), np.float32))
