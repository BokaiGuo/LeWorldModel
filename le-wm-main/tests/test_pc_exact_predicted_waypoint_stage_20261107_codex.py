import numpy as np
import pytest

from experiments.pc_exact_prefix_successor_composition_20261107_codex import rollout_n
from experiments.pc_exact_predicted_waypoint_stage_20261107_codex import reached_waypoint


def test_nominally_completed_blocked_waypoint_is_not_completed():
    actions = np.zeros((1, 5, 2))
    actions[0, 0, 0] = 2 / 3
    trace, blocked = rollout_n([-.1, 0., 0.], actions)
    assert blocked[0, 0]
    assert not reached_waypoint(trace[0], 1, [.1, 0.])


def test_completion_uses_nominal_stage_time_not_fifth_step():
    actions = np.zeros((1, 5, 2))
    actions[0, 0, 0] = 2 / 3
    actions[0, 1, 0] = 1 / 3
    trace, _ = rollout_n([-.1, 2., 0.], actions)
    assert reached_waypoint(trace[0], 1, [.1, 2.])
    assert not reached_waypoint(trace[0], 2, [.1, 2.])


def test_not_due_yet_and_invalid_data():
    trace = np.zeros((6, 3))
    assert not reached_waypoint(trace, 6, [0., 0.])
    with pytest.raises(ValueError):
        reached_waypoint(trace, -1, [0., 0.])
    with pytest.raises(ValueError):
        reached_waypoint(np.full((6, 3), np.nan), 1, [0., 0.])
