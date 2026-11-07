import numpy as np
import pytest

from experiments.pc_tworoom_geometric_stage_repair_20261107_codex import follow_waypoints


@pytest.mark.parametrize("gain", [.8, 1., 1.2])
def test_stages_advance_and_no_overshoot(gain):
    points = np.array([[1.3, 0.], [3.2, 0.], [5., 0.]])
    actions, stages = follow_waypoints([0., 0.], points, gain, 15)
    assert 2 in stages
    assert all(a <= b for a, b in zip(stages, stages[1:]))
    assert np.abs(actions).max() <= 1
    position = np.zeros(2)
    for action, stage in zip(actions, stages):
        before = points[stage] - position
        position += 5 * action
        after = points[stage] - position
        assert np.all(before * after >= -1e-6)


def test_invalid_waypoint_inputs():
    with pytest.raises(ValueError):
        follow_waypoints([0., 0.], [], 1., 10)
    with pytest.raises(ValueError):
        follow_waypoints([0., 0.], [[1., 1.]], 0., 10)
