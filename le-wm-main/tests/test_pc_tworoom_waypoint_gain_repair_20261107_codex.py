import numpy as np
import pytest

from experiments import pc_tworoom_waypoint_gain_repair_20261107_codex as repair


@pytest.mark.parametrize("gain", [0.8, 1.0, 1.2])
def test_waypoints_are_reached_without_overshoot(gain):
    start = np.array([0., 0.])
    points = np.array([[13.2, -2.3], [15.1, 12.7], [2.4, 4.5]])
    actions, stages = repair.waypoint_actions(start, points, gain, 25)
    assert np.max(np.abs(actions)) <= 1
    assert stages[-1] == 2
    position = start.copy()
    for action, stage in zip(actions, stages, strict=True):
        before = points[stage] - position
        position += 5 * action
        after = points[stage] - position
        assert np.all(before * after >= -1e-4)
    for new_stage in (1, 2):
        t = stages.index(new_stage)
        previous = start + 5 * actions[:t].sum(axis=0)
        np.testing.assert_allclose(previous, points[new_stage - 1], atol=1e-5, rtol=0)


def test_bad_geometry_or_gain_rejected():
    with pytest.raises(ValueError):
        repair.waypoint_actions([0., 0.], [[1., 2.]], 0., 5)
    with pytest.raises(ValueError):
        repair.candidate_pool(np.zeros((3, 3)), np.zeros((3, 3)))
