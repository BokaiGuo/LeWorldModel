import numpy as np
import pytest

from experiments.pc_tworoom_crosswall_mechanism_dev_20261107_codex import task, waypoint_action


def test_all_prespecified_tasks_opposite_sides_and_reproducible():
    for i in range(40):
        start, goal = task(i)
        assert (start[0] - 112) * (goal[0] - 112) < 0
        np.testing.assert_array_equal(start, task(i)[0])
        assert 70 <= start[1] <= 100 and 70 <= goal[1] <= 100


def test_witness_advances_stage_and_bounds_action():
    action, stage = waypoint_action([96, 49], [140, 80], True, 5, 0)
    assert stage == 1
    np.testing.assert_array_equal(action, [1, 0])


@pytest.mark.parametrize("index", [-1, 40, 0.1, True])
def test_task_rejects_invalid_index(index):
    with pytest.raises(ValueError):
        task(index)
