import numpy as np
import pytest

from experiments.pc_pointmaze_training_camera_alignment_20261106_codex import training_shift, transform


def test_reference_changes_by_minus_fourteen_in_y():
    goal = np.zeros((224, 224, 3), dtype=np.uint8)
    goal[55:197, 41:183] = [200, 100, 50]
    assert training_shift(goal) == {"dx": 0, "dy": -14}
    current, shifted_goal = transform(goal, goal, "train_center")
    assert np.array_equal(current, shifted_goal)


def test_rejects_missing_boundary():
    with pytest.raises(ValueError):
        training_shift(np.zeros((224, 224, 3), dtype=np.uint8))
