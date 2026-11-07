import numpy as np
import pytest

from experiments.pc_pointmaze_camera_geometry_20261107_codex import camera_grid, errors


def test_camera_grid_is_fixed_and_contains_original():
    grid = camera_grid()
    assert grid.shape == (25, 2)
    assert np.count_nonzero(np.all(grid == [3., 2.5], axis=1)) == 1


def test_image_shape_must_not_broadcast():
    with pytest.raises(ValueError):
        errors(np.zeros((8, 1, 224, 224, 3)), np.zeros((8, 224, 224, 3)))
