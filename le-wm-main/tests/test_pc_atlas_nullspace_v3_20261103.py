import numpy as np
import pytest

from experiments.spectral_reachability.d2p_atlas_nullspace_v3 import task_nullspace
from experiments.spectral_reachability.pc_atlas_nullspace_unicycle_dev_20261103 import run_goal


@pytest.mark.parametrize("matrix", [
    np.array([[1., 0., 0.], [0., 0., 0.]]),
    np.array([[1., 0.], [0., 0.], [0., 0.]]),
    np.zeros((2, 3)),
])
def test_nullspace_is_complete_and_weighted(matrix):
    scale = np.arange(1, matrix.shape[1] + 1.)
    directions = task_nullspace(matrix, np.eye(matrix.shape[0]), scale)
    assert directions.shape[0] == matrix.shape[1] - np.linalg.matrix_rank(matrix)
    np.testing.assert_allclose(matrix @ directions.T, 0, atol=1e-12)


def test_unicycle_has_turn_nullspace_and_valid_witnesses():
    matrix = np.array([[1., 0.], [0., 0.], [0., 1.]])
    directions = task_nullspace(matrix, np.diag([1., 1., 0.]), np.ones(2))
    np.testing.assert_allclose(np.abs(directions[0]), [0., 1.], atol=1e-12)
    row = run_goal(3)
    for item in row["atlas"]["programs"]:
        assert item["witness_replayed"]
        actions = np.asarray(item["actions"])
        assert (np.abs(actions[:, 0]) <= 1).all()
        assert (np.abs(actions[:, 1]) <= np.pi / 2).all()


def test_nullspace_rejects_nonfinite():
    with pytest.raises(ValueError):
        task_nullspace(np.array([[np.nan]]), np.eye(1), np.ones(1))
