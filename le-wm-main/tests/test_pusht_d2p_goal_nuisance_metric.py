import numpy as np
import pytest

from experiments.spectral_reachability.pusht_d2p_goal_nuisance_metric import compile_metric


def test_preserves_orthogonal_task_direction_and_quotients_goal_span():
    goals = np.array([[2., 3., 4.], [3., 3., 4.], [2., 7., 4.]])
    q, receipt = compile_metric(goals)
    assert receipt["rank"] == 2
    np.testing.assert_allclose(q@q, q, atol=1e-14)
    np.testing.assert_allclose(q.T, q, atol=1e-14)
    np.testing.assert_allclose(goals@q, np.tile(goals[0]@q, (3, 1)), atol=1e-14)
    np.testing.assert_allclose(q@[0., 0., 1.], [0., 0., 1.], atol=1e-14)


def test_identical_goals_leave_metric_unchanged():
    q, receipt = compile_metric(np.ones((5, 192)))
    assert receipt["rank"] == 0
    np.testing.assert_array_equal(q, np.eye(192))


@pytest.mark.parametrize("goals", [np.ones((1, 3)), np.ones(3), np.full((2, 3), np.nan),
                                   [[0., 0.], [1., 0.], [0., 1.]]])
def test_invalid_or_empty_task_metric_rejected(goals):
    with pytest.raises(ValueError):
        compile_metric(goals)
