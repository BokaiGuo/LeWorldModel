import numpy as np
import pytest
from experiments.spectral_reachability.run_pusht_observed_oneshot_suffix_control import one_shot


def test_same_ridge_formula_in_raw_parameter_coordinates():
    goal, nominal = np.array([2., 3., 1.]), np.array([0., 1., 1.])
    jac = np.array([[1., 0.], [0., 2.], [.1, .3]])
    projection = np.diag([1., 0., 1.])
    expected = np.linalg.solve(jac.T@projection@jac+.01*np.eye(2), jac.T@projection@(goal-nominal))
    np.testing.assert_allclose(one_shot(goal, nominal, jac, projection)[0], expected)


@pytest.mark.parametrize('damping', [0, -1, True, float('nan')])
def test_invalid_damping(damping):
    with pytest.raises(ValueError): one_shot([1., 1.], [0., 0.], np.eye(2), np.eye(2), damping)


def test_no_response_gives_zero_not_nan():
    np.testing.assert_array_equal(one_shot([1., 2.], [0., 0.], np.zeros((2, 2)), np.eye(2)), [[0., 0.]])
