import numpy as np
import pytest

from experiments.pc_boundary_response_posthoc import differences, probes


def test_boundary_actual_denominators_and_nominal_unchanged():
    nominal = np.array([1, -1, .2, .995], np.float32)
    original = nominal.copy()
    actions, specs = probes(nominal, .01, 4)
    np.testing.assert_array_equal(nominal, original)
    np.testing.assert_array_equal(actions[0], original)
    assert [s["kind"] for s in specs] == ["backward", "forward", "central_secant", "backward"]
    assert np.max(np.abs(actions)) <= 1
    actions64 = actions.astype(np.float64)
    values = np.stack((actions64 @ np.array([2., 3., 4., 5.]), actions64.sum(axis=1)), axis=1)
    matrix, stable = differences(values, np.zeros((len(actions), 2)), specs)
    np.testing.assert_allclose(matrix, [[2, 3, 4, 5], [1, 1, 1, 1]], atol=1e-10)
    assert all(stable)


def test_nonsmooth_branch_is_flagged():
    actions, specs = probes(np.array([0.], np.float32), .05, 1)
    matrix, stable = differences(np.abs(actions), actions >= 0, specs)
    assert not stable[0]
    assert abs(matrix[0, 0]) < 1e-8


def test_bad_nominal_and_unrepresentable_step_fail_closed():
    with pytest.raises(ValueError):
        probes([1.01], .01, 1)
    with pytest.raises(ValueError):
        probes([1.], 1e-12, 1)
