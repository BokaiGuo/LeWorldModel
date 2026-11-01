import numpy as np
import pytest

from experiments.spectral_reachability.p3_raw_chart_ridge_20261030_main import solve_raw_ridge


def test_interior_solution_matches_closed_form_and_returns_raw():
    j = np.eye(10)
    residual = np.full(10, 0.1)
    result = solve_raw_ridge(j, residual)
    np.testing.assert_allclose(result["raw"].ravel(), residual / 1.002, atol=1e-6)
    assert result["raw"].dtype == np.float32
    assert result["coordinate"] == "raw_float32"


def test_box_and_ball_both_respected():
    residual = np.array([10.0] + [2.0] * 9)
    result = solve_raw_ridge(np.eye(10), residual, radius=1.3)
    raw = result["raw"].ravel()
    assert np.max(np.abs(raw)) <= 1
    assert np.linalg.norm(raw) <= 1.3 + 1e-6
    assert raw[0] > 0.999
    assert result["linear_objective"] < np.mean(residual**2)


@pytest.mark.parametrize("j,r", [
    (np.ones((2, 9)), np.ones(2)),
    (np.ones((2, 10)), np.array([np.nan, 0.])),
    (np.ones((2, 10)), np.ones(3)),
])
def test_invalid_inputs_fail_closed(j, r):
    with pytest.raises(ValueError):
        solve_raw_ridge(j, r)
