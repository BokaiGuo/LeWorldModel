import numpy as np
import pytest
from experiments.spectral_reachability.analyze_pusht_sealed_prediction_residuals import residual


def test_exact_decomposition_including_negative_cross():
    r = residual([2., 3.], [1., 8.], [0., 0.], [[1., 0.], [0., 0.]])
    assert r['predicted_cost'] == 2
    assert r['actual_cost'] == .5
    assert r['projected_vector_error_mse'] == .5
    assert r['cross_term'] == -2
    assert r['cost_prediction_error'] == -1.5


def test_perfect_forecast():
    r = residual([1., 2.], [1., 2.], [3., 4.], np.eye(2))
    assert r['projected_vector_error_mse'] == 0
    assert r['cost_prediction_error'] == 0


def test_nonfinite_rejected():
    with pytest.raises(ValueError): residual([np.nan], [0.], [0.], [[1.]])


def test_shape_rejected():
    with pytest.raises(ValueError): residual([0., 0.], [0.], [0.], [[1.]])
