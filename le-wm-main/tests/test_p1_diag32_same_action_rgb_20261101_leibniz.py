import numpy as np
import pytest

from experiments.spectral_reachability import p1_diag32_same_action_rgb_20261101_leibniz as diag


def test_fixed_costs_and_identity_mse():
    actual = np.zeros((4, 192))
    predicted = np.ones((4, 192))
    rows = diag.metrics(actual, predicted, np.zeros(192), [{"i": i} for i in range(4)])
    assert [r["step"] for r in rows] == [0, 5, 10, 15]
    assert all(r["predicted_goal_mse"] == r["predicted_actual_latent_rmse"] == 1 for r in rows)
    assert diag.costs()["encode_calls"] == 4
    assert diag.costs()["predict_calls"] == 3


def test_rejects_bad_shape_and_nan():
    with pytest.raises(ValueError):
        diag.metrics(np.zeros((3, 192)), np.zeros((4, 192)), np.zeros(192), [{}] * 4)
    x = np.zeros((4, 192))
    x[0, 0] = np.nan
    with pytest.raises(ValueError):
        diag.metrics(x, np.zeros((4, 192)), np.zeros(192), [{}] * 4)


def test_cannot_use_unscoped_output(tmp_path):
    with pytest.raises(ValueError):
        diag.exclusive(tmp_path / (diag.NAME + "_bad"))
