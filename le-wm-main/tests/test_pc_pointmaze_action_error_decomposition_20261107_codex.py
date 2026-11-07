import numpy as np
import pytest

from experiments.pc_pointmaze_action_error_decomposition_20261107_codex import analyze


def test_common_bias_decomposition_and_oracle():
    actual = np.array([[0., 1.], [2., 3.], [4., 5.]])
    pred = actual + np.array([2., -1.])
    row = analyze(pred, actual, np.array([1., 1.]), [1., 2., 3.])
    assert row["raw_mse"] == pytest.approx(2.5)
    assert row["common_bias_mse"] == pytest.approx(2.5)
    assert row["centered_error_mse"] == pytest.approx(0)
    assert row["effect_cosine"] == pytest.approx(1)
    assert row["selected"]["oracle_bias"] == row["selected"]["actual_encoding"]


def test_action_inversion_and_zero_variation():
    actual = np.array([[-1.], [0.], [1.]])
    row = analyze(-actual, actual, np.array([0.]), [0., 1., 2.])
    assert row["effect_cosine"] == pytest.approx(-1.)
    assert row["centered_error_mse"] == pytest.approx(4 * row["zero_change_baseline_mse"])
    assert analyze(np.zeros_like(actual), actual, np.array([0.]), [0., 1., 2.])["effect_cosine"] is None


def test_incompatible_or_nonfinite_inputs_fail_closed():
    with pytest.raises(ValueError):
        analyze(np.ones((2, 1)), np.ones((3, 1)), np.ones(1), [1., 2.])
    with pytest.raises(ValueError):
        analyze(np.array([[np.nan], [1.]]), np.ones((2, 1)), np.ones(1), [1., 2.])
