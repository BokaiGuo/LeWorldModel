import numpy as np
import pytest

from experiments.spectral_reachability.p1_cp10_margin_decomposition_20261031 import decompose


def test_exact_decomposition_with_nonzero_shared_and_action_error():
    goal = np.zeros(192)
    actual = {"open": np.ones(192), "feedback": np.full(192, 2.0)}
    pred = {"open": np.full(192, 1.5), "feedback": np.full(192, 1.0)}
    row = decompose(goal, pred, actual)
    assert row["predicted_margin"] == pytest.approx(-1.25)
    assert row["actual_margin"] == pytest.approx(3.0)
    assert row["cross_term_difference"] + row["error_mse_difference"] == pytest.approx(-4.25)
    assert row["action_error_difference_mse"] == pytest.approx(2.25)
    assert row["error_cosine"] == pytest.approx(-1.0)


def test_rejects_malformed_or_nonfinite():
    zeros = np.zeros(192)
    with pytest.raises(ValueError):
        decompose(zeros, {"open": zeros, "feedback": np.full(192, np.nan)},
                  {"open": zeros, "feedback": zeros})
    with pytest.raises(ValueError):
        decompose(zeros[:-1], {"open": zeros, "feedback": zeros},
                  {"open": zeros, "feedback": zeros})


def test_identical_additive_error_can_change_margin():
    goal = np.zeros(192)
    actual = {"open": np.ones(192), "feedback": np.full(192, 2.0)}
    pred = {arm: value - 2.0 for arm, value in actual.items()}
    row = decompose(goal, pred, actual)
    assert row["action_error_difference_mse"] == pytest.approx(0.0)
    assert row["actual_margin"] == pytest.approx(3.0)
    assert row["predicted_margin"] == pytest.approx(-1.0)
    assert row["cross_term_difference"] == pytest.approx(-4.0)
    assert row["error_mse_difference"] == pytest.approx(0.0)
