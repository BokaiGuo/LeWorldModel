"""Saved-array algebra fixtures, not neural CPU experiments or efficacy."""
import numpy as np
import pytest

from experiments.spectral_reachability.analyze_reacher_d2p_goal_encoder_metric import square_root


def test_zero_gram_identity():
    np.testing.assert_array_equal(square_root(np.zeros((192,192))), np.eye(192))


def test_diagonal_metric_has_declared_ridge():
    diagonal = np.linspace(0., 3., 192)
    result = square_root(np.diag(diagonal))
    np.testing.assert_allclose(result @ result, np.diag(.01/(diagonal+.01)), atol=1e-14)


@pytest.mark.parametrize("defect", ["shape", "asymmetric", "negative", "nonfinite"])
def test_reject_invalid_gram(defect):
    gram = np.eye(192)
    if defect == "shape": gram = np.eye(191)
    if defect == "asymmetric": gram[0,1] = .1
    if defect == "negative": gram[0,0] = -1.
    if defect == "nonfinite": gram[0,0] = np.nan
    with pytest.raises(ValueError): square_root(gram)
