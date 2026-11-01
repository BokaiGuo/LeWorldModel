import numpy as np
import pytest
from experiments.spectral_reachability.p1_fixed40_reach_cross_decomposition_20261029 import split


def test_constant_j_has_no_jacobian_effect():
    rng = np.random.default_rng(11)
    j = rng.normal(size=(192,10))
    result = split(j, np.ones(192), j, np.ones(192)*2)
    assert result['jacobian_effect'] == 0
    assert result['residual_effect'] == pytest.approx(result['total_difference'])
    assert abs(result['identity_error']) < 1e-12


def test_constant_residual_has_no_residual_effect():
    rng = np.random.default_rng(12)
    r = rng.normal(size=192)
    result = split(np.zeros((192,10)), r, rng.normal(size=(192,10)), r)
    assert result['residual_effect'] == 0
    assert result['jacobian_effect'] == pytest.approx(result['total_difference'])
