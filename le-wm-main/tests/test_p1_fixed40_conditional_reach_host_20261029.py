import numpy as np
import pytest
from experiments.spectral_reachability.p1_fixed40_conditional_reach_host_20261029 import metrics
from experiments.spectral_reachability import agent_p1_joint_vreach_saved_diff_20261029_v1 as independent


def test_zero_jacobian_cannot_reduce_residual():
    result = metrics(np.zeros((192,10)), np.zeros(192), np.ones(192))
    assert result['own_reach_value'] == 1.0
    assert result['norm'] == 0.0


def test_invalid_jacobian_rejected():
    with pytest.raises(ValueError):
        metrics(np.zeros((192,9)), np.zeros(192), np.ones(192))


def test_constrained_ridge_matches_independent_solver():
    rng = np.random.default_rng(41)
    j = rng.normal(size=(192, 10)) * 0.1
    target = rng.normal(size=192) * 4
    result = metrics(j, np.zeros(192), target)
    x, _ = independent.independent_solve(j, target, 0.002, np.sqrt(5)*0.75)
    other = independent.compare(j, target, x, 0.002)['score_raw']
    assert result['norm'] <= np.sqrt(5)*0.75 + 1e-12
    assert result['own_reach_value'] == pytest.approx(other, rel=1e-9)
