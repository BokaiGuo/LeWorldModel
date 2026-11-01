import numpy as np
import pytest
from experiments.spectral_reachability.pusht_full_packet_compiler_20261027 import (
    bounded_ridge,spectral_bank,CAP,make_transition,
)


def test_unconstrained_ridge_matches_direct_solve():
    rng=np.random.default_rng(3);k=rng.normal(size=(192,10));r=rng.normal(size=192)*.01
    got=bounded_ridge(k,r)
    expected=np.linalg.solve(k.T@k+.002*np.eye(10),k.T@r)
    np.testing.assert_allclose(got['params'],expected,atol=1e-14)
    assert got['ball_multiplier']==0


def test_active_ball_satisfies_kkt_and_score_matches_applied_action():
    k=np.diag([1.,3.]);r=np.array([100.,100.])
    got=bounded_ridge(k,r,cap=.75);u=np.asarray(got['params'])
    assert np.isclose(np.linalg.norm(u),.75)
    assert got['ball_multiplier']>0
    np.testing.assert_allclose((k.T@k+(.002+got['ball_multiplier'])*np.eye(2))@u,k.T@r)
    assert np.isclose(got['own_reach_value'],((r-k@u)@(r-k@u)+.002*(u@u))/2)


@pytest.mark.parametrize('width',[2,10])
def test_same_ten_candidates_and_total_norm_cap(width):
    rng=np.random.default_rng(2);j=rng.normal(size=(192,width))
    result=spectral_bank(j,np.zeros(192),np.ones(192),np.eye(192))
    values=np.asarray(result['params'])
    assert values.shape==(10,width)
    assert np.max(np.linalg.norm(values,axis=1))<=CAP+1e-12
    assert np.array_equal(values[0],np.zeros(width))


def test_full_contains_constant_linear_reach():
    from experiments.spectral_reachability.pusht_full_packet_chart_20261027 import chart_matrix
    rng=np.random.default_rng(4);j=rng.normal(size=(192,10));r=rng.normal(size=192)
    full=bounded_ridge(j,r);constant=bounded_ridge(j@chart_matrix('constant'),r)
    assert full['own_reach_value']<=constant['own_reach_value']+1e-14


@pytest.mark.parametrize('bad',[np.full((2,2),np.nan),np.ones((2,2),complex),np.ones((2,2),bool)])
def test_bad_response_rejected(bad):
    with pytest.raises(ValueError):
        bounded_ridge(bad,np.ones(2))


def test_cpu_neural_entry_rejected_before_runtime_import():
    class Device:
        type='cpu'
    with pytest.raises(ValueError,match='CUDA'):
        make_transition(object(),Device())
