import numpy as np
import pytest
from experiments.spectral_reachability.pusht_task_spectral_bank import synthesize


def make(jac=None, goal=None):
    return synthesize(jacobian=np.diag([2.,1.]) if jac is None else jac,
                      nominal_terminal=np.zeros(2), goal=np.ones(2) if goal is None else goal,
                      projection=np.eye(2))


def test_complete_bounded_bank():
    r=make(); p=np.asarray(r['params'])
    assert p.shape==(10,1,2)
    assert np.array_equal(p[0],np.zeros((1,2)))
    assert np.max(np.linalg.norm(p[:,0],axis=1)) <= .75+1e-15
    assert np.max(np.abs(np.tanh(p))) < 1


def test_progress_matches_augmented_ridge_before_cap():
    r=make(goal=np.full(2,.01)); p=np.asarray(r['params'])[1,0]
    expected=np.linalg.lstsq(np.vstack([np.diag([2.,1.]),.1*np.eye(2)]),np.r_[np.full(2,.01),[0.,0.]],rcond=None)[0]
    np.testing.assert_allclose(p,expected,atol=1e-15)


def test_response_changes_spectral_directions():
    a=np.asarray(make()['right_vectors']); b=np.asarray(make(jac=[[1.,1.],[0.,1.]])['right_vectors'])
    assert not np.array_equal(a,b)


def test_zero_jacobian_no_fake_progress():
    assert np.array_equal(make(jac=np.zeros((2,2)))['params'][1],[[0.,0.]])


@pytest.mark.parametrize('jac',[np.full((2,2),np.nan),np.ones((2,3)),np.ones((2,2),dtype=bool),np.ones((2,2))*1j])
def test_bad_jacobian_rejected(jac):
    with pytest.raises(ValueError): make(jac=jac)


def test_deterministic():
    assert make()==make()
