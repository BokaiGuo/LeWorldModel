import numpy as np
import pytest
from experiments.spectral_reachability.pusht_fixed_residual_response import response,central_difference


def test_zero_response_not_improvement():
    r=response(np.zeros((3,2)),np.ones(3),np.eye(3))
    assert r['action_improvement']==0


def test_response_matches_closed_form():
    j=np.array([[2.,0.],[0.,1.],[0.,0.]])
    r=response(j,np.ones(3),np.eye(3))
    assert r['action_improvement']==pytest.approx((4/4.01+1/1.01)/3)


def test_fixed_residual_stronger_response_improves():
    j=np.eye(2); r=np.ones(2);q=np.eye(2)
    assert response(j*2,r,q)['action_improvement']>response(j,r,q)['action_improvement']


def test_fd_parameter_units():
    e=float(np.float32(.05)); j=np.array([[1.,2.],[3.,4.]])
    p=[dict(axis=a,sign=s,status='completed',vector=(j[:,a]*s*e).tolist()) for a in (0,1) for s in (-1,1)]
    np.testing.assert_allclose(central_difference(p,.05),j)
    p[0]['status']='observation_unavailable'
    with pytest.raises(ValueError):central_difference(p,.05)


@pytest.mark.parametrize('j',[np.ones((2,2))*1j,np.ones((2,2),dtype=bool),np.full((2,2),np.nan)])
def test_bad_jacobian(j):
    with pytest.raises(ValueError):response(j,np.ones(2),np.eye(2))
