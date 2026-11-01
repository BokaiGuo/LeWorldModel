import numpy as np
import pytest
from experiments.spectral_reachability.main_suffix_linearization_audit_20261028_v1 import terms


def test_exact_linear_model_has_no_curvature():
    n=np.ones(192);j=np.zeros((192,10));j[:,0]=1;p=np.zeros(10);p[0]=-.5
    r=terms(n,j,p,n+j@p,np.zeros(192))
    assert r['linear_delta']==r['nonlinear_delta']==-.75
    assert r['nonlinear_remainder_MSE']==r['expansion_residual']==0


def test_good_local_step_can_be_bad_nonlinear_step_without_solver_sign_error():
    n=np.ones(192);j=np.zeros((192,10));j[:,0]=1;p=np.zeros(10);p[0]=-.5
    r=terms(n,j,p,np.full(192,2.),np.zeros(192))
    assert r['local_regularized_delta']<0<r['nonlinear_delta']
    assert r['linear_delta']+r['nonlinear_remainder_cross']+r['nonlinear_remainder_square']==r['nonlinear_delta']


def test_invalid_shape_is_not_a_negative_result():
    with pytest.raises(ValueError):
        terms(np.zeros(191),np.zeros((192,10)),np.zeros(10),np.zeros(192),np.zeros(192))
