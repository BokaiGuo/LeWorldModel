"""Analytic NumPy fixtures, not neural or control evidence."""
import numpy as np
import pytest
from experiments.spectral_reachability import audit_prefix_span_MAIN_20261028_v1 as m


def test_full_coordinates_with_only_three_proposal_directions():
    p=np.zeros((10,10));p[1]=np.ones(10);p[2,0]=1;p[3,0]=-1;p[4,1]=1;p[5,1]=-1
    j=np.zeros((192,10));j[:10]=np.eye(10)
    c=m.coverage(p,j)
    assert c['logit_dimension']==c['root_jacobian_rank']==10
    assert c['proposal_span_rank']==3 and c['omitted_logit_dimensions']==7
    assert c['omitted_J_frobenius_energy_ratio']==pytest.approx(.7)


def test_zero_bank_and_zero_jacobian_are_not_imputed_nonzero():
    c=m.coverage(np.zeros((10,10)),np.zeros((192,10)))
    assert c['proposal_span_rank']==0 and c['root_jacobian_rank']==0
    assert c['omitted_J_frobenius_energy_ratio'] is None


@pytest.mark.parametrize('value',[np.zeros((9,10)),np.full((10,10),np.nan)])
def test_bad_bank_refused(value):
    with pytest.raises(ValueError):m.coverage(value,np.zeros((192,10)))
