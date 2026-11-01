import numpy as np
import pytest
from experiments.spectral_reachability import main_raw_coordinate_response_20261028_v1 as m


def test_identical_response_can_have_opposite_objective_direction():
    pm=np.ones(192);pp=3*pm;am=-pp;ap=-pm
    r=m.pair(pm.tolist(),pp.tolist(),am.tolist(),ap.tolist(),np.zeros(192).tolist(),2.)
    assert r['vector_cosine']==pytest.approx(1)
    assert r['strict_direction_reversal'] and r['terms']['response']==0


def test_zero_physical_secant_is_not_cosine_zero():
    a=np.ones(192).tolist();b=(2*np.ones(192)).tolist()
    r=m.pair(a,b,a,a,np.zeros(192).tolist(),.01)
    assert r['actual_secant_norm']==0 and r['vector_cosine'] is None
    assert not r['strict_direction_reversal']


def test_fixed_baseline_constant_offset_and_response_separate():
    a=np.ones(192);g=np.zeros(192)
    r=m.anchored_terms((a+2).tolist(),(a+3).tolist(),a.tolist(),(a+1).tolist(),g.tolist(),.02,
                       (a+2).tolist(),a.tolist())
    assert r['constant_offset_for_both_probes']
    assert r['action_response_term']==0 and r['common_anchor_term']!=0


def test_random_fixed_anchor_identity():
    rng=np.random.default_rng(281028)
    for i in range(20):
        a=[rng.normal(size=192).tolist() for j in range(7)]
        r=m.anchored_terms(*a[:5],.002,*a[5:])
        assert abs(r['identity_residual'])<1e-10
