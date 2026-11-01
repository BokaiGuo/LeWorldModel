from copy import deepcopy
import numpy as np
import pytest
from experiments.spectral_reachability import main_suffix_incumbent_guard_20261028_v1 as m


def candidate(value):
    return dict(context_sha256='a'*64, goal192=[0.]*192, commands=np.full((5,2),.125,np.float32).tolist(),
        predicted192=[value]*192, domain_verified=True, prediction_verified=True)


@pytest.mark.parametrize('old,new,selected',[(1.,2.,'incumbent'),(1.,1.,'incumbent'),(1.,.5,'proposal')])
def test_guard_keeps_legitimate_incumbent_on_harm_or_tie(old,new,selected):
    a,b=candidate(old),candidate(new);before=deepcopy((a,b))
    r=m.choose(a,b)
    assert r['selected']==selected and r['selected_Q']<=r['incumbent_Q']
    assert not r['physical_improvement_guaranteed'] and not r['P1_admission']
    assert (a,b)==before


@pytest.mark.parametrize('fault',['context','goal','nan','labels','domain','prediction','raw_precision','raw_bound'])
def test_domain_time_goal_and_outcome_leak_failclosed(fault):
    a,b=candidate(1.),candidate(.5)
    if fault=='context':b['context_sha256']='b'*64
    elif fault=='goal':b['goal192'][0]=1.
    elif fault=='nan':b['predicted192'][0]=float('nan')
    elif fault=='labels':b['native_success']=True
    elif fault=='domain':b['domain_verified']=False
    elif fault=='prediction':b['prediction_verified']=False
    elif fault=='raw_precision':b['commands'][0][0]=.10000000001
    else:b['commands'][0][0]=2.
    with pytest.raises(ValueError):m.choose(a,b)
