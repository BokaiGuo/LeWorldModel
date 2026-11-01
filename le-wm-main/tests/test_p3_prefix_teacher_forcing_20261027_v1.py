from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.spectral_reachability import p3_prefix_teacher_forcing_20261027_v1 as m


def fixture():
    c=dict(m.base.META,latent_history=np.zeros((3,192)).tolist(),action_history=np.zeros((2,10)).tolist())
    seg=[deepcopy(c) for _ in range(3)]
    seg[0]['action_history'][1][0]=.3
    return c,seg


def test_causal_prefix_not_future():
    c,s=fixture(); original=deepcopy(s)
    observed={1:np.ones(192),2:np.full(192,2.)}
    out=m.causal_context(c,s,observed,1)
    assert out['latent_history']==[c['latent_history'][1],c['latent_history'][2],observed[1].tolist()]
    assert out['action_history']==s[0]['action_history'] and s==original
    out=m.causal_context(c,s,observed,2)
    assert out['latent_history']==[c['latent_history'][2],observed[1].tolist(),observed[2].tolist()]


@pytest.mark.parametrize('boundary',[0,3,-1])
def test_reject_endpoint_or_noncausal_boundary(boundary):
    c,s=fixture()
    with pytest.raises(ValueError):m.causal_context(c,s,{},boundary)


def test_missing_observation_never_substitutes_prediction():
    c,s=fixture()
    with pytest.raises(ValueError):m.causal_context(c,s,{2:np.ones(192)},2)


def test_cap_order_nonrefundable_and_cpu_rejected():
    events=[];cap=m.Cap(events.append,lambda:None);cap.active='00'
    x=SimpleNamespace(device=SimpleNamespace(type='cuda'),shape=(1,3,10))
    cap.dispatch('action_encoder',lambda x:7,(x,),{})
    with pytest.raises(ValueError):cap.dispatch('action_encoder',lambda x:7,(x,),{})
    x.shape=(1,3,192);x.device.type='cpu'
    with pytest.raises(ValueError):cap.dispatch('predict',lambda x:7,(x,),{})


def test_failed_dispatch_cannot_retry():
    cap=m.Cap(lambda e:None,lambda:None);cap.active='00'
    x=SimpleNamespace(device=SimpleNamespace(type='cuda'),shape=(1,3,10))
    def fail(x):raise RuntimeError('injected')
    with pytest.raises(RuntimeError):cap.dispatch('action_encoder',fail,(x,),{})
    assert len(cap.entries)==1 and not cap.entries[0]['completed']
    with pytest.raises(ValueError):cap.dispatch('action_encoder',lambda x:7,(x,),{})


def test_saved_actual_task_census():
    p=m.material()
    assert len(p['tasks'])==17
    assert sum(t['boundary_step']==5 for t in p['tasks'])==16
    assert sum(t['boundary_step']==10 for t in p['tasks'])==1
    assert all(len(t['commands'])==5 for t in p['tasks'])
    assert p['P1_admission'] is p['P3_admission'] is False
