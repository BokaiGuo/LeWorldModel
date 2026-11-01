"""Logic fixtures only; no neural forward, CUDA allocation, or physics."""
from copy import deepcopy
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import p3_fixed_horizon_budget_schedule_20261027_v2 as s


def context():
    return dict(layout='states_with_between_state_actions_v1',context_length=3,
        latent_dim=2,action_dim=10,latent_history=[[0.,0.]]*3,action_history=[[0.]*10]*2)


def bank():
    return [np.tile([i/16,0.],(5,1)) for i in range(16)]


def replay(c,a):
    child=deepcopy(c)
    x=np.asarray(c['latent_history'][-1])+[float(a[0,0])+1,0.]
    child['latent_history']=c['latent_history'][1:]+[x.tolist()]
    child['action_history']=c['action_history'][1:]+[[float(a[0,0]),0.]*5]
    return child


def search(**kw):
    return s.run_search(**(dict(root_context=context(),primitive_bank=bank(),replay=replay,
        goal=[10.,10.],projection=np.eye(2),mode='dynamic')|kw))


@pytest.mark.parametrize('chi,k2,k3',[(0.,2,6),(.1249,2,6),(.125,3,5),(.5,4,4),(.875,6,2),(1.,6,2)])
def test_frozen_allocation(chi,k2,k3):
    assert s.allocate(chi,'dynamic')==(k2,k3)
    assert s.allocate(chi,'uniform')==(4,4)


@pytest.mark.parametrize('bad',[True,-.1,1.1,float('nan'),float('inf'),None])
def test_bad_chi(bad):
    with pytest.raises(ValueError):
        s.allocate(bad,'dynamic')


def test_goal_related_svd_not_singular_values_only():
    children=[replay(context(),a) for a in bank()]
    parallel=s.response_geometry(context(),children,[10.,0.],np.eye(2))
    orthogonal=s.response_geometry(context(),children,[0.,10.],np.eye(2))
    assert parallel['chi']==pytest.approx(0.)
    assert orthogonal['chi']==pytest.approx(1.)
    assert parallel['singular_values']==orthogonal['singular_values']
    assert parallel['response_rank']==1
    assert orthogonal['residual']==[0.,10.]
    assert len(orthogonal['projected_response'])==16


def test_same_chi_as_original_response_definition_in_goal_projection_space():
    from experiments.spectral_reachability.p3_adaptive_runtime_20261027 import RuntimePolicy,response_signal
    root=context();children=[replay(root,a) for a in bank()]
    q=np.array([[.5,.5],[.5,.5]])
    expected,details=response_signal(root,children,[0.,10.],q,RuntimePolicy('fixed_source'))
    got=s.response_geometry(root,children,[0.,10.],q)
    assert got['chi']==pytest.approx(expected.unreachable_fraction,abs=1e-15)
    assert got['response_rank']==details['response_rank']
    np.testing.assert_allclose(got['residual'],[5.,5.])


@pytest.mark.parametrize('mutation',['partial','zero_residual','zero_response','bad_q','no_goal'])
def test_signal_prerequisites(mutation):
    c=context(); children=[replay(c,a) for a in bank()];g=[10.,0.];q=np.eye(2)
    if mutation=='partial': children.pop()
    elif mutation=='zero_residual': g=[0.,0.]
    elif mutation=='zero_response': children=[deepcopy(c) for _ in range(16)]
    elif mutation=='bad_q': q=np.ones((2,2))
    else: g=None
    with pytest.raises(ValueError):
        s.response_geometry(c,children,g,q)


@pytest.mark.parametrize('mode',['dynamic','uniform'])
def test_fixed_h3_exact_fullbank_budget_and_replay(mode):
    r=search(mode=mode)
    assert r['status']=='completed' and len(r['events'])==144
    assert r['planning_edge_attempts']==144 and r['svd_calls']==1
    assert sum(r['allocation'])==8
    assert [len(l['parents']) for l in r['layers']]==[1,*r['allocation']]
    assert all(e['succeeded'] for e in r['events'])
    for layer in r['layers']:
        for parent in layer['parents']:
            assert [e['primitive_index'] for e in r['events'] if e['parent_path']==parent and
                    e['depth']==layer['depth']]==list(range(16))
    best=r['best_plan']; assert best['depth']==3 and len(best['actions'])==15
    got=context()
    for a in np.asarray(best['actions']).reshape(3,5,2): got=replay(got,a)
    assert got==best['context']


def test_no_shallow_winner_when_shallow_cost_smaller():
    r=search(goal=[1.,1.])
    assert min(e['cost'] for e in r['events'] if e['depth']==1)<r['best_plan']['cost']
    assert r['best_plan']['depth']==3


def test_arms_same_root_signal_and_inputs_unchanged():
    c,b=context(),bank(); old=deepcopy(c)
    a=search(root_context=c,primitive_bank=b,mode='uniform')
    z=search(root_context=c,primitive_bank=b,mode='dynamic')
    assert a['signal']==z['signal'] and c==old
    for x,y in zip(b,bank()): np.testing.assert_array_equal(x,y)


@pytest.mark.parametrize('fail_at',[1,16,17,80,144])
def test_failed_edges_preserved_no_refund_no_partial_winner(fail_at):
    calls=0
    def failing(c,a):
        nonlocal calls
        calls+=1
        if calls==fail_at: raise RuntimeError('injected edge failure')
        return replay(c,a)
    r=search(mode='uniform',replay=failing)
    assert r['status']=='incomplete' and r['best_plan'] is None
    assert r['planning_edge_attempts']==calls and len(r['events'])==calls
    assert sum(not e['succeeded'] for e in r['events'])==1
    if fail_at<=16: assert r['signal'] is None and r['allocation'] is None
    assert len(r['events']) in (16,80,144)


def test_invalid_bank_before_replay():
    with pytest.raises(ValueError): search(primitive_bank=bank()[:-1])


def test_zero_response_no_fallback_to_uniform():
    r=search(replay=lambda c,a:deepcopy(c),mode='uniform')
    assert r['status']=='signal_unavailable' and r['planning_edge_attempts']==16
    assert r['allocation'] is None and r['best_plan'] is None


def test_modules_import_without_torch_or_simulator():
    script=('import sys; from experiments.spectral_reachability import '
            'p3_fixed_horizon_budget_schedule_cuda_20261027_v2 as m; '
            'assert not any(n.split(".")[0] in {"torch","mujoco","pymunk"} for n in sys.modules)')
    result=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
