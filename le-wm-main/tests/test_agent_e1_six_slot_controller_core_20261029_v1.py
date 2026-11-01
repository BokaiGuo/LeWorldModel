"""RAM only; no model, no physics, no production."""
from copy import deepcopy
import numpy as np
import pytest
import test_agent_experiment1_common_controller_20261028_v1 as fixture
from experiments.spectral_reachability import agent_e1_six_slot_controller_core_20261029_v1 as m


def transition(task):
    def forward(before,raw,phase,path):
        c=deepcopy(before);out=[]
        for i in range(len(raw)//5):
            x=np.asarray(raw[i*5:i*5+5],np.float32)
            z=np.asarray(c['latent_history'][-1],np.float32)+np.float32(x.sum()*.01)
            c=m.old.beam.rolled(task,c,x,z);out.append(c)
        return out
    return forward


def test_learned_six_slot_original_math_and_replay_barrier():
    t=fixture.task(0); events=[];meter=m.Attempts()
    target=t['root_context']['latent_history'][-1]
    result=m.plan(t,0,m.ARMS[1],target,transition(t),events.append,meter)
    assert len(result['candidates'])==72 and meter.attempted==meter.returned==144
    r=m.replay_slot(t,result,transition(t),events.append,meter)
    assert meter.attempted==meter.returned==148 and r['whole']==r['segmented']
    with pytest.raises(ValueError):m.score_after_barrier([r],m.digest([r]))


def test_unchanged_original_math_for_two_legacy_arms():
    t=fixture.task(0);target=t['root_context']['latent_history'][-1]
    for arm in (m.ARMS[0],m.ARMS[2]):
        a=[];b=[]
        actual=m.optimize(t,0,arm,target,transition(t),a.append)
        expected=m.old.optimize(t,0,arm,target,transition(t),b.append)
        assert actual==expected and a==b


def test_failure_predebit_and_no_retry():
    meter=m.Attempts()
    def fail():raise RuntimeError('fixture failed')
    with pytest.raises(RuntimeError):meter.invoke(2,fail)
    assert meter.attempted==2 and meter.returned==0 and meter.poisoned
    with pytest.raises(ValueError):meter.invoke(1,lambda:[1])


def test_barrier_rejects_missing_reorder_or_changed_hash():
    rows=[dict(root=r,arm=a,whole=[{'v':1},{'v':2}],segmented=[{'v':1},{'v':2}]) for r,a in m.SLOTS]
    seal=m.digest(rows)
    assert all(x['context_json_equal'] for x in m.score_after_barrier(rows,seal))
    rows[0]['whole'][0]['v']=3
    with pytest.raises(ValueError):m.score_after_barrier(rows,seal)
    rows.reverse()
    with pytest.raises(ValueError):m.score_after_barrier(rows,m.digest(rows))


def test_empty_replay_cannot_pass_barrier():
    rows=[dict(root=r,arm=a,whole=[],segmented=[]) for r,a in m.SLOTS]
    with pytest.raises(ValueError):m.score_after_barrier(rows,m.digest(rows))
