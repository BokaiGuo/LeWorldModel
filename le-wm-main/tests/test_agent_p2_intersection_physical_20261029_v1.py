"""RAM-only physical callback and saved transport tests; no gym/neural."""
from copy import deepcopy
from functools import lru_cache
import hashlib
from pathlib import Path
import pytest

from experiments.spectral_reachability import agent_p2_intersection_physical_20261029_v1 as m
from test_agent_p2_common_resource_physical_20261029_v1 import fixture as old_fixture, MemoryStore, Env, fake_place, h, reseal


def fixture(missing=()):
    old,data,c,u=old_fixture(missing)
    c['schema']=m.PRODUCER+'.prepare'
    receipt=dict(schema=m.PRODUCER+'.consume',status='all6_joint_box_saved_verified_not_scientific_admission',
        report_sha256=h('planner report'),fixed_denominator=6,plans=deepcopy(old['model_witnesses']),
        accounts=[dict(root=r,arm=a,used={'predict_API':123}) for r,_,a in m.ORDER],
        missing_plan_indices=list(missing),partial_arm_indices=[],new_neural_calls=0,new_physics_calls=0,
        training=0,physical_release=False,P1_admission=False,P2_mechanism=False,equal_FLOPs=False,actual_cost_equality=False)
    args=dict(planner_output='/RAM/planner',planner_report_sha=h('planner report'),planner_receipt='/RAM/receipt.json',
        planner_receipt_sha=h('receipt'),planner_source_sha=h('source'))
    return m.assemble(args,c,receipt,data),data


def execute(fault=None,at=11,missing=()):
    p,data=fixture(missing);store=MemoryStore();envs=[]
    def factory():
        e=Env(fault=fault,at=at);envs.append(e);return e
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(m,'place',fake_place);mp.setattr(m,'validate_env',lambda _:None)
        mp.setattr(m,'sha',lambda path:hashlib.sha256(store.files[Path(path).name]).hexdigest())
        r=m.execute(p,data,store,factory,authorization=m.AUTH,prepared='/RAM/prepare.json',prepare_sha256=h('prepare'))
    return r,store,p,data,envs


@lru_cache(maxsize=1)
def complete(): return execute()


def test_full6_real_native_callback_reconstruction_and_census():
    r,s,p,_,envs=complete()
    assert r['status']==m.RECORDED,r['errors']
    assert len(m.validate_saved(r,s.events,p))==6
    assert len(envs)==6 and r['counts']=={k:dict(attempted=v,completed=v) for k,v in m.contract()['caps'].items()}
    assert len([n for n in s.files if n.endswith('.png')])==486
    assert all(row['own_returned']==15 and not row['missing_checkpoints'] for row in r['rows'])


def test_saved_consume_png_artifacts_and_census(monkeypatch,tmp_path):
    r,s,p,data,_=complete()
    for name,body in s.files.items(): (tmp_path/name).write_bytes(body)
    monkeypatch.setattr(m,'verify',lambda *args:(deepcopy(p),data))
    result=m.consume(str(tmp_path),hashlib.sha256(s.files['report.json']).hexdigest())
    assert result['saved_PNG_verified']==486 and result['fixed_denominator']==6 and not result['physical_release']
    (tmp_path/'extra').write_bytes(b'bad')
    with pytest.raises(ValueError):m.consume(str(tmp_path),hashlib.sha256(s.files['report.json']).hexdigest())


@pytest.mark.parametrize('field',['action','context','order','account','unknown','reward','count'])
def test_tampering_rejected(field):
    orig,s,p,_,_=complete();r=deepcopy(orig);p=deepcopy(p);events=deepcopy(s.events)
    if field=='action':p['plans'][0]['commands'][0][0]+=0.01
    elif field=='context':p['producer_receipt']['plans'][0]['contexts'][0]['action_history'][1][0]+=0.01
    elif field=='order':p['compiled']['root_arm_order'].reverse()
    elif field=='account':p['producer_receipt']['accounts'][0]['arm']='grasp'
    elif field=='unknown':events.insert(-1,dict(kind='unknown'))
    elif field=='reward':next(e for e in events if e['kind']=='returned_transition' and e['phase']=='plan')['reward']+=1
    else:r['counts']['physics_step']['completed']-=1
    reseal(events,r)
    with pytest.raises((ValueError,KeyError,StopIteration)):m.validate_saved(r,events,p)


def test_missing_and_native_stop_no_padding():
    r,s,p,_,envs=execute(fault='truncate',at=15,missing=(2,))
    assert r['status']==m.RECORDED and len(m.validate_saved(r,s.events,p))==6 and len(envs)==5
    assert r['rows'][2]['status']=='export_missing' and all(x['own_returned']==5 for x in r['rows'] if x['plan'])
    assert all(15 in x['missing_checkpoints'] for x in r['rows'] if x['plan'])


def test_failed_execution_retains_six():
    r,s,p,_,_=execute(fault='throw',at=11)
    assert r['status']=='partial_or_failed' and len(r['rows'])==6 and r['errors']
    assert len(m.validate_saved(r,s.events,p))==6


def test_authorization_before_io(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('unexpected input read'))
    with pytest.raises(ValueError):m.run('/RAM/p',h('p'),'/RAM/output',lambda:pytest.fail('env'),authorization='bad')
