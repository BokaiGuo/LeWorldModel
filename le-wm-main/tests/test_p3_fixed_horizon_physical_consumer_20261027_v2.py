"""Saved CUDA receipts and fake-environment contract tests, no physics/neural work."""
from copy import deepcopy
import builtins
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.spectral_reachability import p3_fixed_horizon_physical_consumer_20261027_v2 as p

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def saved():
    body,_=p.planning.validate_prepared(ROOT/p.PREPARED,p.PLANNING_PREPARE_SHA)
    report=p.io.decode((ROOT/p.REPORT).read_bytes())
    return body,report


def test_real_all_eight_h3_paths_and_1176_accounting(saved):
    body,report=saved
    plans=p.plans_from_saved(body,report)
    assert len(plans)==8 and sum(x['planning_predict_samples'] for x in plans)==1152
    assert sum(x['verification_predict_samples'] for x in plans)==24
    assert [x['method'] for x in plans]==[m for order in body['arm_order'] for m in order]
    for plan in plans:
        assert len(plan['commands'])==15 and plan['planned_horizon_tokens']==3
        packet=next(x for x in body['packets'] if x['root_sha256']==plan['root_sha256'])
        assert plan['commands']==sum([packet['primitive_bank'][i] for i in plan['primitive_indices']],[])
        assert plan['upstream_predict_samples']==147


@pytest.mark.parametrize('change',['missing','duplicate','foreign','phase','context_tokens','predict_count',
    'vjp','bad_path','bool_path','actions','horizon','binding','event','chi','winner','cost'])
def test_mutations_rejected(saved,change):
    body,original=saved; report=deepcopy(original);row=report['rows'][0]
    if change=='missing':report['rows'].pop()
    elif change=='duplicate':report['rows'][1]=deepcopy(row)
    elif change=='foreign':row['root_sha256']='0'*64
    elif change=='phase':row['prediction_receipt']['phases'][1]['predict_samples']=2
    elif change=='context_tokens':row['processed_action_context_tokens']+=1
    elif change=='predict_count':row['prediction_receipt']['attempted_predict_samples']=146
    elif change=='vjp':row['runtime_action_vjp_rows']=1
    elif change=='bad_path':row['search']['best_plan']['primitive_indices'][0]=16
    elif change=='bool_path':row['search']['best_plan']['primitive_indices'][0]=True
    elif change=='actions':row['search']['best_plan']['actions'][0][0]+=.001
    elif change=='horizon':row['search']['best_plan']['depth']=2
    elif change=='binding':row['prediction_receipt']['binding']['task_id']='different_task'
    elif change=='event':row['search']['events'][0]['parent_path']=[1]
    elif change=='chi':row['search']['signal']['chi']+=.01
    elif change=='winner':row['search']['best_plan']['context']['latent_history'][-1][0]+=.01
    else:row['search']['best_plan']['cost']+=.01
    with pytest.raises((ValueError,KeyError)):
        p.plans_from_saved(body,report)


class Env:
    def __init__(self,missing_at=None,fail_at=None):self.n=0;self.missing_at=missing_at;self.fail_at=fail_at
    def render(self):
        rgb=np.zeros((224,224,3),np.uint8)
        if self.n!=self.missing_at:rgb[20:22,20:22]=[65,105,225]
        return rgb
    def step(self,a):
        self.n+=1
        if self.n==self.fail_at:raise RuntimeError('injected failed step')
        return None,0.,False,False,{}


def fixture_plan():
    return dict(root_index=0,root_sha256='a'*64,method='uniform',selection_sha256='b'*64,
        planned_horizon_tokens=3,planned_commands=15,corrected_search_horizon=3,
        commands=np.zeros((15,2)).tolist(),upstream_predict_samples=147)


def branch(tmp_path,env=None,plan=None,reset=lambda:dict(context='bound')):
    return p.physical_branch(plan=plan or fixture_plan(),output=tmp_path,env=env or Env(),reset=reset,
        state_of=lambda e:np.array([e.n,0,0,0,0,0,0],float),evaluate=lambda state:dict(
            pose={'pose_cost':float(state[0])},official={'official_state_distance':float(state[0])}))


def test_h3_durable_full_path(tmp_path):
    row=branch(tmp_path)
    p.verify_branch(fixture_plan(),row)
    assert row['completed_commands']==15 and row['full_horizon_completed']
    assert row['endpoint_fresh'] and row['endpoint_scope']=='full_H3'
    assert len(list(tmp_path.glob('*.png')))==16


@pytest.mark.parametrize('kwargs,attempts,returns,fresh',[
    ({'missing_at':0},0,0,True),({'missing_at':6},6,6,True),({'fail_at':7},7,6,False)])
def test_guard_or_execution_abort_never_promoted(tmp_path,kwargs,attempts,returns,fresh):
    row=branch(tmp_path,env=Env(**kwargs));p.verify_branch(fixture_plan(),row)
    assert row['attempted_commands']==attempts and row['completed_commands']==returns
    assert not row['full_horizon_completed'] and row['endpoint_fresh']==fresh
    assert row['endpoint_scope']!='full_H3'


def test_reject_old_h2_before_environment_access(tmp_path):
    plan=fixture_plan();plan['commands']=plan['commands'][:10];plan['planned_commands']=10
    with pytest.raises(ValueError):branch(tmp_path,plan=plan)


def test_journal_tampering_rejected(tmp_path):
    row=branch(tmp_path)
    with Path(row['journal_path']).open('a') as stream:stream.write('{}\n')
    with pytest.raises(ValueError,match='hash'):p.verify_branch(fixture_plan(),row)


def test_reset_failure_saved_and_no_stale_endpoint(tmp_path):
    def fail():raise RuntimeError('wrong root RGB')
    row=branch(tmp_path,reset=fail)
    assert row['status']=='branch_failed' and not row['endpoint_fresh']
    assert len(list(tmp_path.glob('*.physical.json')))==1


def test_optional_encoding_all_branches_after_physics_saved(tmp_path):
    row=branch(tmp_path,env=Env(missing_at=6))
    p.write(tmp_path/'physical_report.json',{'rows':[row]})
    before=next(tmp_path.glob('*.physical.json')).read_bytes()
    def setup():
        def encode(path):
            assert next(tmp_path.glob('*.physical.json')).read_bytes()==before
            return [0.]*192
        return encode
    receipts=p.optional_encodings([row],tmp_path,setup)
    assert receipts[0]['attempted_encode_samples']==1 and receipts[0]['status']=='completed'
    assert receipts[0]['endpoint_scope']=='guard_partial_or_terminated'


def test_encoding_failure_and_missing_endpoint_are_retained(tmp_path):
    row=branch(tmp_path,env=Env(fail_at=7))
    p.write(tmp_path/'physical_report.json',{'rows':[row]})
    def setup():return lambda path: (_ for _ in ()).throw(AssertionError('must not encode stale image'))
    receipts=p.optional_encodings([row],tmp_path,setup)
    assert receipts[0]['status']=='blocked_stale_endpoint' and receipts[0]['attempted_encode_samples']==0


def test_materialize_all_eight_slots(tmp_path,saved):
    plans=p.plans_from_saved(*saved)
    rows=[];p.materialize_missing(plans,rows,tmp_path)
    assert len(rows)==8 and all(r['status']=='blocked' for r in rows)
    assert all(r['physical_env_step_attempts']==0 and not r['endpoint_fresh'] for r in rows)


def test_prepare_and_validate_without_torch_or_physics(tmp_path,monkeypatch):
    original=builtins.__import__
    def no_runtime(name,*args,**kw):
        assert name.split('.')[0] not in {'torch','gymnasium','pymunk','mujoco'},name
        return original(name,*args,**kw)
    monkeypatch.setattr(builtins,'__import__',no_runtime)
    out=tmp_path/'prepare';r=p.prepare(ROOT,out)
    h=hashlib.sha256((out/'prepare.json').read_bytes()).hexdigest()
    protocol=p.validate_prepared(out,h)
    assert len(protocol['plans'])==8 and r['planned_selected_path_steps']==120
    assert r['new_predict_samples']==r['new_physical_steps']==0 and not r['model_loaded']
    with pytest.raises(ValueError,match='NEW'):p.prepare(ROOT,out)
    with pytest.raises(ValueError,match='SHA'):p.validate_prepared(out,'0'*64)
