"""Saved CUDA receipts and fake-environment contract tests, no physics/neural work."""
from copy import deepcopy
import builtins
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import p3_action_jacobian_physical_consumer_20261027_v3 as p

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
    return dict(schema=p.NAME+'.plan',planning_report_sha256=p.REPORT_SHA,root_index=0,root_sha256='a'*64,method='uniform',selection_sha256='b'*64,
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


@pytest.mark.parametrize('mutation',['schema','report_cost','row_cost','jacobian','pipeline','inherited_census'])
def test_v3_schema_and_inherited_unknown_cost_cannot_be_relabeled(saved,mutation):
    body,original=saved;report=deepcopy(original);row=report['rows'][0]
    if mutation=='schema':report['schema']='p3_fixed_horizon_budget_schedule.v2.report'
    elif mutation=='report_cost':report['inherited_cost']['predict_samples']=0
    elif mutation=='row_cost':row['inherited_cost']['vjp_directions']=0
    elif mutation=='jacobian':row['new_jacobian_materializations']=1
    elif mutation=='pipeline':report['total_pipeline_cost_equivalent']=True
    else:report['inherited_arrays_charged_once']=8
    with pytest.raises(ValueError):p.plans_from_saved(body,report)


def test_ties_keep_eight_v3_plans_and_unknown_inherited_cost(saved):
    plans=p.plans_from_saved(*saved)
    assert len(plans)==8 and all(x['schema']==p.NAME+'.plan' for x in plans)
    assert all(x['planning_report_sha256']==p.REPORT_SHA for x in plans)
    assert all(x['inherited_cost']['predict_samples'] is None for x in plans)
    assert plans[-1]['commands']==plans[-2]['commands']  # still two independent execution slots


def test_v2_physical_plan_rejected_before_environment(tmp_path):
    plan=fixture_plan();plan['schema']='p3_fixed_horizon_physical_consumer_20261027_v2.plan'
    with pytest.raises(ValueError,match='v3'):
        branch(tmp_path,plan=plan,reset=lambda:(_ for _ in ()).throw(AssertionError('no env access')))


def test_stale_endpoint_delta_not_reported_as_actual(tmp_path):
    first=branch(tmp_path,plan=fixture_plan())
    other=fixture_plan();other['method']='dynamic'
    second=branch(tmp_path,env=Env(fail_at=7),plan=other)
    pair=p.comparisons([fixture_plan(),other],[first,second])[0]
    assert pair['classification']=='unmatched' and pair['dynamic_minus_uniform_pose_cost'] is None


def test_eight_tied_branches_each_encode_once_metadata_only(tmp_path,saved):
    plans=p.plans_from_saved(*saved);rows=[];calls=[]
    for plan in plans:
        row=branch(tmp_path,plan=plan);p.verify_branch(plan,row);rows.append(row)
    p.write(tmp_path/'physical_report.json',dict(schema=p.NAME+'.physical_report',rows=rows))
    def setup():
        def encode(path):calls.append(str(path));return [0.]*192
        return encode
    receipts=p.optional_encodings(rows,tmp_path,setup)
    assert len(receipts)==len(calls)==len(set(calls))==8
    assert sum(r['physical_env_step_returns'] for r in rows)==120
    assert all(r['status']=='completed' for r in receipts)


def test_execute_uses_v3_report_schema_and_preserves_cost_metadata_only(tmp_path,monkeypatch,saved):
    """Full new entrypoint with metadata stubs; never import a simulator/model."""
    plans=p.plans_from_saved(*saved);roots=[r for r in saved[0]['roots']]
    env=Env();env.unwrapped=env
    def eval_state(goal,state):return False,0.
    env.eval_state=eval_state
    env.variation_space={'agent':{'color':SimpleNamespace(value=np.array([65,105,225]))}}
    env.close=lambda:None
    protocol=dict(plans=plans,roots=roots,root_seed=1,root_oversample_factor=4,
        planning_protocol_id=saved[1]['protocol_id'],references={r:{} for r in roots},
        sources_sha256={str(Path(__file__).resolve()):p.durable.digest(__file__)})
    monkeypatch.setattr(p,'validate_prepared',lambda *a:protocol)
    def warmup(environment,root,history_len):
        environment.n=0
        return [environment.render()]*3,np.zeros((3,2)),environment.render()
    base=SimpleNamespace(gym=SimpleNamespace(make=lambda *a,**kw:env),
        collect_nonterminal_boundary_contact_roots=lambda *a,**kw:[{'sha':r,'goal':[0.]*7} for r in roots],
        root_fingerprint=lambda r:r['sha'],
        physical=SimpleNamespace(_reset_and_warmup=warmup,_pose_label=lambda s,g:{'pose_cost':float(s[0])}),
        p0=SimpleNamespace(packets=SimpleNamespace(state_of=lambda e:np.array([e.n,0,0,0,0,0,0]))))
    feedback=SimpleNamespace(verify_rgb_context=lambda *a:{'bound':'metadata'},official_checked=lambda *a:{'distance':0.})
    prefix='experiments.spectral_reachability.'
    monkeypatch.setitem(sys.modules,prefix+'audit_pusht_d2p_goal_agent_intervention',base)
    monkeypatch.setitem(sys.modules,prefix+'run_pusht_d2p_budget_guard_cuda',SimpleNamespace())
    monkeypatch.setitem(sys.modules,prefix+'run_pusht_sampling_comparison_20261027_v2',feedback)
    out=tmp_path/'fake_execute'
    report=p.execute_physics(prepared=tmp_path,prepare_sha256='metadata fixture',output=out)
    assert report['schema']==p.NAME+'.report' and len(report['rows'])==8
    assert report['physical_env_step_attempts']==report['physical_env_step_returns']==120
    assert report['inherited_cost']==p.planning.inherited_cost()
    assert report['upstream_predict_samples_charged_once']==1176 and report['new_predict_calls']==0
    assert report['total_pipeline_cost_equivalent'] is False
    assert p.io.decode((out/'physical_report.json').read_bytes())['schema']==p.NAME+'.physical_report'
