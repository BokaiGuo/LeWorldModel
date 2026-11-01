"""Non-neural algebra and fake-environment tests. Never execute physics."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

MOD='experiments.spectral_reachability.p2_grasp_h1_guarded_physical_20261027_v1'


def m(): return importlib.import_module(MOD)


def test_import_never_imports_neural_or_physics():
    subprocess.run([sys.executable,'-c',f'import {MOD}; import sys; assert not set(["torch","gymnasium","pymunk"]) & set(sys.modules)'],check=True)


def plan(outside=False):
    mod=m();a=[[0.]*10 for _ in range(3)]
    if outside: a[2][9]=10.
    p=dict(schema=mod.NAME+'.plan',root_index=0,root_sha256='a'*64,method='grasp_h1',
        planning_report_sha256='b'*64,planned_horizon_tokens=3,
        corrected_search_horizon=3,input_sha256='c'*64,goal_rgb_sha256='d'*64,
        **mod.planning.export_actions(a))
    p['dispatch_status']='eligible' if p['physical_box_supported'] else 'rejected_raw_out_of_box'
    p['selection_sha256']=mod.io.sha(mod.io.canonical(p).encode())
    return p


class Env:
    def __init__(self,missing=None,fail=None,render_fail=None,terminate=None):
        self.n=0;self.missing=missing;self.fail=fail;self.render_fail=render_fail;self.terminate=terminate
    def render(self):
        if self.n==self.render_fail: raise RuntimeError('render failed')
        rgb=np.zeros((224,224,3),np.uint8)
        if self.n!=self.missing: rgb[20:23,20:23]=[65,105,225]
        return rgb
    def step(self,a):
        self.n+=1
        if self.n==self.fail: raise RuntimeError('step failed')
        return None,0.,self.n==self.terminate,False,{}


def branch(tmp_path,env=None,p=None,reset=lambda:dict(context='same')):
    mod=m()
    return mod.physical_branch(plan=p or plan(),output=tmp_path,reset=reset,env=env or Env(),
        state_of=lambda e:np.array([e.n,0,0,0,0,0,0]),evaluate=lambda s:dict(pose=dict(pose_cost=float(s[0])),official=dict(official_state_distance=float(s[0]))))


def test_out_of_box_whole_plan_rejected_without_any_env_access(tmp_path):
    class NoAccess:
        def __getattr__(self,k): raise AssertionError('no env access for rejected plan')
    p=plan(True)
    r=branch(tmp_path,env=NoAccess(),p=p,reset=lambda:(_ for _ in ()).throw(AssertionError('reset forbidden')))
    assert r['status']=='rejected_raw_out_of_box'
    assert r['physical_env_step_attempts']==r['physical_env_step_returns']==0
    assert r['commands']==p['commands'] and not r['endpoint_fresh']
    assert not list(tmp_path.glob('*.png'))
    m().verify_branch(p,r)


def test_complete_fifteen_not_old_ten(tmp_path):
    r=branch(tmp_path)
    assert r['completed_commands']==15 and r['full_horizon_completed']
    assert r['endpoint_scope']=='full_15' and r['endpoint_fresh']
    assert len(list(tmp_path.glob('*.png')))==16
    m().verify_branch(plan(),r)


@pytest.mark.parametrize('kw,attempts,returns,fresh',[
    ({'missing':0},0,0,True),({'missing':6},6,6,True),
    ({'fail':7},7,6,False),({'render_fail':7},7,7,False),({'terminate':4},4,4,True)])
def test_partial_and_failures_are_never_full_or_stale_endpoint(tmp_path,kw,attempts,returns,fresh):
    r=branch(tmp_path,env=Env(**kw));m().verify_branch(plan(),r)
    assert r['physical_env_step_attempts']==attempts and r['physical_env_step_returns']==returns
    assert r['endpoint_fresh']==fresh and not r['full_horizon_completed']


def test_reset_failure_durable_and_no_endpoint(tmp_path):
    r=branch(tmp_path,reset=lambda:(_ for _ in ()).throw(RuntimeError('wrong frame')))
    assert r['status']=='branch_failed' and not r['endpoint_fresh']
    assert r['physical_env_step_attempts']==0 and r['failures']
    m().verify_branch(plan(),r)


@pytest.mark.parametrize('field,value', [('commands',[[0.,0.]]*10),('planned_commands',10),('dispatch_status','eligible'),('selection_sha256','0'*64)])
def test_malformed_plan_cannot_dispatch(tmp_path,field,value):
    p=plan(True) if field=='dispatch_status' else plan();p[field]=value
    with pytest.raises(ValueError): branch(tmp_path,p=p)


def test_journal_tampering_fails(tmp_path):
    r=branch(tmp_path);path=Path(r['journal_path']);path.write_text(path.read_text()+'{}\n')
    with pytest.raises(ValueError,match='journal'): m().verify_branch(plan(),r)


def test_same_root_context_goal_and_normalized_history():
    mod=m();packet=dict(root_sha256='a'*64,history_rgb_sha256='b'*64,goal_rgb_sha256='c'*64,
        input_sha256='d'*64,history_actions=[[-mod.planning.MEAN[i%2]/mod.planning.STD[i%2] for i in range(10)]]*2)
    frames=[np.zeros((224,224,3),np.uint8)]*3;goal=np.zeros((224,224,3),np.uint8)
    packet['history_rgb_sha256']=mod.rgb_sha(np.stack(frames));packet['goal_rgb_sha256']=mod.rgb_sha(goal)
    out=mod.check_context(packet,frames,np.zeros((3,2),np.float32),goal)
    assert out['input_sha256']=='d'*64
    frames=list(frames);frames[-1]=frames[-1].copy();frames[-1][0,0]=1
    with pytest.raises(ValueError,match='RGB'): mod.check_context(packet,frames,np.zeros((3,2)),goal)
    with pytest.raises(ValueError,match='history'): mod.check_context(packet,[np.zeros((224,224,3),np.uint8)]*3,np.ones((3,2)),goal)


def test_meter_separates_warmup_steps_and_attempt_without_return():
    mod=m();events=[];e=mod.StepMeter(Env(fail=3),events.append)
    e.phase='warmup';e.step([0,0]);e.step([0,0]);e.phase='selected'
    with pytest.raises(RuntimeError): e.step([0,0])
    assert e.counts['warmup']==dict(attempts=2,returns=2)
    assert e.counts['selected']==dict(attempts=1,returns=0)
    assert len([x for x in events if x['kind']=='env_step_attempt'])==3


def test_pairs_only_full_matching_context_get_difference():
    mod=m();roots=['a','b','c','d'];rows=[]
    for root in roots:
        for method in mod.planning.METHODS:
            rows.append(dict(root_sha256=root,method=method,status='completed',context_identity={'same':root},
                endpoint_fresh=True,full_horizon_completed=True,physical_env_step_returns=15,pose=dict(pose_cost=1 if method=='grasp_h1' else 2)))
    rows[2].update(status='observation_unavailable',full_horizon_completed=False,physical_env_step_returns=6)
    rows[3].update(status='observation_unavailable',full_horizon_completed=False,physical_env_step_returns=6)
    rows[4].update(status='rejected_raw_out_of_box',endpoint_fresh=False,full_horizon_completed=False,physical_env_step_returns=0)
    out=mod.comparisons(roots,rows[::-1])
    assert [p['classification'] for p in out]==['full-path','guard-partial','unmatched','full-path']
    assert out[0]['grasp_minus_action_adam_pose_cost']==-1
    assert out[1]['grasp_minus_action_adam_pose_cost'] is None and out[2]['grasp_minus_action_adam_pose_cost'] is None


def test_all_eight_missing_or_rejected_materialized(tmp_path):
    mod=m();plans=[]
    for i in range(4):
        for method in mod.planning.METHODS:
            p=plan(i==0);p.update(root_index=i,root_sha256=str(i)*64,method=method)
            p.pop('selection_sha256');p['selection_sha256']=mod.io.sha(mod.io.canonical(p).encode());plans.append(p)
    rows=[];mod.materialize_missing(plans,rows,tmp_path)
    assert len(rows)==8 and sum(r['status']=='rejected_raw_out_of_box' for r in rows)==2
    assert all(r['physical_env_step_attempts']==0 for r in rows)


def test_prepare_refuses_bad_source_sha_without_importing_physics(tmp_path):
    path=Path(__file__).resolve().parents[1]/m().DEFAULT_REPORT
    with pytest.raises(ValueError,match='SHA'): m().prepare(Path(__file__).resolve().parents[1],tmp_path/'new',path,'0'*64)


@pytest.fixture(scope='module')
def saved():
    mod=m();root=Path(__file__).resolve().parents[1]
    body,_=mod.planning.validate_prepared(root/mod.PREPARED,mod.PLANNING_PREPARE_SHA)
    path=root/mod.DEFAULT_REPORT
    report=mod.io.decode(path.read_bytes())
    return body,report,mod.planning.digest(path)


def test_real_eight_slots_four_rejects_keep_exact_unclipped_actions(saved):
    mod=m();body,report,h=saved
    plans=mod.plans_from_saved(body,report,h)
    assert len(plans)==8 and sum(p['dispatch_status']=='rejected_raw_out_of_box' for p in plans)==4
    rows=mod.io.index(report['rows'],('root_sha256','method'))
    for p in plans:
        assert p['commands']==rows[p['root_sha256'],p['method']]['plan']['commands']
        assert p['planned_commands']==15


@pytest.mark.parametrize('mutation',['missing','duplicate','schema','raw','phase','tokens','backward','goal','configuration'])
def test_real_receipt_mutations_rejected(saved,mutation):
    mod=m();body,original,h=saved;report=deepcopy(original);row=report['rows'][0]
    if mutation=='missing':report['rows'].pop()
    elif mutation=='duplicate':report['rows'][1]=deepcopy(row)
    elif mutation=='schema':report['schema']='oldphysicalreport'
    elif mutation=='raw':row['plan']['commands'][0][0]+=.01
    elif mutation=='phase':row['transition_counts']['verification']['completed']=2
    elif mutation=='tokens':row['module_work']['predictor']['requested_context_tokens']+=1
    elif mutation=='backward':row['receipt']['backward_completed']=899
    elif mutation=='goal':row['verification'][0][0]+=.01
    else:row['receipt']['config']['lr_s']=.2
    with pytest.raises(ValueError):mod.plans_from_saved(body,report,h)


def test_actual_large_journal_streamed_with_all_actions_and_module_coordinates(saved):
    mod=m();root=Path(__file__).resolve().parents[1];body,report,h=saved
    receipt=mod.verify_planning_journal((root/mod.DEFAULT_REPORT).parent/'events.jsonl',report)
    assert receipt['transitions']==21648 and receipt['arms']==8
    assert receipt['bytes']>32_000_000


def test_offline_actual_prepare_validates_all_sources_and_eight_plans(tmp_path,saved):
    mod=m();root=Path(__file__).resolve().parents[1];out=tmp_path/'prepared'
    subprocess.run([sys.executable,'-c',f'from {MOD} import main; import sys; main(); assert not set(["torch","gymnasium","pymunk"]) & set(sys.modules)',
        '--prepare','--planning-report',str(root/mod.DEFAULT_REPORT),'--planning-report-sha256',saved[2],'--output',str(out)],
        check=True,capture_output=True,text=True)
    p=mod.validate_prepared(out,mod.planning.digest(out/'prepare.json'))
    assert len(p['plans'])==8 and p['eligible_arms']==4
    assert p['max_selected_path_steps']==p['max_warmup_steps']==60
    assert p['new_model_loads']==p['new_predict_samples']==p['new_neural_encodes']==0
    with pytest.raises(ValueError,match='NEW'):mod.prepare(root,out,root/mod.DEFAULT_REPORT,saved[2])
    with pytest.raises(ValueError,match='SHA'):mod.validate_prepared(out,'0'*64)


def test_actual_source_zero_warmup_normalization_matches_all_packets(saved):
    mod=m()
    raw=np.zeros((2,5,2),np.float32)
    expected=((raw-np.array(mod.planning.MEAN,np.float32))/np.array(mod.planning.STD,np.float32)).reshape(2,10)
    for packet in saved[0]['packets']:
        np.testing.assert_allclose(expected,packet['history_actions'],rtol=0,atol=2e-6)


def test_step_budget_blocks_before_environment_dispatch():
    mod=m();env=Env();meter=mod.StepMeter(env,lambda e:None,limits={'selected':1});meter.phase='selected'
    meter.step([0.,0.])
    with pytest.raises(ValueError,match='budget'):meter.step([0.,0.])
    assert env.n==1 and meter.counts['selected']==dict(attempts=1,returns=1)


def test_meter_budget_journal_is_root_arm_bound():
    mod=m();events=[];meter=mod.StepMeter(Env(),events.append)
    meter.binding=dict(root_sha256='a'*64,method='grasp_h1');meter.phase='warmup'
    meter.step([0.,0.])
    assert all(e['root_sha256']=='a'*64 and e['method']=='grasp_h1' for e in events)
