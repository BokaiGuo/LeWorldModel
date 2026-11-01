"""Non-neural saved-plan and guard fixtures; no physical/CUDA efficacy evidence."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.spectral_reachability import p3_matched_physical_20261027_v1 as p


@pytest.fixture
def saved():
    root=Path(__file__).resolve().parents[1]/'results'
    def read(path):
        return json.loads((root/path).read_text())
    return (read('P3_ADAPTIVE_RUNTIME_20261027_cuda_v1/P3_ADAPTIVE_RUNTIME_report_20261027.json'),
        read('P3_ADAPTIVE_RUNTIME_20261027_cuda_v1/P3_ADAPTIVE_RUNTIME_protocol_20261027.json'),
        read('P3_ADAPTIVE_RUNTIME_ACCOUNTING_AUDIT_20261027_main_v2/report.json'),
        read('pusht_d2p_metric_program_preflight4_v1_20261026_cuda/report.json'))


def test_real_saved_all_eight_exact_paths_and_variable_endpoints(saved):
    snapshot=deepcopy(saved)
    plans=p.selected_plans(*saved)
    assert len(plans)==8
    assert sorted(x['planned_commands'] for x in plans)==[10]*5+[15]*3
    index={(r['root_sha256'],r['method']):r for r in saved[0]['rows']}
    for plan in plans:
        row=index[plan['root_sha256'],plan['method']]
        assert plan['commands']==row['search']['best_plan']['actions']
        assert len(plan['predictor_context']['latent_history'])==3
        assert len(plan['predictor_context']['action_history'])==2
    for i in (0,2):
        plan=next(x for x in plans if x['root_index']==i and x['method']==p.METHODS[1])
        assert plan['corrected_search_horizon']==plan['planned_horizon_tokens']==2
    assert saved==snapshot


@pytest.mark.parametrize('field',['commands','cost','correction','protocol','cohort','duplicate','path_bool','goal'])
def test_mutations_rejected(saved,field):
    report,protocol,correction,source=deepcopy(saved)
    best=report['rows'][0]['search']['best_plan']
    if field=='commands': best['actions'][0][0]+=.01
    elif field=='cost': best['cost']+=.1
    elif field=='correction': correction['corrections'][0]['actual_horizon']=1
    elif field=='protocol': protocol['protocol']['predict_cap']=148
    elif field=='cohort': source['protocol']['roots']=source['protocol']['roots'][::-1]
    elif field=='duplicate': report['rows'][0]=report['rows'][1]
    elif field=='path_bool': best['primitive_indices'][0]=True
    else: source['rows'][0]['compiled']['goal_vectors'][0][0]+=1
    with pytest.raises((ValueError,KeyError)):
        p.selected_plans(report,protocol,correction,source)


class Env:
    def __init__(self,missing_at=None,terminate_at=None,fail_at=None,interrupt_at=None):
        self.n=0
        self.missing_at=missing_at; self.terminate_at=terminate_at
        self.fail_at=fail_at; self.interrupt_at=interrupt_at
    def render(self):
        rgb=np.zeros((224,224,3),np.uint8)
        if self.n!=self.missing_at: rgb[20:22,20:22]=[65,105,225]
        return rgb
    def step(self,action):
        self.n+=1
        if self.n==self.fail_at: raise RuntimeError('env failed')
        if self.n==self.interrupt_at: raise KeyboardInterrupt('interrupted')
        return None,0.,self.n==self.terminate_at,False,{}


def state(env):
    return np.array([env.n,0,0,0,0],float)


@pytest.mark.parametrize('steps',[10,15])
def test_guard_every_step_no_padding(steps):
    events=[]
    row,rgb=p.guarded_execute(Env(),np.zeros((steps,2)),root_id='root',state_of=state,
        event=lambda v,r:events.append(deepcopy(v)))
    assert row['status']=='completed'
    assert row['completed_commands']==row['attempted_commands']==steps
    assert len(row['guard']['observations'])==steps+1
    assert sum(e['kind']=='attempt' for e in events)==steps
    assert row['actual_endpoint_primitive_steps']==steps


@pytest.mark.parametrize('options,attempts,completed,status',[
    ({'missing_at':0},0,0,'observation_unavailable'),
    ({'missing_at':3},3,3,'observation_unavailable'),
    ({'terminate_at':2},2,2,'environment_terminated'),
    ({'fail_at':4},4,3,'execution_failed'),
    ({'interrupt_at':4},4,3,'execution_failed')])
def test_aborts_keep_attempted_costs(options,attempts,completed,status):
    row,_=p.guarded_execute(Env(**options),np.zeros((15,2)),root_id='root',state_of=state,event=lambda v,r:None)
    assert row['status']==status
    assert row['attempted_commands']==attempts
    assert row['completed_commands']==completed
    assert not row['full_horizon_completed']


def plan():
    return dict(root_index=0,root_sha256='root',method='fixed',selection_sha256='seal',
        planned_horizon_tokens=3,planned_commands=15,corrected_search_horizon=3,
        commands=np.zeros((15,2)).tolist())


def branch(tmp_path,env=None,reset=lambda:dict(identity='same'),evaluate=lambda s:dict(official={},pose={})): 
    return p.physical_branch(plan=plan(),repetition=0,output=tmp_path,reset=reset,
        env=env or Env(),state_of=state,evaluate=evaluate)


def test_physical_persisted_before_encoding_and_encoder_failure_retained(tmp_path):
    row=branch(tmp_path)
    physics=list(tmp_path.glob('*.physical.json'))
    assert len(physics)==1
    before=physics[0].read_bytes()
    def encode(path):
        assert physics[0].read_bytes()==before and path.exists()
        raise RuntimeError('CUDA encoder failed')
    receipts=p.encode_saved([row],tmp_path,encode)
    assert receipts[0]['status']=='encoding_failed'
    assert receipts[0]['attempted_encode_samples']==1
    assert physics[0].read_bytes()==before
    events=[json.loads(s) for s in Path(row['journal_path']).read_text().splitlines()]
    assert len([e for e in events if e['kind']=='attempt'])==15
    assert len(list(tmp_path.glob('*.png')))==16


def test_reset_failure_retained_and_encoding_blocked(tmp_path):
    def reset(): raise ValueError('root mismatch')
    row=branch(tmp_path,reset=reset)
    assert row['status']=='branch_failed' and row['attempted_commands']==0
    assert list(tmp_path.glob('*.physical.json'))
    result=p.encode_saved([row],tmp_path,lambda path:pytest.fail('no encode allowed'))
    assert result[0]['attempted_encode_samples']==0


def test_evaluator_failure_does_not_erase_trace_or_execution_costs(tmp_path):
    def evaluate(state): raise ValueError('label failure')
    row=branch(tmp_path,evaluate=evaluate)
    assert row['status']=='branch_failed'
    assert row['completed_commands']==15 and len(row['trace'])==15


def test_replay_exact_ignores_only_artifact_paths(tmp_path):
    row=branch(tmp_path)
    other=deepcopy(row); other['terminal_rgb_path']='other.png'
    assert p.replay_equal(row,other)
    other['trace'][0]['state'][0]+=1
    assert not p.replay_equal(row,other)
    assert not p.replay_equal({}, {})


@pytest.mark.parametrize('bad',[np.zeros((5,2)),np.zeros((16,2)),np.ones((15,2),bool),
    np.full((15,2),np.nan),np.full((15,2),1.01)])
def test_invalid_paths_rejected_before_environment(bad):
    with pytest.raises(ValueError):
        p.guarded_execute(None,bad,root_id='root',state_of=state,event=lambda v,r:None)
