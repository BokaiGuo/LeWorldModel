"""Metadata/guard/durability tests only; no GPU or real trajectory execution."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import p2_matched_sampling_physical_20261027_v1 as p


PLANNING=Path('results/p2_matched_sampling_20261027_v1_cuda_v1')


def test_real_saved_cuda_report_all_eight_selections_and_provenance():
    _,_,_,_,bindings,plans,upstream,refs,_=p.prepare(PLANNING)
    assert len(plans)==len({(r['root_sha256'],r['method']) for r in plans})==8
    assert len(refs)==4
    for i in range(4):
        rows=[r for r in plans if r['root_index']==i]
        assert [r['method'] for r in rows]==(list(p.sampling.ARMS) if i%2==0 else list(p.sampling.ARMS[::-1]))
    for plan in plans:
        assert len(plan['commands'])==plan['planned_commands']==10
        assert plan['planned_horizon_tokens']==2
        assert plan['root']['root_rgb_sha256']==refs[plan['root_sha256']]['history_rgb_sha256']
        assert plan['root']['goal_rgb_sha256']==refs[plan['root_sha256']]['goal_rgb_sha256']
    assert str((PLANNING/'report.json').resolve()) in bindings


@pytest.mark.parametrize('field',['source_seal','fixture','commands','cost','arm'])
def test_real_saved_tampering_rejected(field):
    _,protocol,prepared,source=p.sampling.load_prepared(PLANNING)
    report=json.loads((PLANNING/'report.json').read_text())
    arm=report['rows'][0]['rows'][0]
    if field=='source_seal':report['source_seals'][0]['manifest_sha256']='0'*64
    if field=='fixture':arm['fixture_only']=True
    if field=='commands':arm['witness']['receipt']['actions'][0][0]+=.01
    if field=='cost':arm['selected_model_cost']+=1
    if field=='arm':report['rows'][0]['rows'][1]['arm']='cem'
    with pytest.raises(ValueError):p.plans_from_saved(report,protocol,source)


class Env:
    def __init__(self,missing_at=None,fail_at=None,terminate_at=None):
        self.n=0;self.missing_at=missing_at;self.fail_at=fail_at;self.terminate_at=terminate_at
    def render(self):
        rgb=np.zeros((224,224,3),np.uint8)
        if self.n!=self.missing_at:rgb[20:22,20:22]=[65,105,225]
        return rgb
    def step(self,action):
        self.n+=1
        if self.n==self.fail_at:raise RuntimeError('fixture env error')
        return None,0.,self.n==self.terminate_at,False,{}
    def close(self):return None


def state(env):return np.asarray([env.n,0,0,0,0,0,0],float)


def plan(i=0,arm='cem'):
    return dict(root_index=i,root_sha256=str(i)*64,method=arm,selection_sha256='seal',
        planned_horizon_tokens=2,planned_commands=10,corrected_search_horizon=2,
        commands=np.zeros((10,2)).tolist())


def branch(tmp_path,env=None,reset=lambda:{'identity':'fixture'},evaluate=lambda s:{'official':{},'pose':{}}):
    return p.physical_branch(plan=plan(),output=tmp_path,reset=reset,
        env=env or Env(),state_of=state,evaluate=evaluate)


@pytest.mark.parametrize('options,attempted,completed,status',[
    ({},10,10,'completed'),({'missing_at':0},0,0,'observation_unavailable'),
    ({'missing_at':3},3,3,'observation_unavailable'),({'fail_at':4},4,3,'execution_failed'),
    ({'terminate_at':2},2,2,'environment_terminated')])
def test_guard_trace_costs_and_artifacts_persisted(tmp_path,options,attempted,completed,status):
    row=branch(tmp_path,Env(**options))
    assert row['status']==status
    assert row['attempted_commands']==attempted and row['completed_commands']==completed
    assert len(row['trace'])==completed
    saved=list(tmp_path.glob('*.physical.json'))
    assert len(saved)==1 and json.loads(saved[0].read_text())==row
    events=[json.loads(s) for s in Path(row['journal_path']).read_text().splitlines()]
    assert sum(e['kind']=='attempt' for e in events)==attempted


def test_reset_mismatch_retained_with_zero_actions(tmp_path):
    def fail():raise ValueError('official history RGB mismatch')
    row=branch(tmp_path,reset=fail)
    assert row['status']=='branch_failed' and row['attempted_commands']==0
    assert row['terminal_rgb_path'] is None
    assert list(tmp_path.glob('*.physical.json'))


def test_evaluator_error_keeps_full_trace(tmp_path):
    def fail(s):raise ValueError('fixture evaluator')
    row=branch(tmp_path,evaluate=fail)
    assert row['status']=='branch_failed' and row['completed_commands']==10 and len(row['trace'])==10


def test_physical_report_precedes_any_encoder_setup_and_failures_preserved(tmp_path):
    row=branch(tmp_path)
    before=list(tmp_path.glob('*.physical.json'))[0].read_bytes()
    with pytest.raises(ValueError,match='persist'):p.optional_encodings([row],tmp_path,lambda:None)
    p.write(tmp_path/'physical_report.json',dict(rows=[row]))
    def setup():
        assert list(tmp_path.glob('*.physical.json'))[0].read_bytes()==before
        def encode(path):raise RuntimeError('fixture CUDA encode failure')
        return encode
    receipts=p.optional_encodings([row],tmp_path,setup)
    assert receipts[0]['status']=='encoding_failed' and receipts[0]['attempted_encode_samples']==1
    assert list(tmp_path.glob('*.physical.json'))[0].read_bytes()==before


def test_cuda_setup_failure_materialized_per_branch_without_neural_work(tmp_path):
    rows=[]
    p.materialize_missing([plan(i,arm) for i in range(4) for arm in ('cem','random')],rows,tmp_path)
    assert len(rows)==8 and len(list(tmp_path.glob('*.physical.json')))==8
    p.write(tmp_path/'physical_report.json',dict(rows=rows))
    def fail():raise RuntimeError('allocation failed')
    result=p.optional_encodings(rows,tmp_path,fail)
    assert len(result)==8 and all(r['status']=='encoding_setup_failed' and r['attempted_encode_samples']==0 for r in result)
    assert len(list(tmp_path.glob('*.encoding.json')))==8


@pytest.mark.parametrize('raw',[np.zeros((5,2)),np.zeros((15,2)),np.full((10,2),np.nan),np.ones((10,2),bool)])
def test_h2_only_before_environment_access(tmp_path,raw):
    value=plan();value['commands']=raw
    with pytest.raises(ValueError):p.physical_branch(plan=value,output=tmp_path,
        reset=lambda:pytest.fail('environment touched'),env=None,state_of=None,evaluate=None)


def test_whole_stack_rgb_identity_and_changed_goal_rejected():
    env=Env();frames=[env.render() for _ in range(3)];goal=env.render()
    import hashlib
    ref=dict(history_rgb_sha256=hashlib.sha256(np.stack(frames).tobytes()).hexdigest(),
             goal_rgb_sha256=hashlib.sha256(goal.tobytes()).hexdigest())
    assert p.verify_rgb_context(frames,goal,ref)==ref
    goal=goal.copy();goal[0,0,0]=1
    with pytest.raises(ValueError):p.verify_rgb_context(frames,goal,ref)


def test_model_free_run_never_calls_cuda_or_loads_model(tmp_path,monkeypatch):
    env=Env();env.unwrapped=env
    env.variation_space={'agent':{'color':SimpleNamespace(value=np.asarray([65,105,225]))}}
    env.eval_state=lambda goal,s:(bool(np.linalg.norm(np.asarray(goal)[:4]-s[:4])<20),float(np.linalg.norm(np.asarray(goal)-s)))
    roots=[dict(id=str(i)*64,goal=np.zeros(7)) for i in range(4)]
    plans=[plan(i,arm) for i in range(4) for arm in ('cem','random')]
    frames=[env.render() for _ in range(3)];goal=env.render()
    import hashlib
    refs={r['id']:dict(history_rgb_sha256=hashlib.sha256(np.stack(frames).tobytes()).hexdigest(),
                      goal_rgb_sha256=hashlib.sha256(goal.tobytes()).hexdigest()) for r in roots}
    def reset(e,root,history_len):
        e.n=0
        return frames,np.zeros((3,2)),goal
    def forbidden(*a,**k):pytest.fail('neural/CUDA entry in model-free physical execution')
    base=SimpleNamespace(gym=SimpleNamespace(make=lambda *a,**k:env),
        collect_nonterminal_boundary_contact_roots=lambda *a,**k:roots,
        root_fingerprint=lambda r:r['id'],require_cuda_allocation=forbidden,load_compatible_pretrained=forbidden,
        physical=SimpleNamespace(_reset_and_warmup=reset,_pose_label=lambda s,g:dict(pose_cost=float(s[0]))),
        p0=SimpleNamespace(packets=SimpleNamespace(state_of=state)))
    checked=SimpleNamespace(sources_before_allocate=lambda *a:[])
    monkeypatch.setattr(p,'prepare',lambda *a:(base,checked,{}, {},{},plans,
        {'protocol':{'root_seed':0}},refs,{'policy':'unused'}))
    output=tmp_path/p.TAG
    p.run(planning='fixture',output=output)
    report=json.loads((output/'report.json').read_text())
    assert report['status']=='calibration_recorded'
    assert report['new_predict_calls']==report['attempted_terminal_encode_samples']==0
    assert report['host'] is None and not report['encodings']
    assert report['selected_path_env_step_returns']==80
    assert all(report['gates'].values())
    assert len(list(output.glob('*.physical.json')))==8
    with pytest.raises(FileExistsError):p.run(planning='fixture',output=output)
