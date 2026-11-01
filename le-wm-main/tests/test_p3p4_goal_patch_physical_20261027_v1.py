"""Metadata/saved-array and fake-env tests; no neural or simulator execution."""
import builtins
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from experiments.spectral_reachability import p3p4_goal_patch_physical_20261027_v1 as m


class Env:
    def __init__(self,missing=None,fail=None,renderfail=None,terminate=None):
        self.n=0;self.missing=missing;self.fail=fail;self.renderfail=renderfail;self.terminate=terminate
    def render(self):
        if self.n==self.renderfail:raise RuntimeError('fixture render')
        rgb=np.zeros((224,224,3),np.uint8)
        if self.n!=self.missing:rgb[20:22,20:22]=[65,105,225]
        return rgb
    def step(self,action):
        self.n+=1
        if self.n==self.fail:raise RuntimeError('fixture step')
        return None,0.,self.n==self.terminate,False,{}


def plan(h=2,root=0,arm='old_Q'):
    p=dict(schema=m.NAME+'.plan',root_index=root,root_sha256=str(root)*64,method=arm,
        planned_horizon_tokens=h,planned_commands=5*h,corrected_search_horizon=3,commands=np.zeros((5*h,2)).tolist(),
        goal=np.zeros(192).tolist(),projection=np.eye(192).tolist(),old_projection=(2*np.eye(192)).tolist(),
        predicted_old_Q_cost=4.,predicted_goal_patch_Q_cost=1.,whole_bank_sha256='bank',metric_sha256='metric')
    p['selection_sha256']=m.digest(p);return p


def branch(tmp_path,h=2,env=None,reset=lambda:dict(context='same'),p=None):
    return m.physical_branch(plan=p or plan(h),output=tmp_path,reset=reset,env=env or Env(),state_of=lambda e:np.full(7,e.n,float),
        evaluate=lambda s:dict(official=dict(official_state_distance=float(s[0])),pose=dict(pose_cost=float(s[0]))))


@pytest.mark.parametrize('h',[2,3])
def test_all_commands_and_durable_witness(tmp_path,h):
    r=branch(tmp_path,h);m.verify_branch(plan(h),r)
    assert r['completed_commands']==5*h and r['full_horizon_completed'] and r['endpoint_scope']==f'full_H{h}'
    assert len(list(tmp_path.glob('*.png')))==5*h+1


@pytest.mark.parametrize('h',[2,3])
@pytest.mark.parametrize('kwargs,a,b,fresh',[
    ({'missing':0},0,0,True),({'missing':6},6,6,True),({'fail':7},7,6,False),
    ({'renderfail':7},7,7,False),({'terminate':7},7,7,True)])
def test_partial_guard_charges(tmp_path,h,kwargs,a,b,fresh):
    r=branch(tmp_path,h,Env(**kwargs));m.verify_branch(plan(h),r)
    assert r['physical_env_step_attempts']==a and r['physical_env_step_returns']==b
    assert not r['full_horizon_completed'] and r['endpoint_fresh']==fresh
    assert r['endpoint_scope'] not in ('full_H2','full_H3')


def test_failed_reset_durable(tmp_path):
    def fail():raise ValueError('fixture identity failure')
    r=branch(tmp_path,reset=fail);assert r['status']=='branch_failed' and not r['endpoint_fresh']
    assert len(list(tmp_path.glob('*.physical.json')))==1


@pytest.mark.parametrize('damage',['count','action','horizon','hash','arm'])
def test_bad_plan_before_reset(tmp_path,damage):
    p=plan()
    if damage=='count':p['planned_commands']=15
    elif damage=='action':p['commands'][0][0]=float('nan')
    elif damage=='horizon':p['planned_horizon_tokens']=True
    elif damage=='arm':p['method']='dynamic'
    else:p['selection_sha256']='wrong'
    with pytest.raises(ValueError):branch(tmp_path,p=p,reset=lambda:pytest.fail('must not reset'))


def test_tampered_journal(tmp_path):
    r=branch(tmp_path);Path(r['journal_path']).write_bytes(b'{}\n')
    with pytest.raises(ValueError):m.verify_branch(plan(),r)


def test_no_early_cuda_setup(tmp_path):
    with pytest.raises(ValueError):m.optional_encodings([],[],tmp_path,lambda e:pytest.fail('early CUDA setup'))


@pytest.mark.parametrize('kwargs,n,status',[({},1,'completed'),({'missing':6},1,'completed'),({'fail':7},0,'blocked_stale_endpoint'),({'renderfail':7},0,'blocked_stale_endpoint')])
def test_encode_scope_both_Q(tmp_path,kwargs,n,status):
    r=branch(tmp_path,env=Env(**kwargs));m.write(tmp_path/'physical_report.json',dict(rows=[r]))
    def setup(event):
        assert (tmp_path/'physical_report.json').is_file()
        return lambda path:np.ones(192)
    receipts=m.optional_encodings([plan()],[r],tmp_path,setup);e=receipts[0]
    assert e['attempted_encode_samples']==n and e['status']==status
    if n:assert e['actual_old_Q_cost']==4 and e['actual_goal_patch_Q_cost']==1 and e['endpoint_scope']==r['endpoint_scope']


@pytest.mark.parametrize('stage',['setup','encode','shape','RGB'])
def test_encoding_failures_durable(tmp_path,stage):
    r=branch(tmp_path);m.write(tmp_path/'physical_report.json',dict(rows=[r]))
    if stage=='RGB':Path(r['terminal_rgb_path']).write_bytes(b'broken')
    def setup(event):
        if stage=='setup':raise RuntimeError('fixture setup')
        def encode(path):
            if stage=='encode':raise RuntimeError('fixture encode')
            return [0.]*3
        return encode
    e=m.optional_encodings([plan()],[r],tmp_path,setup)[0]
    assert e['status'] in ('encoding_setup_failed','encoding_failed')
    assert len(list(tmp_path.glob('*.encoding.json')))==1


def pair_fixture(hs=(2,3),partial=False):
    plans=[];rows=[]
    for i in range(4):
        for arm,h in zip(m.replay.ARMS,hs):
            p=plan(h,i,arm);plans.append(p)
            rows.append(dict(root_sha256=p['root_sha256'],method=arm,selection_sha256=p['selection_sha256'],planned_horizon_tokens=h,planned_commands=5*h,
                context_identity=dict(same='context'),full_horizon_completed=not partial,endpoint_fresh=True,physical_env_step_returns=6 if partial else 5*h,pose=dict(pose_cost=1 if arm=='old_Q' else 2)))
    return plans,rows


@pytest.mark.parametrize('hs,partial,label',[((2,2),False,'full_H2_pair'),((3,3),False,'full_H3_pair'),((2,3),False,'full_mixed_H2_H3'),((2,3),True,'partial_same_steps')])
def test_strata(hs,partial,label):
    pairs=m.comparisons(*pair_fixture(hs,partial))
    assert all(p['classification']==label for p in pairs)
    assert all((p['new_minus_old_pose_cost'] is None)==(partial or hs[0]!=hs[1]) for p in pairs)


def test_unmatched_context_or_stale():
    ps,rs=pair_fixture();rs[0].pop('context_identity');rs[2]['endpoint_fresh']=False
    pairs=m.comparisons(ps,rs);assert pairs[0]['classification']==pairs[1]['classification']=='partial_or_unmatched'


def test_materialize_all_eight(tmp_path):
    ps,_=pair_fixture();rows=[];m.materialize_missing(ps,rows,tmp_path)
    assert len(rows)==8 and all(r['status']=='blocked' and r['physical_env_step_attempts']==0 for r in rows)


@pytest.fixture(scope='module')
def exports():
    p=m.replay.validate_prepared(m.PROJECT/m.REPLAY_PREPARED,m.REPLAY_SHA)
    # Explicit fake post-CUDA exporter fixtures: not actual replay evidence.
    source=p['plans'];out=[]
    for x in source:
        e=dict(deepcopy(x),replay_report_sha256='fixture_report',replay_witness={'fixture':True},replay_shared_resources={'fixture':True},inherited_resources=p['inherited_resources'])
        e['export_sha256']=m.digest(e);out.append(e)
    return dict(expected_exports=source,replay_report_sha256='fixture_report'),out


def test_exported_varying_horizon_plans(exports):
    p,es=exports;plans=m.plans_from_exports(p,es)
    assert len(plans)==8 and sum(p['planned_commands'] for p in plans)==90
    for p in plans:m.validate_plan(p)


@pytest.mark.parametrize('damage',['missing','duplicate','goal','oldQ','newQ','bank','context','actions','report','root','hash','horizon'])
def test_export_binding_mutations(exports,damage):
    p,es=deepcopy(exports);e=es[0]
    if damage=='missing':es.pop()
    elif damage=='duplicate':es[1]=deepcopy(e)
    elif damage=='goal':e['goal'][0]+=1
    elif damage=='oldQ':e['old_projection'][0][0]+=1
    elif damage=='newQ':e['projection'][0][0]+=1
    elif damage=='bank':e['bank_sha256']='wrong'
    elif damage=='context':e['root_context']['action_history'][0][0]+=1
    elif damage=='actions':e['actions'][0][0]+=1
    elif damage=='report':e['replay_report_sha256']='wrong'
    elif damage=='root':e['root_sha256']='wrong'
    elif damage=='hash':e['export_sha256']='wrong'
    else:e['horizon']=3
    with pytest.raises((ValueError,KeyError)):m.plans_from_exports(p,es)


def test_source_prepare_no_neural_physics(tmp_path,monkeypatch):
    original=builtins.__import__
    def no_runtime(name,*a,**kw):
        assert name.split('.')[0] not in {'torch','gymnasium','pymunk','mujoco'},name
        return original(name,*a,**kw)
    monkeypatch.setattr(builtins,'__import__',no_runtime)
    out=tmp_path/'source';r=m.prepare(m.PROJECT,out);sha=m.io.sha((out/'prepare.json').read_bytes())
    assert r['new_neural_calls']==r['new_physical_steps']==0
    p=m.validate_prepared(out,sha);assert p['stage']=='source_prepared_awaiting_replay'
    with pytest.raises(ValueError):m.execute_physics(out,sha,tmp_path/'forbidden')
    assert not (tmp_path/'forbidden').exists()
    with pytest.raises(ValueError):m.prepare(m.PROJECT,out)
    with pytest.raises(ValueError):m.validate_prepared(out,'wrong')


def test_bind_roundtrip_metadata_fixture_only(tmp_path,monkeypatch,exports):
    source=tmp_path/'source';bound=tmp_path/'bound';run=tmp_path/'fake_replay'
    m.prepare(m.PROJECT,source);sha=m.io.sha((source/'prepare.json').read_bytes())
    run.mkdir();m.write(run/'report.json',dict(explicit_fixture_only=True))
    report_sha=m.io.sha((run/'report.json').read_bytes());_,es=deepcopy(exports)
    for e in es:
        e['replay_report_sha256']=report_sha;e.pop('export_sha256');e['export_sha256']=m.digest(e)
    monkeypatch.setattr(m.replay,'selected_actions',lambda path:deepcopy(es))
    m.bind_report(source,sha,run,report_sha,bound)
    p=m.validate_prepared(bound,m.io.sha((bound/'prepare.json').read_bytes()))
    assert p['stage']=='report_bound_ready' and len(p['plans'])==8
    assert sum(x['planned_commands'] for x in p['plans'])==90
    with pytest.raises(ValueError):m.bind_report(source,sha,run,'wrong',tmp_path/'bad')
    (run/'report.json').write_text('{}')
    with pytest.raises(ValueError):m.validate_prepared(bound,m.io.sha((bound/'prepare.json').read_bytes()))


def test_incoming_warmup_token_exact():
    history=np.asarray([[0.,0.],[.1,-.1],[.2,.1]],np.float32)
    packet=np.repeat(history[:,None,:],5,axis=1).reshape(3,10)
    mean,std=(np.asarray(m.helpers.NORMALIZATION[k],np.float32) for k in ('mean','std'))
    expected=((packet-np.tile(mean,5))/np.tile(std,5))[1:]
    assert m.helpers.validate_reset_history(history,expected)['normalized_between_history']==expected.tolist()
    damaged=expected.copy();damaged[0,0]=np.nextafter(damaged[0,0],np.float32(100))
    with pytest.raises(ValueError):m.helpers.validate_reset_history(history,damaged)


@pytest.mark.parametrize('damage',[None,'foreign','duplicate','goalshape','nan'])
def test_root_and_goal_census_before_steps(damage):
    roots=[dict(state=np.full(7,i).tolist(),goal=np.full(7,i+1).tolist()) for i in range(4)]
    fingerprint=lambda r:str(int(r['state'][0]));refs={str(i):{} for i in range(4)}
    if damage=='foreign':roots[3]['state'][0]=9
    elif damage=='duplicate':roots[3]=deepcopy(roots[0])
    elif damage=='goalshape':roots[3]['goal'].pop()
    elif damage=='nan':roots[3]['goal'][0]=float('nan')
    if damage:
        with pytest.raises(ValueError):m.helpers.validate_roots(roots,refs,fingerprint)
    else:assert len(m.helpers.validate_roots(roots,refs,fingerprint))==4
