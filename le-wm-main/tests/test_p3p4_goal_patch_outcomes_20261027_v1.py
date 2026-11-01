"""Pure saved-array arithmetic tests; no neural or physics execution."""
import builtins
from copy import deepcopy
import numpy as np
import pytest
from experiments.spectral_reachability import p3p4_goal_patch_outcomes_20261027_v1 as m


def fixture(h=2):
    history=np.zeros((3,2),np.float32)
    mean,std=(np.asarray(m.physical.helpers.NORMALIZATION[k],np.float32) for k in ('mean','std'))
    tokens=((np.repeat(history[:,None,:],5,axis=1).reshape(3,10)-np.tile(mean,5))/np.tile(std,5))[1:]
    p=dict(schema=m.physical.NAME+'.plan',root_index=0,root_sha256='a'*64,method='old_Q',planned_horizon_tokens=h,planned_commands=5*h,
        corrected_search_horizon=3,commands=np.zeros((5*h,2)).tolist(),goal=np.zeros(192).tolist(),projection=np.eye(192).tolist(),old_projection=(2*np.eye(192)).tolist(),
        predictor_context=dict(action_history=tokens.tolist()),selected_predicted_context=dict(latent_history=np.ones((3,192)).tolist()),
        predicted_old_Q_cost=4.,predicted_goal_patch_Q_cost=1.,whole_bank_sha256='bank',metric_sha256='metric')
    p['selection_sha256']=m.physical.digest(p)
    identity=dict(history_rgb_sha256='history',goal_rgb_sha256='goalRGB',raw_history=history.tolist(),normalized_between_history=tokens.tolist(),state=[0.]*7,goal=[0.]*7)
    trace=[dict(kind='observation',primitive_step=i+1,command=[0.,0.],state=[0.]*7,reward=0.,terminated=False,truncated=False) for i in range(5*h)]
    r=dict(root_index=0,root_sha256=p['root_sha256'],method='old_Q',selection_sha256=p['selection_sha256'],repetition=0,
        planned_horizon_tokens=h,planned_commands=5*h,physical_env_step_attempts=5*h,physical_env_step_returns=5*h,attempted_commands=5*h,completed_commands=5*h,
        trace=trace,status='completed',full_horizon_completed=True,endpoint_fresh=True,endpoint_scope=f'full_H{h}',failures=[],context_identity=identity,final_state=[0.]*7,
        official=m.arithmetic.labels(np.zeros(7),np.zeros(7)),pose=m.arithmetic.pose(np.zeros(7),np.zeros(7)),terminal_rgb_sha256='terminal')
    e=dict(root_sha256=p['root_sha256'],method='old_Q',selection_sha256=p['selection_sha256'],planned_horizon_tokens=h,endpoint_scope=f'full_H{h}',
        attempted_encode_samples=1,status='completed',failure=None,terminal_rgb_sha256='terminal',terminal_vector=(np.ones(192)*2).tolist(),
        whole_bank_sha256='bank',metric_sha256='metric',actual_old_Q_cost=16.,actual_goal_patch_Q_cost=4.,predicted_old_Q_cost=4.,predicted_goal_patch_Q_cost=1.)
    return p,r,e,{k:identity[k] for k in ('history_rgb_sha256','goal_rgb_sha256')}


@pytest.mark.parametrize('h',[2,3])
def test_recompute_both_Q_pose(h):
    r=m.branch(*fixture(h));assert r['full_selected_horizon']
    assert r['full_metrics']['actual_old_Q_cost']==16 and r['full_metrics']['actual_goal_patch_Q_cost']==4 and r['full_metrics']['pose_cost']==0


@pytest.mark.parametrize('damage',['root','selection','horizon','attempts','trace','RGB','warmup','state','pose','official','actualOldQ','actualNewQ','predOldQ','predNewQ','image','bank','metric'])
def test_mutation_rejected(damage):
    p,r,e,ref=fixture()
    if damage=='root':r['root_sha256']='wrong'
    elif damage=='selection':r['selection_sha256']='wrong'
    elif damage=='horizon':r['planned_horizon_tokens']=3
    elif damage=='attempts':r['physical_env_step_attempts']=9
    elif damage=='trace':r['trace'][0]['command'][0]=1
    elif damage=='RGB':r['context_identity']['goal_rgb_sha256']='wrong'
    elif damage=='warmup':r['context_identity']['normalized_between_history'][0][0]+=1
    elif damage=='state':r['final_state'][0]=1
    elif damage=='pose':r['pose']['pose_cost']=1
    elif damage=='official':r['official']['official_state_distance']=1
    elif damage=='actualOldQ':e['actual_old_Q_cost']+=1
    elif damage=='actualNewQ':e['actual_goal_patch_Q_cost']+=1
    elif damage=='predOldQ':e['predicted_old_Q_cost']+=1
    elif damage=='predNewQ':e['predicted_goal_patch_Q_cost']+=1
    elif damage=='image':e['terminal_rgb_sha256']='wrong'
    elif damage=='bank':e['whole_bank_sha256']='wrong'
    else:e['metric_sha256']='wrong'
    with pytest.raises(ValueError):m.branch(p,r,e,ref)


def test_partial_and_missing_encoding():
    p,r,e,ref=fixture();r.update(status='observation_unavailable',full_horizon_completed=False,endpoint_scope='partial_H2',physical_env_step_attempts=4,physical_env_step_returns=4,attempted_commands=4,completed_commands=4)
    r['trace']=r['trace'][:4];e['endpoint_scope']='partial_H2'
    x=m.branch(p,r,e,ref);assert x['full_metrics'] is None and x['actual_Q']['actual_old_Q_cost']==16
    y=m.branch(p,r,None,ref);assert y['actual_Q']['actual_old_Q_cost'] is None
    r['endpoint_fresh']=False
    with pytest.raises(ValueError):m.branch(p,r,e,ref)


def pair_fixture():
    roots=[str(i)*64 for i in range(4)];rows=[]
    for i,root in enumerate(roots):
        for arm in m.physical.replay.ARMS:
            r=m.branch(*fixture());r.update(root_index=i,root_sha256=root,method=arm)
            if arm=='goal_patch_Q':
                r['full_metrics']['predicted_goal_patch_Q_cost']-=.5
                r['full_metrics']['predicted_old_Q_cost']+=.5
                r['full_metrics']['pose_cost']+=2
                r['full_metrics']['actual_goal_patch_Q_cost']+=1
            rows.append(r)
    return rows,roots


def test_mixed_partial_strata_and_metric_direction():
    rows,roots=pair_fixture();rows[3]['selected_horizon']=3
    for x in rows[4:6]:x.update(full_selected_horizon=False,returned_commands=4,full_metrics=None)
    rows[6].update(initial_context=None,full_selected_horizon=False,endpoint_fresh=False,full_metrics=None)
    result=m.summarize(rows,roots)
    assert [p['classification'] for p in result['pairs']]==['full_H2_pair','full_mixed_H2_H3','partial_same_steps','partial_or_unmatched']
    assert result['full_same_horizon_pair_count']==1
    assert result['pairs'][1]['new_minus_old_full_endpoint_descriptive'] is not None
    assert result['strata']['full_mixed_H2_H3']['same_h_full_summary'] is None
    assert result['pairs'][0]['predicted_vs_actual_directions']['goal_patch']['pose_cost']=='strict_reversal'
    assert result['pairs'][0]['predicted_vs_actual_directions']['old']['pose_cost']=='same_direction'
    assert result['pairs'][2]['partial_equal_step_observed_delta'] is not None


@pytest.mark.parametrize('damage',['missing','duplicate','context','goal','index'])
def test_pair_identity(damage):
    rows,roots=pair_fixture()
    if damage=='missing':rows.pop()
    elif damage=='duplicate':rows[-1]=deepcopy(rows[0])
    elif damage=='context':rows[1]['initial_context']['state'][0]+=1
    elif damage=='goal':rows[1]['initial_context']['goal'][0]+=1
    else:rows[1]['root_index']=3
    with pytest.raises(ValueError):m.summarize(rows,roots)


def test_actual_saved_evidence_without_runtime_imports(monkeypatch):
    original=builtins.__import__
    def no_runtime(name,*a,**kw):
        assert name.split('.')[0] not in {'torch','gymnasium','pymunk','mujoco'},name
        return original(name,*a,**kw)
    monkeypatch.setattr(builtins,'__import__',no_runtime)
    path=m.PROJECT/m.RUN/'report.json';sha=m.io.sha(path.read_bytes());reader=m.io.Evidence(m.PROJECT)
    result=m.compute(path.parent,sha,reader)
    assert len(result['rows'])==8 and result['physical_env_step_attempts']<=90
    assert result['full_same_horizon_pair_count']<=2


def test_prepare_exclusive_tamper(tmp_path):
    out=tmp_path/'prepared';m.prepare(m.PROJECT,out);sha=m.io.sha((out/'prepare.json').read_bytes())
    assert m.validate_prepared(out,sha)['contract']==m.contract()
    with pytest.raises(ValueError):m.prepare(m.PROJECT,out)
    with pytest.raises(ValueError):m.validate_prepared(out,'wrong')
    (out/'protocol.json').write_text('{}')
    with pytest.raises(ValueError):m.validate_prepared(out,sha)
