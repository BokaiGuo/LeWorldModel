"""Array, saved-evidence and fake-artifact tests; no neural/real physics."""
from copy import deepcopy
import builtins
from pathlib import Path
import numpy as np
import pytest
from experiments.spectral_reachability import p3_horizon_residual_outcomes_20261027_v4 as a

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/a.DEFAULT_RUN


def fixture(h=2):
    history=np.zeros((3,2),np.float32);mean=np.asarray(a.physical.NORMALIZATION['mean'],np.float32);std=np.asarray(a.physical.NORMALIZATION['std'],np.float32)
    tokens=np.repeat(((history-mean)/std)[:,None,:],5,axis=1).reshape(3,10)[1:]
    plan=dict(schema=a.physical.NAME+'.plan',root_index=0,root_sha256='a'*64,method='uniform',
        planned_horizon_tokens=h,planned_commands=5*h,corrected_search_horizon=3,commands=np.zeros((5*h,2)).tolist(),
        goal=np.zeros(192).tolist(),projection=np.eye(192).tolist(),goal_vector_sha256='goal',projection_sha256='Q',
        predictor_context={'action_history':tokens.tolist()},selected_predicted_context={'latent_history':np.zeros((3,192)).tolist()},predicted_Q_cost=0.)
    plan['selection_sha256']=a.physical.planning.digest(plan)
    identity=dict(history_rgb_sha256='history',goal_rgb_sha256='goalRGB',raw_history=history.tolist(),
        normalized_between_history=tokens.tolist(),state=[0.]*7,goal=[0.]*7)
    trace=[dict(kind='observation',primitive_step=i+1,command=[0.,0.],state=[0.]*7,reward=0.,terminated=False,truncated=False) for i in range(5*h)]
    row=dict(root_index=0,root_sha256=plan['root_sha256'],method='uniform',selection_sha256=plan['selection_sha256'],repetition=0,
        planned_horizon_tokens=h,planned_commands=5*h,physical_env_step_attempts=5*h,physical_env_step_returns=5*h,
        attempted_commands=5*h,completed_commands=5*h,trace=trace,status='completed',full_horizon_completed=True,
        endpoint_fresh=True,endpoint_scope=f'full_H{h}',failures=[],context_identity=identity,final_state=[0.]*7,
        official=a.labels(np.zeros(7),np.zeros(7)),pose=a.pose(np.zeros(7),np.zeros(7)),terminal_rgb_sha256='terminal')
    enc=dict(root_sha256=plan['root_sha256'],method='uniform',selection_sha256=plan['selection_sha256'],planned_horizon_tokens=h,
        endpoint_scope=f'full_H{h}',attempted_encode_samples=1,status='completed',failure=None,terminal_rgb_sha256='terminal',
        terminal_vector=np.zeros(192).tolist(),goal_vector_sha256='goal',projection_sha256='Q',original_Q_cost=0.)
    ref={k:identity[k] for k in ('history_rgb_sha256','goal_rgb_sha256')}
    return plan,row,enc,ref


@pytest.mark.parametrize('h',[2,3])
def test_full_metrics(h):
    result=a.branch_outcome(*fixture(h));assert result['full_selected_horizon']
    assert result['actual_Q_cost']==result['predicted_Q_cost']==result['full_metrics']['pose_cost']==0


@pytest.mark.parametrize('damage',['root','selection','horizon','attempts','trace','goalRGB','warmup','state','pose','official','actualQ','predQ','image','encroot','encQ'])
def test_mutations_fail(damage):
    p,r,e,ref=fixture()
    if damage=='root':r['root_sha256']='b'*64
    elif damage=='selection':r['selection_sha256']='b'*64
    elif damage=='horizon':r['planned_horizon_tokens']=3
    elif damage=='attempts':r['physical_env_step_attempts']=9
    elif damage=='trace':r['trace'][0]['command'][0]=.1
    elif damage=='goalRGB':r['context_identity']['goal_rgb_sha256']='wrong'
    elif damage=='warmup':r['context_identity']['normalized_between_history'][0][0]+=.1
    elif damage=='state':r['final_state'][0]=1
    elif damage=='pose':r['pose']['pose_cost']=3
    elif damage=='official':r['official']['official_state_distance']=3
    elif damage=='actualQ':e['original_Q_cost']=3
    elif damage=='predQ':p['predicted_Q_cost']=3;p['selection_sha256']=a.physical.planning.digest({k:v for k,v in p.items() if k!='selection_sha256'});r['selection_sha256']=e['selection_sha256']=p['selection_sha256']
    elif damage=='image':e['terminal_rgb_sha256']='wrong'
    elif damage=='encroot':e['root_sha256']='wrong'
    else:e['projection_sha256']='wrong'
    with pytest.raises(ValueError):a.branch_outcome(p,r,e,ref)


def test_partial_encoding_never_promoted():
    p,r,e,ref=fixture();r.update(status='observation_unavailable',full_horizon_completed=False,endpoint_scope='partial_H2',
        physical_env_step_attempts=4,physical_env_step_returns=4,attempted_commands=4,completed_commands=4)
    r['trace']=r['trace'][:4];e['endpoint_scope']='partial_H2'
    x=a.branch_outcome(p,r,e,ref)
    assert not x['full_selected_horizon'] and x['full_metrics'] is None and x['actual_Q_cost']==0


def test_missing_encoding_is_null():
    p,r,_,ref=fixture();x=a.branch_outcome(p,r,None,ref)
    assert x['full_selected_horizon'] and x['actual_Q_cost'] is None


def test_stale_endpoint_cannot_encode():
    p,r,e,ref=fixture();r.update(status='execution_failed',endpoint_fresh=False,full_horizon_completed=False)
    with pytest.raises(ValueError):a.branch_outcome(p,r,e,ref)


@pytest.mark.parametrize('pred,actual,label',[(-1,1,'strict_reversal'),(1,-1,'strict_reversal'),(-1,-2,'same_direction'),(1,2,'same_direction'),
    (0,1,'tie_in_either'),(-1,0,'tie_in_either'),(0,0,'tie_in_either'),(-1,None,'unavailable')])
def test_strict_reversal(pred,actual,label):assert a.reversal(pred,actual)==label


def pairs_fixture():
    rows=[];roots=[str(i)*64 for i in range(4)]
    for i,root in enumerate(roots):
        for mode in a.physical.METHODS:
            r=a.branch_outcome(*fixture());r.update(root_sha256=root,root_index=i,method=mode)
            if mode=='dynamic':
                r['full_metrics']['predicted_Q_cost']=-1
                r['full_metrics']['actual_Q_cost']=1
                r['full_metrics']['pose_cost']=2
                r['full_metrics']['official_state_distance']=0
            rows.append(r)
    return rows,roots


def test_separate_full_partial_mixed_and_ties():
    rows,roots=pairs_fixture();rows[3]['selected_horizon']=3
    for x in rows[4:6]:x.update(full_selected_horizon=False,returned_commands=4,full_metrics=None)
    rows[6].update(initial_context=None,full_selected_horizon=False,endpoint_fresh=False,full_metrics=None)
    result=a.summarize(rows,roots)
    assert [p['classification'] for p in result['pairs']]==['full_H2_pair','full_mixed_H2_H3','partial_same_steps','partial_or_unmatched']
    assert result['full_same_horizon_pair_count']==1
    p=result['pairs'][0];assert p['predicted_Q_vs_actual_direction']=={'actual_Q_cost':'strict_reversal','pose_cost':'strict_reversal','official_state_distance':'tie_in_either'}
    assert all(p['dynamic_minus_uniform'] is None for p in result['pairs'][1:])
    assert result['strata']['full_H2_pair']['delta_descriptive']['pose_cost']['n']==1
    assert result['strata']['full_H3_pair']['delta_descriptive']['pose_cost']['n']==0


@pytest.mark.parametrize('damage',['missing','duplicate','context','goal','index'])
def test_strict_root_pairing(damage):
    rows,roots=pairs_fixture()
    if damage=='missing':rows.pop()
    elif damage=='duplicate':rows[-1]=deepcopy(rows[0])
    elif damage=='context':rows[1]['initial_context']['state'][0]=1
    elif damage=='goal':rows[1]['initial_context']['goal'][0]=1
    else:rows[1]['root_index']=3
    with pytest.raises(ValueError):a.summarize(rows,roots)


@pytest.fixture(scope='module')
def saved():
    protocol=a.physical.validate_prepared(ROOT/a.PHYSICAL_PREPARED,a.PHYSICAL_SHA)
    return protocol,a.io.decode((RUN/'report.json').read_bytes())


def test_real_saved_artifacts_and_costs(saved):
    protocol,report=saved;plans=a.io.index(protocol['plans'],('root_sha256','method'));enc=a.io.index(report['encodings'],('root_sha256','method'))
    reader=a.io.Evidence(ROOT);outcomes=[]
    for row in report['rows']:
        key=row['root_sha256'],row['method'];plan=plans[key]
        a.audit_branch_artifacts(RUN,plan,row,reader)
        outcomes.append(a.branch_outcome(plan,row,enc[key],protocol['references'][key[0]]))
    result=a.summarize(outcomes,protocol['roots']);assert len(result['pairs'])==4
    assert len(outcomes)==8 and all(x['actual_Q_cost'] is not None for x in outcomes)
    reader.recheck()


def test_real_complete_arithmetic_no_runtime(saved,monkeypatch):
    original=builtins.__import__
    def no_runtime(name,*args,**kwargs):
        assert name.split('.')[0] not in {'torch','gymnasium','pymunk','mujoco'},name
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',no_runtime)
    protocol=dict(physical_prepared=str(ROOT/a.PHYSICAL_PREPARED))
    result=a.load_and_compute(RUN,protocol,a.io.Evidence(ROOT),a.physical.durable.digest(RUN/'report.json'))
    assert result['status']=='descriptive_arithmetic_completed' and not result['P3_admission']
    assert result['arithmetic_verification_overhead']['neural_calls']==0


@pytest.mark.parametrize('damage',['journal_sha','endpoint_scope','guard_pixels','terminal_rgb','command'])
def test_real_journal_and_guard_mutations_rejected(saved,damage):
    protocol,report=saved;row=deepcopy(report['rows'][0]);plan=next(p for p in protocol['plans'] if p['root_sha256']==row['root_sha256'] and p['method']==row['method'])
    if damage=='journal_sha':row['journal_sha256']='0'*64
    elif damage=='endpoint_scope':row['endpoint_scope']='partial_H2'
    elif damage=='guard_pixels':row['guard']['observations'][0]['canonical_agent_pixels']+=1
    elif damage=='terminal_rgb':row['terminal_rgb_sha256']='0'*64
    else:
        plan=deepcopy(plan);plan['commands'][0][0]+=.01
    with pytest.raises(ValueError):a.audit_branch_artifacts(RUN,plan,row,a.io.Evidence(ROOT))


@pytest.mark.parametrize('damage',['pairs','total_steps','total_encoding','predict','source','model','enc_scope','encoding_stats'])
def test_final_report_semantic_drift_rejected(saved,damage):
    class MutatingReader(a.io.Evidence):
        def json(self,path,expected=None):
            value=super().json(path,expected)
            if Path(path)==RUN/'report.json':
                if damage=='pairs':value['pairs'][0]['classification']='full_H3_pair'
                elif damage=='total_steps':value['physical_env_step_returns']-=1
                elif damage=='total_encoding':value['attempted_terminal_encode_samples']-=1
                elif damage=='predict':value['new_predict_calls']=1
                elif damage=='source':value['source_verified']=False
                elif damage=='model':value['model_unchanged']=False
                elif damage=='enc_scope':value['encodings'][0]['endpoint_scope']='partial_H2'
                else:value['encoding_stats']['encode_calls']=9
            return value
    with pytest.raises(ValueError):a.load_and_compute(RUN,dict(physical_prepared=str(ROOT/a.PHYSICAL_PREPARED)),MutatingReader(ROOT),a.physical.durable.digest(RUN/'report.json'))


def test_prepare_verify_sealed_own_files(tmp_path,monkeypatch):
    original=builtins.__import__
    def no_runtime(name,*args,**kwargs):
        assert name.split('.')[0] not in {'torch','gymnasium','pymunk','mujoco'},name
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',no_runtime)
    out=tmp_path/'prepared';r=a.prepare(ROOT,out)
    protocol=a.validate_prepared(out,a.physical.durable.digest(out/'prepare.json'))
    assert r['new_neural_calls']==r['new_physical_steps']==0 and protocol['contract']==a.contract()
    with pytest.raises(ValueError):a.prepare(ROOT,out)
    with pytest.raises(ValueError):a.validate_prepared(out,'0'*64)
    (out/'protocol.json').write_text('{}')
    with pytest.raises(ValueError):a.validate_prepared(out,a.physical.durable.digest(out/'prepare.json'))
