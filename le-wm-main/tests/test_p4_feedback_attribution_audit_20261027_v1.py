"""Arithmetic/mock tests only; no environment, model, torch or old runner imports."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1]/'experiments/spectral_reachability/p4_feedback_attribution_audit_20261027_v1.py'
spec = importlib.util.spec_from_file_location('p4_attribution', SOURCE)
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def ctx(latents, raw):
    return dict(layout='states_with_between_state_actions_v1',context_length=3,
                latent_dim=2,action_dim=10,latent_history=latents,
                action_history=a.normalized(raw).tolist())


def fixture():
    """Two-dimensional non-neural mock; all dispatches and receipts explicit."""
    zero = np.zeros((5,2)).tolist()
    initial = ctx([[0.,0.]]*3,[zero,zero])
    p = dict(root_sha256='mock-root',goal=[2.,2.],q=np.eye(2).tolist(),prefix=zero,
             between_raw=[zero,zero],initial_context=initial,saved_predicted_prefix=[.1,.1],
             physical_goal=[0.]*7)
    events,rows,slots,predict = [],[],[],[]
    def emit(kind, **kw):
        e = dict(kind=kind,event_index=len(events),**deepcopy(kw)); events.append(e); return e
    def steps(arm,start):
        for k in range(start+1,start+6):
            emit('step_attempt',arm=arm,primitive_step=k,command=[0.,0.])
            emit('step_returned',arm=arm,primitive_step=k)
            emit('observation',arm=arm,primitive_step=k,state=[0.,0.,float(k),0.,0.,0.,0.])
    for arm in a.ARMS:
        predict.append(dict(method=arm+'_prefix',phase='evaluation',completed=True,failed_predict_calls=0,
                            horizon_tokens=1,context_lengths={'3':1},predict_calls=1,predict_samples=1))
        steps(arm,0); emit('prefix_identity',arm=arm,pixels_exact=True)
        predicted = ctx([[0.,0.],[0.,0.],[.1,.1]],[zero,zero])
        observed = deepcopy(predicted)
        for b in a.BOUNDARIES:
            emit('boundary_begin',arm=arm,boundary=b,between_raw=[zero,zero])
            emit('encode_begin'); emit('jacobian_begin'); emit('bank_begin')
            chosen = observed if arm=='actual' else predicted
            emit('planning_context',arm=arm,boundary=b,observed_context=observed,
                 predicted_context=predicted,chosen_context=chosen)
            candidates=[]
            for i in range(10):
                terminal=[i*.1,i*.1]
                candidates.append(dict(candidate_id=str(i),predicted_terminal=terminal,
                                       nonlinear_terminal_cost=a.cost(terminal,np.array(p['goal']),np.eye(2)),commands=zero+zero))
            winner=min(candidates,key=lambda v:v['nonlinear_terminal_cost'])
            successor=ctx(chosen['latent_history'][1:]+[winner['predicted_terminal']],[zero,zero])
            plan=dict(root_sha256=p['root_sha256'],arm=arm,context=chosen,goal=p['goal'],q=p['q'],prefix=zero,
                      candidates=candidates,selected_candidate_id=winner['candidate_id'],independent_replay_exact=True)
            s=dict(root_sha256=p['root_sha256'],arm=arm,boundary=b,context_sha256=a.content_sha(chosen),
                   metric_sha256=a.content_sha([p['goal'],p['q']]),incoming_between_raw=[zero,zero],
                   compiler_plan=plan,commands=zero,successor=successor)
            sha=a.content_sha(s); emit('selection',selection_sha256=sha,**s)
            slots.append(dict(arm=arm,boundary=b,root_sha256=p['root_sha256'],status='completed',
                              selection_sha256=sha,selected_candidate_id=winner['candidate_id']))
            steps(arm,b)
            emit('packet_complete',arm=arm,boundary=b,next_between_raw=[zero,zero],predicted_context=successor)
            predicted=successor
            observed=ctx(observed['latent_history'][1:]+[[b*.1,b*.1]],[zero,zero])
            for suffix,n,phase in [('compile',2,'planning'),('bank',10,'planning'),('independent_replay',1,'evaluation')]:
                predict.append(dict(method=f'{arm}_{b}_{suffix}',phase=phase,completed=True,failed_predict_calls=0,
                                    horizon_tokens=1,context_lengths={'3':n},predict_calls=n,predict_samples=n))
        rows.append(dict(arm=arm,root_sha256=p['root_sha256'],status='completed',completed_packets=4,
                         selected_attempts=20,selected_returns=20))
    cost=dict(predict_samples=80,selected_attempts=40,selected_returns=40,setup_attempts=0,setup_returns=0,
              encode_attempts=6,encode_completed=6,encode_frames_attempted=18,visual_encoder_attempts=6,
              visual_encoder_completed=6,visual_encoder_frames_attempted=18,visual_encoder_frames_completed=18,
              jacobian_attempts=6,jacobian_completed=6,bank_attempts=6,bank_completed=6,
              action_encoder_attempts=80,action_encoder_completed=80,action_encoder_input_tokens=240)
    report=dict(status='calibration_recorded',failures=[],model_unchanged=True,rows=rows,slots=slots,cost=cost,
                predict=predict,actual_predict_samples=80,scalar_vjp_directions_completed=1152,svd_calls_completed=12)
    emit('final_census',rows=rows,slots=slots,cost=cost,predict=predict)
    return p,report,events


def test_mock_full_and_missing_final_latent():
    result=a.analyze(*fixture())
    assert result['arithmetic_verified'] and not result['P4_admission']
    assert len(result['rounds'])==6
    final=[v for v in result['rounds'] if v['boundary']==15]
    assert len(final)==2 and all(v['prediction_error'] is None and v['observed_Q_cost_change'] is None for v in final)
    assert all(v['next_latent_status']=='unconfirmed_not_saved' for v in final)


@pytest.mark.parametrize('kind,field,value',[
    ('selection','root_sha256','wrong-root'),('planning_context','boundary',99),
    ('boundary_begin','between_raw',np.ones((2,5,2)).tolist()),
    ('step_attempt','command',[1.,1.]),('prefix_identity','pixels_exact',False),
    ('packet_complete','next_between_raw',[]),('observation','state',[float('nan')]*7),
])
def test_corrupt_event_rejected(kind,field,value):
    p,r,e=fixture(); next(v for v in e if v['kind']==kind)[field]=value
    with pytest.raises(ValueError): a.analyze(p,r,e)


@pytest.mark.parametrize('kind',['planning_context','selection','observation','step_attempt','packet_complete'])
def test_duplicate_event_rejected(kind):
    p,r,e=fixture(); e.append(deepcopy(next(v for v in e if v['kind']==kind)))
    for i,v in enumerate(e): v['event_index']=i
    with pytest.raises(ValueError): a.analyze(p,r,e)


@pytest.mark.parametrize('field',['predict_samples','action_encoder_input_tokens','selected_returns','setup_returns',
                                 'jacobian_completed','encode_frames_attempted','bank_completed'])
def test_budget_corruption(field):
    p,r,e=fixture(); r['cost'][field]+=1
    with pytest.raises(ValueError): a.analyze(p,r,e)


def test_feedback_substitution():
    p,r,e=fixture()
    c=next(v for v in e if v['kind']=='planning_context' and v['arm']=='actual' and v['boundary']==10)
    c['chosen_context']=deepcopy(c['predicted_context'])
    with pytest.raises(ValueError,match='substitution'): a.analyze(p,r,e)


def test_history_wrong_shift():
    p,r,e=fixture()
    c=next(v for v in e if v['kind']=='planning_context' and v['boundary']==10)
    c['observed_context']['latent_history'][0]=[8.,8.]
    with pytest.raises(ValueError,match='rolling history'): a.analyze(p,r,e)


def test_receipt_duplicate_and_failure():
    for change in ('duplicate','failed'):
        p,r,e=fixture()
        if change=='duplicate': r['predict'].append(deepcopy(r['predict'][0]))
        else: r['predict'][0]['completed']=False
        with pytest.raises(ValueError): a.analyze(p,r,e)


def test_residual_identity():
    goal=np.array([2.,-1.]); q=np.array([[1.,.2],[.2,2.]])
    d=a.discrepancy([0.,1.],[1.,0.],goal,q)
    assert d['projected_mse']>=0 and abs(d['identity_error'])<1e-12


def test_pose_wrap_and_velocity_separate():
    value=a.pose([0.,0.,3.,4.,2*np.pi,100.,0.],[0.]*7)
    assert value['pose_cost']==pytest.approx(5.)
    assert value['official_state_distance']>100


@pytest.mark.parametrize('raw',['{"a":1,"a":2}','{"a":NaN}','{"a":1e999}'])
def test_strict_json(raw):
    with pytest.raises(ValueError): a.strict_json(raw)


def test_missing_second_run_and_no_admission(tmp_path):
    result=a.run(tmp_path/(a.TAG+'_empty'),results=tmp_path)
    assert result['missing_runs']==2 and not result['two_run_evidence_verified']
    assert result['missing_run_status']=='unconfirmed' and not result['P1_admission'] and not result['P4_admission']
    with pytest.raises(ValueError): a.run(tmp_path/(a.TAG+'_empty'),results=tmp_path)


def test_discovery_excludes_short_runs(tmp_path):
    short=tmp_path/'p4_short'; short.mkdir()
    (short/'protocol.json').write_text(json.dumps({'contract':{'planned_steps_per_arm':10,'boundaries':[5]}}))
    (short/'report.json').write_text('{}')
    paths,inventory=a.discover(tmp_path)
    assert not paths and not inventory[0]['eligible_20step3round']


def test_input_mutation(tmp_path):
    path=tmp_path/'input.json'; path.write_text('{}')
    inputs=a.Inputs(); inputs.json(path); path.write_text('{"changed":1}')
    with pytest.raises(ValueError): inputs.stable()


def test_no_runtime_imports():
    import ast
    tree=ast.parse(SOURCE.read_text())
    imports=[n.module if isinstance(n,ast.ImportFrom) else v.name
             for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))
             for v in (n.names if isinstance(n,ast.Import) else [None])]
    assert not any(x and any(s in x for s in ('torch','pymunk','gym','runtime','compiler')) for x in imports)
