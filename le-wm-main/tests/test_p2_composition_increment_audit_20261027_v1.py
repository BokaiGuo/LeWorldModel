"""NumPy/mock only; no model, physics, saved-outcome execution or SVD."""
import builtins
from copy import deepcopy
import sys
import numpy as np
import pytest
from experiments.spectral_reachability import p2_composition_increment_audit_20261027_v1 as m


def counters(n):return dict(predict_calls=n,predict_samples=n,action_encoder_calls=n,action_encoder_samples=n,predict_context_tokens=3*n,encoder_context_tokens=3*n,failed_entries=0)


def fixture():
    rows=[];plans=[];records=[]
    for i in range(4):
        root=f'{i:064x}';results=[];ex={arm:counters(n) for arm,n in zip(m.ARMS,(84,164))};at={arm:counters(n) for arm,n in zip(m.ARMS,(88,168))}
        ops=dict(jacobian_calls=41,scalar_output_vjp_directions=7872,svd_calls=42,bounded_solve_calls=41)
        rr=dict(unique=counters(252),shared=counters(4),exclusive=ex,standalone_attributed=at,completed_operations=ops)
        for arm in m.ARMS:
            composition=arm==m.ARMS[1];cost=1. if composition else 2.
            cs=[dict(index=k,candidate_id=arm+str(k),completed=True,cost=cost+k,actions=np.zeros((10,2)).tolist()) for k in range(40)]
            results.append(dict(arm=arm,attempts=cs))
            accounting=dict(shared=counters(4),exclusive=deepcopy(ex[arm]),paired_unique=counters(252),standalone_attributed=deepcopy(at[arm]),completed_operations=deepcopy(ops))
            p=dict(root_index=i,root_sha256=root,method=arm,root=dict(context='same',goal_rgb_sha256='goal'),action_domain=dict(low=[-1.,-1.],high=[1.,1.]),
                goal_vector_sha256='goal0',projection_sha256='Q',planned_horizon_tokens=2,planned_commands=10,commands=np.zeros((10,2)).tolist(),
                selected_candidate_id=arm+'0',receipt_sha256=arm+'receipt',planning_accounting=deepcopy(accounting))
            r=dict(root_index=i,root_sha256=root,arm=arm,goal=[0.]*7,initial_state=[0.]*7,full_H2=True,status='completed',returned_commands=10,observed_commands=10,
                selected_candidate_id=p['selected_candidate_id'],receipt_sha256=p['receipt_sha256'],predicted_q_mse=cost,planning_accounting=accounting,
                full_H2_metrics=dict(predicted_q_mse=cost,real_encoder_q_mse=3. if composition else 2.,object_pose_cost=2. if composition else 1.,official_state_distance=0.),observed_endpoint_metrics=None)
            rows.append(r);plans.append(p)
        records.append(dict(root_index=i,root_sha256=root,resource_receipt=rr,results=results))
    totals=dict(actual_unique_predict_calls=1008,actual_unique_action_encoder_calls=1008,completed_jacobian_calls=164,completed_scalar_output_vjp_directions=31488,completed_svd_calls=168)
    return dict(rows=rows,P1_admission=False,P2_admission=False,planning_totals=totals),dict(rows=records,P1_admission=False,P2_admission=False,**totals),plans


def test_stable_root_not_order_and_no_admission():
    o,p,plans=fixture();o['rows'].reverse();p['rows'].reverse();plans.reverse()
    r=m.audit(o,p,plans,dict(status='fail',gates=dict(mechanism=False)))
    assert r['integrity_verified'] and r['full_same_H2_pairs']==4
    assert [x['root_index'] for x in r['pairs']]==list(range(4))
    assert not r['P2_admission'] and not r['equal_predict_budget'] and not r['scientific_increment_established']
    assert r['known_unique_neural_cost']['predict_samples']==1008
    pair=r['pairs'][0];assert pair['budget']['standalone_predict']==[88,168]
    assert pair['prediction_directions']['real_encoder_q_mse']==pair['prediction_directions']['object_pose_cost']=='strict_reversal'


@pytest.mark.parametrize('damage',['duplicate','missing','foreign','index','goal','context','Q','domain','steps','commands','prefix','slots','winner','cost','admission','actual_bool'])
def test_invalid_evidence_rejected(damage):
    o,p,ps=fixture()
    if damage=='duplicate':o['rows'][1]=deepcopy(o['rows'][0])
    elif damage=='missing':o['rows'].pop()
    elif damage=='foreign':o['rows'][0]['root_sha256']='wrong'
    elif damage=='index':o['rows'][0]['root_index']=3
    elif damage=='goal':o['rows'][0]['goal'][0]+=1
    elif damage=='context':ps[0]['root']['context']='wrong'
    elif damage=='Q':ps[0]['projection_sha256']='wrong'
    elif damage=='domain':ps[0]['action_domain']['low']=[0.,0.]
    elif damage=='steps':ps[0]['planned_commands']=9
    elif damage=='commands':ps[0]['commands'][0][0]=np.nan
    elif damage=='prefix':p['rows'][0]['results'][0]['attempts'][1]['actions'][0][0]=.5
    elif damage=='slots':p['rows'][0]['results'][0]['attempts'].pop()
    elif damage=='winner':ps[0]['selected_candidate_id']='wrong'
    elif damage=='cost':o['rows'][0]['predicted_q_mse']=999.
    elif damage=='admission':p['P1_admission']=True
    else:o['rows'][0]['full_H2']=1
    with pytest.raises(ValueError):m.audit(o,p,ps)


@pytest.mark.parametrize('scope',['shared','exclusive','paired_unique','standalone_attributed'])
def test_missing_budget_reported_not_zero(scope):
    o,p,ps=fixture();o['rows'][0]['planning_accounting'][scope].pop('predict_samples');ps[0]['planning_accounting']=deepcopy(o['rows'][0]['planning_accounting'])
    r=m.audit(o,p,ps)
    assert r['known_unique_neural_cost'] is None and r['pairs'][0]['budget']['equal_predict_budget'] is None
    assert any('budget_missing:' in g for g in r['prerequisite_gaps']) and not r['P2_admission']


def test_missing_operation_unknown():
    o,p,ps=fixture();p['rows'][0]['resource_receipt']['completed_operations'].pop('svd_calls')
    r=m.audit(o,p,ps);assert r['known_completed_operations']['svd_calls'] is None


def test_entire_budget_and_global_fields_missing():
    o,p,ps=fixture();o['rows'][0].pop('planning_accounting');ps[0].pop('planning_accounting')
    r=m.audit(o,p,ps);assert r['known_unique_neural_cost'] is None and not r['P2_admission']
    o,p,ps=fixture();p.pop('actual_unique_predict_calls')
    r=m.audit(o,p,ps);assert 'global_neural_budget_missing:actual_unique_predict_calls' in r['prerequisite_gaps']


def test_loader_missing_artifact_machine_refusal(monkeypatch):
    from types import SimpleNamespace
    fake=SimpleNamespace(evidence=SimpleNamespace(EvidenceReader=lambda:SimpleNamespace(hashes={})))
    def missing(*args):raise FileNotFoundError('mock missing source')
    fake.load_and_compute=missing
    key='experiments.spectral_reachability.p2_composition_paired_outcomes_20261027_v2'
    monkeypatch.setitem(sys.modules,key,fake)
    import experiments.spectral_reachability as package
    monkeypatch.setattr(package,'p2_composition_paired_outcomes_20261027_v2',fake,raising=False)
    r=m.run();assert r['status']=='refused_missing_or_inconsistent_evidence' and not r['P2_admission']


@pytest.mark.parametrize('damage',['double_shared','attribution','tokens','negative','bool','global'])
def test_inconsistent_budget_rejected(damage):
    o,p,ps=fixture();b=o['rows'][0]['planning_accounting']
    if damage=='double_shared':b['paired_unique']['predict_samples']+=4
    elif damage=='attribution':b['standalone_attributed']['predict_calls']+=1
    elif damage=='tokens':b['exclusive']['encoder_context_tokens']+=1
    elif damage=='negative':b['shared']['predict_calls']=-1
    elif damage=='bool':b['shared']['predict_calls']=True
    else:p['actual_unique_predict_calls']-=4
    ps[0]['planning_accounting']=deepcopy(b)
    with pytest.raises(ValueError):m.audit(o,p,ps)


def test_partials_retained_no_full_delta():
    o,p,ps=fixture();r=o['rows'][0];r.update(full_H2=False,status='observation_unavailable',returned_commands=6,observed_commands=6,full_H2_metrics=None,observed_endpoint_metrics=dict(object_pose_cost=3.))
    out=m.audit(o,p,ps);assert out['full_same_H2_pairs']==3
    assert out['pairs'][0]['composition_minus_no_composition'] is None and out['pairs'][0]['observed_partials']


def test_missing_actual_Q_no_fake_improvement():
    o,p,ps=fixture();o['rows'][0]['full_H2_metrics']['real_encoder_q_mse']=None
    r=m.audit(o,p,ps);assert r['pairs'][0]['prediction_directions']['real_encoder_q_mse']=='unavailable'
    assert 'actual_terminal_Q_missing' in r['pairs'][0]['missing']


def test_missing_P1_and_affirmative_mock_cannot_admit():
    for p1 in (None,dict(status='pass',P1_admission=True,gates={'mock':True})):
        r=m.audit(*fixture(),p1=p1);assert r['P2_admission'] is False and r['original_P1']['admission_revalidated'] is False


def test_pure_core_no_runtime_import(monkeypatch):
    original=builtins.__import__
    def guard(name,*a,**kw):
        assert name.split('.')[0] not in {'torch','gymnasium','pymunk','mujoco'},name
        return original(name,*a,**kw)
    monkeypatch.setattr(builtins,'__import__',guard);assert m.audit(*fixture())['new_neural_calls']==0


@pytest.mark.parametrize('p,a,label',[(-1,1,'strict_reversal'),(-1,-1,'same_direction'),(0,2,'tie_in_either'),(-1,0,'tie_in_either'),(-1,None,'unavailable')])
def test_directions(p,a,label):assert m.direction(p,a)==label
