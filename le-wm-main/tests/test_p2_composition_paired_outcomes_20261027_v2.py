"""Pure saved-data arithmetic tests; no model or simulator calls."""
from copy import deepcopy
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.spectral_reachability import p2_composition_paired_outcomes_20261027_v2 as m


@pytest.fixture(scope='module')
def bundle():
    execution=json.loads((m.DEFAULT_PHYSICAL/'protocol.json').read_text())
    _,protocol,prepared,context=m.physical.load_prepared(execution['prepared'])
    report=json.loads((m.DEFAULT_PHYSICAL/'report.json').read_text())
    compiled={r['root_sha256']:dict(r['compiled'],_normalization=context['planning_protocol']['context']['normalization'])
        for r in context['source']['rows']}
    return protocol,report,compiled


def rows_from(bundle):
    protocol,report,compiled=bundle
    enc=m.index(report['encodings'],('root_sha256','method'),8)
    branches=m.index(report['rows'],('root_sha256','method'),8)
    return [m.branch_outcome(p,branches[p['root_sha256'],p['method']],enc[p['root_sha256'],p['method']],compiled[p['root_sha256']])
        for p in protocol['plans']]


def test_real_eight_full_original_metrics_and_strict_identity_join(bundle):
    rows=rows_from(bundle)
    result=m.summarize(list(reversed(rows)))
    assert len(rows)==8 and result['full_H2_pair_count']==4
    assert all(r['classification']=='full_H2' and r['full_H2_metrics']['real_encoder_q_mse'] is not None for r in rows)
    keyed={(r['root_sha256'],r['arm']):r for r in rows}
    for p in result['paired']:
        assert p['composition_minus_no_composition']['object_pose_cost']==(
            keyed[p['root_sha256'],m.ARMS[1]]['full_H2_metrics']['object_pose_cost']-
            keyed[p['root_sha256'],m.ARMS[0]]['full_H2_metrics']['object_pose_cost'])
    assert all(v['n']==4 for v in result['deltas_descriptive'].values())


@pytest.mark.parametrize('defect',['duplicate','missing','wrong_arm','goal','initial','root_index'])
def test_pairing_never_uses_array_order_or_changed_bindings(bundle,defect):
    rows=rows_from(bundle)
    if defect=='duplicate': rows[-1]=deepcopy(rows[0])
    if defect=='missing': rows.pop()
    if defect=='wrong_arm': rows[-1]['arm']='cem'
    if defect in ('goal','initial','root_index'):
        peer=next(r for r in rows if r['root_sha256']==rows[0]['root_sha256'] and r['arm']!=rows[0]['arm'])
        if defect=='goal': peer['goal'][0]+=1
        if defect=='initial': peer['initial_state'][0]+=1
        if defect=='root_index': peer['root_index']+=1
    with pytest.raises((ValueError,KeyError)): m.summarize(rows)


def make_partial(bundle,unobserved=0):
    protocol,report,compiled=bundle
    plan=deepcopy(protocol['plans'][0])
    row=deepcopy(next(r for r in report['rows'] if (r['root_sha256'],r['method'])==(plan['root_sha256'],plan['method'])))
    row.update(status='observation_unavailable',full_horizon_completed=False,
        attempted_commands=3+unobserved,completed_commands=3+unobserved,
        physical_env_step_attempts=3+unobserved,physical_env_step_returns=3+unobserved)
    row['trace']=row['trace'][:3]
    row['final_state']=row['trace'][-1]['state']
    row['guard']['primitive_commands_authorized']=3+unobserved
    row['guard']['abstained']=True
    goal=np.asarray(row['context_identity']['goal'])
    row['official']=m.labels(row['final_state'],goal)
    row['pose']=m.pose(np.asarray(row['final_state']),goal)
    enc=dict(root_sha256=plan['root_sha256'],method=plan['method'],repetition=0,
        status='blocked_fixed_horizon',attempted_encode_samples=0,failure='fixture partial')
    return plan,row,enc,compiled[plan['root_sha256']]


@pytest.mark.parametrize('unobserved',[0,1])
def test_partial_and_return_without_observation_never_full_h2_or_q(bundle,unobserved):
    result=m.branch_outcome(*make_partial(bundle,unobserved))
    assert result['classification']=='partial_or_failed' and result['full_H2_metrics'] is None
    assert result['observed_endpoint_step']==3 and result['returned_commands']==3+unobserved
    assert result['returned_without_observation']==unobserved
    assert result['observed_endpoint_metrics'] is not None
    rows=rows_from(bundle)
    rows=[result if (r['root_sha256'],r['arm'])==(result['root_sha256'],result['arm']) else r for r in rows]
    summary=m.summarize(rows)
    assert summary['full_H2_pair_count']==3
    assert len(summary['paired'])==4 and len(summary['partial_or_failed_root_ids'])==1
    assert all(v['n']==3 for v in summary['deltas_descriptive'].values())
    assert next(p for p in summary['paired'] if not p['both_full_H2'])['composition_minus_no_composition'] is None


def test_partial_cannot_claim_completed_encoding(bundle):
    plan,row,encoding,compiled=make_partial(bundle)
    encoding.update(status='completed',attempted_encode_samples=1,failure=None,terminal_vector=[0.]*192)
    with pytest.raises(ValueError,match='partial'): m.branch_outcome(plan,row,encoding,compiled)


def test_evaluator_failure_after_ten_steps_keeps_but_quarantines_encoding(bundle):
    protocol,report,compiled=bundle
    plan=protocol['plans'][0]
    row=deepcopy(next(r for r in report['rows'] if (r['root_sha256'],r['method'])==(plan['root_sha256'],plan['method'])))
    enc=next(r for r in report['encodings'] if (r['root_sha256'],r['method'])==(plan['root_sha256'],plan['method']))
    row.update(status='branch_failed',failures=[dict(phase='evaluator',message='fixture label failure')])
    result=m.branch_outcome(plan,row,enc,compiled[plan['root_sha256']])
    assert result['returned_commands']==10 and result['classification']=='partial_or_failed'
    assert result['full_H2_metrics'] is None and result['excluded_complete_endpoint_encoding']


@pytest.mark.parametrize('defect',['selection','action','goal','official','pose','count','guard','state','vector'])
def test_branch_mutations_reject(bundle,defect):
    protocol,report,compiled=bundle
    plan=deepcopy(protocol['plans'][0])
    row=deepcopy(next(r for r in report['rows'] if (r['root_sha256'],r['method'])==(plan['root_sha256'],plan['method'])))
    enc=deepcopy(next(r for r in report['encodings'] if (r['root_sha256'],r['method'])==(plan['root_sha256'],plan['method'])))
    if defect=='selection': row['selection_sha256']='0'*64
    if defect=='action': row['trace'][0]['command'][0]+=.01
    if defect=='goal': row['context_identity']['goal']=[0.]*6
    if defect=='official': row['official']['official_state_distance']+=1
    if defect=='pose': row['pose']['pose_cost']+=1
    if defect=='count': row['physical_env_step_returns']=11
    if defect=='guard': row['guard']['root_id']='0'*64
    if defect=='state': row['final_state'][0]+=1
    if defect=='vector': enc['terminal_vector'][0]=float('nan')
    with pytest.raises(ValueError): m.branch_outcome(plan,row,enc,compiled[plan['root_sha256']])


def test_q_original_projection_goal_not_unweighted_or_new_target():
    goal=np.ones(192); vector=goal.copy(); vector[0]+=2
    q=np.eye(192); q[0,0]=3
    compiled=dict(goal_vectors=[goal.tolist()],scoring=dict(metric=dict(projection=q.tolist())))
    assert np.isclose(m.q_cost(vector,compiled),36/192)
    assert np.isclose(m.q_cost(goal,compiled),0)


def test_descriptive_sign_and_empty_subset_no_inference():
    assert m.describe([-3,-1,0,1])['wins']==2
    assert m.describe([-1,0,1,1],higher=True)['wins']==2
    assert m.describe([])['n']==0 and m.describe([])['mean'] is None
    assert m.BOUNDARY['inferential_tests_performed'] is False
    assert m.BOUNDARY['P2_admission'] is False and m.BOUNDARY['equal_total_compute'] is False


def test_real_arithmetic_audits_durable_artifacts_without_execution(monkeypatch):
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    for name in ('require_cuda_allocation','load_compatible_pretrained','collect_nonterminal_boundary_contact_roots'):
        monkeypatch.setattr(base,name,lambda *a,**k:pytest.fail('execution forbidden'))
    monkeypatch.setattr(base.gym,'make',lambda *a,**k:pytest.fail('real simulator forbidden'))
    reader=m.evidence.EvidenceReader()
    result=m.load_and_compute(m.DEFAULT_PHYSICAL,reader)
    assert result['full_H2_pair_count']==4 and result['selected_path_env_step_attempts']==80
    assert result['new_neural_calls']==result['new_physical_steps']==0
    assert result['planning_totals']['actual_unique_predict_calls']==1008
    assert any(p.endswith('.events.jsonl') for p in reader.hashes)
    assert any(p.endswith('.png') for p in reader.hashes)
    assert 'physical.run(' not in inspect.getsource(m)
