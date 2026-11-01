"""Saved five-arm evidence + scalar/array arithmetic; no model or physics runs."""
from copy import deepcopy
import json

import numpy as np
import pytest

from experiments.spectral_reachability import p2_gradient_mpc_matched_comparison_20261027_v1 as m


@pytest.fixture(scope='module')
def bundle():
    reader = m.evidence.EvidenceReader()
    data = m.load_inputs(m.PROJECT,reader)
    m.audit_artifacts(data,reader)
    reader.unchanged()
    return data


def test_actual_five_arm_complete_root_pairs_and_overhead(bundle):
    result = m.compare_rows(m.build_rows(bundle))
    assert result['paired_roots']==4 and result['branches']==20
    assert len(result['paired_summaries'])==4
    assert len(result['paired_state_error_summaries']['cem'])==7
    assert not any(result[k] for k in ('P1_admission','P2_admission','composition_admission',
        'physical_efficacy_verified','fair_total_budget_comparison','prediction_budget_matched',
        'same_backend_established','official_gradient_implementation','official_MPPI'))
    assert result['new_neural_calls']==result['new_physical_steps']==0
    cost = result['cost_accounting']
    assert cost['four_root_total_predict_samples']=={a:352 if a==m.previous.SRA_ARM else 336 for a in m.ARMS}
    assert cost['gradient_autograd_grad_calls']==cost['gradient_action_optimizer_steps']==156
    assert cost['gradient_model_optimizer_steps']==0
    assert cost['sra_jacobian_calls']==4 and cost['sra_svd_calls']==8
    assert cost['sra_scalar_output_vjp_directions']==768
    assert cost['per_root']['cem']['autograd_grad_calls'] is None
    assert result['backend_contracts'][m.GRADIENT_ARM]==m.MATH_POLICY
    assert all('unknown' in result['backend_contracts'][a] for a in m.BASE_ARMS)
    assert all(0<=result['official_successes'][a]<=4 for a in m.ARMS)
    assert 'no CI' in result['uncertainty_scope']


def test_result_array_orders_do_not_define_pairs(bundle):
    data = deepcopy(bundle)
    data['compilation']['rows'].reverse()
    for group in (*data['groups'],data['sra'],data['gradient']):
        group['plans'].reverse()
        group['planning']['rows'].reverse()
        group['report']['rows'].reverse()
        group['report']['encodings'].reverse()
        if group in data['groups']:
            for row in group['planning']['rows']: row['rows'].reverse()
    expected = m.compare_rows(m.build_rows(bundle))
    assert m.compare_rows(m.build_rows(data))==expected
    assert m.compare_rows(m.build_rows(bundle)[::-1])==expected


@pytest.mark.parametrize('defect',['missing','duplicate','wrong_root','wrong_arm','root_index','goal',
    'context','state','action','step_order','selection','pose','official','encoding','encoding_nan',
    'missing_encoding','encode_root','count','abort','budget','budget_root','encoder_work',
    'predict_score','planning_total','autograd','vjp','update_count','candidate','gradient',
    'logit_update','winner','provenance','fixture','overhead','backend','schedule','admission'])
def test_gradient_saved_evidence_mutations_reject(bundle,defect):
    data = deepcopy(bundle)
    group = data['gradient']
    branch = group['report']['rows'][0]
    row = group['planning']['rows'][0]
    result = row['result']
    if defect=='missing': group['report']['rows'].pop()
    if defect=='duplicate': group['report']['rows'][-1]=deepcopy(branch)
    if defect=='wrong_root': branch['root_sha256']='0'*64
    if defect=='wrong_arm': branch['method']='cem'
    if defect=='root_index': branch['root_index']=999
    if defect=='goal': branch['context_identity']['goal_rgb_sha256']='0'*64
    if defect=='context': branch['context_identity']['history_rgb_sha256']='0'*64
    if defect=='state': branch['final_state'][0] += 1
    if defect=='action': branch['trace'][0]['command'][0] += .1
    if defect=='step_order': branch['trace'][0]['primitive_step']=2
    if defect=='selection': branch['selection_sha256']='0'*64
    if defect=='pose': branch['pose']['pose_cost'] += .01
    if defect=='official': branch['official']['official_success']=not branch['official']['official_success']
    if defect=='encoding': group['report']['encodings'][0]['terminal_vector'].pop()
    if defect=='encoding_nan': group['report']['encodings'][0]['terminal_vector'][0]=float('nan')
    if defect=='missing_encoding': group['report']['encodings'].pop()
    if defect=='encode_root': group['report']['encodings'][0]['root_sha256']='0'*64
    if defect=='count': branch['completed_commands']=9
    if defect=='abort': branch['status']='observation_unavailable'
    if defect=='budget': result['planning_predict_samples']=79
    if defect=='budget_root': row['actual_prediction_budget']['binding']['root_id']='0'*64
    if defect=='encoder_work': row['action_encoder_calls']=83
    if defect=='predict_score': result['selected_model_cost'] += 1
    if defect=='planning_total': group['planning']['actual_predict_calls']=335
    if defect=='autograd': result['completed_autograd_grad_calls']=38
    if defect=='vjp': group['planning']['completed_scalar_objective_vjp_directions']=155
    if defect=='update_count': result['optimizer_steps']=38
    if defect=='candidate': result['attempts'][0]['commands'][0][0][0] += .1
    if defect=='gradient': result['attempts'][0]['raw_gradient'][0] += 1
    if defect=='logit_update': result['attempts'][0]['next_logits'][0] += .1
    if defect=='winner': result['selected_candidate_id']='random_00'
    if defect=='provenance': result['witness']['receipt']['provenance']['source_sha256']='0'*64
    if defect=='fixture': result['fixture_only']=True
    if defect=='overhead': group['report']['planning_accounting']['autograd_grad_calls']=0
    if defect=='backend': group['planning_protocol']['schedule']['attention_backend']='automatic'
    if defect=='schedule': group['planning_protocol']['schedule']['learning_rate']=.2
    if defect=='admission': row['P2_admission']=True
    with pytest.raises(ValueError): m.build_rows(data)


@pytest.mark.parametrize('field',['root_binding_sha256','metric_sha256','goal_vectors_sha256','physical_identity','action_domain'])
def test_cross_arm_fixed_identity_must_match(bundle,field):
    rows = m.build_rows(bundle)
    rows[-1][field]='mismatch'
    with pytest.raises(ValueError,match='cross-arm'): m.compare_rows(rows)


@pytest.mark.parametrize('field',['autograd_grad_calls','scalar_objective_vjp_directions','action_optimizer_steps',
    'total_predict_samples','planning_predict_samples','processed_context_tokens'])
def test_no_masked_AD_or_prediction_overhead(bundle,field):
    rows = m.build_rows(bundle)
    row = next(r for r in rows if r['arm']==m.GRADIENT_ARM)
    row['budget'][field]=0
    with pytest.raises(ValueError): m.compare_rows(rows)


@pytest.mark.parametrize('defect',['gradient_backend','invent_reference_backend','invent_reference_AD'])
def test_backend_and_unknown_reference_resources_are_not_assumed_equal(bundle,defect):
    rows = m.build_rows(bundle)
    row = next(r for r in rows if r['arm']==(m.GRADIENT_ARM if defect=='gradient_backend' else 'cem'))
    if defect=='gradient_backend': row['backend_policy']='automatic'
    if defect=='invent_reference_backend': row['backend_policy']=m.MATH_POLICY
    if defect=='invent_reference_AD': row['budget']['autograd_grad_calls']=0
    with pytest.raises(ValueError): m.compare_rows(rows)


@pytest.mark.parametrize('defect',['candidate','packet','cost','derivative','missing_AD','extra_last_AD','chronology'])
def test_AD_journal_exact_candidate_packet_cost_and_backward_receipts(bundle,defect):
    group = bundle['gradient']
    row = group['planning']['rows'][0]
    events = deepcopy(group['journals'][row['root_sha256']])
    if defect=='candidate': next(e for e in events if 'candidate_id' in e)['cost'] += 1
    if defect=='packet': next(e for e in events if e.get('kind')=='candidate_packet')['packet']=1
    if defect=='cost': next(e for e in events if e.get('kind')=='candidate_cost')['cost'] += 1
    if defect=='derivative': next(e for e in events if e.get('kind')=='autograd_grad')['extra_forward_requested']=2
    if defect=='missing_AD': events.pop(next(i for i,e in enumerate(events) if e.get('kind')=='autograd_grad'))
    if defect=='extra_last_AD': events.append(dict(kind='autograd_grad',candidate_index=39,completed=True))
    if defect=='chronology': events[2],events[3]=events[3],events[2]
    with pytest.raises(ValueError): m.validate_ad_journal(row,events)


@pytest.mark.parametrize('defect',['gate','branch','encoding','journal','overhead','admission','partial'])
def test_durable_artifact_disagreement_is_not_silently_accepted(bundle,monkeypatch,defect):
    data = deepcopy(bundle)
    group = data['gradient']
    if defect=='gate': group['report']['gates']={}
    if defect=='journal': group['report']['rows'][0]['journal_sha256']='0'*64
    if defect=='overhead': group['report']['planning_accounting']['autograd_grad_calls']=0
    if defect=='admission': group['report']['P2_admission']=True
    if defect=='partial': group['report']['full_H2_all']=False
    reader = m.evidence.EvidenceReader()
    original = reader.load
    def altered(path,*args):
        value = original(path,*args)
        if m.GRADIENT_RUN in str(path):
            if defect=='branch' and str(path).endswith('.physical.json'): value['final_state'][0] += 1
            if defect=='encoding' and str(path).endswith('.encoding.json'): value['terminal_vector'][0] += 1
        return value
    monkeypatch.setattr(reader,'load',altered)
    with pytest.raises(ValueError): m.audit_artifacts(data,reader)


def test_explicit_delta_direction_sample_sd_and_state_coordinate_errors(bundle):
    result = m.compare_rows(m.build_rows(bundle))
    for pair in result['rows']:
        for arm in m.BASE_ARMS:
            for metric in m.METRICS:
                assert pair['gradient_minus'][arm][metric]==pair['arms'][m.GRADIENT_ARM][metric]-pair['arms'][arm][metric]
            assert np.array_equal(pair['gradient_minus_state_absolute_error'][arm],
                np.asarray(pair['arms'][m.GRADIENT_ARM]['state_absolute_error'])-np.asarray(pair['arms'][arm]['state_absolute_error']))
    values = [r['gradient_minus']['cem']['object_pose_cost'] for r in result['rows']]
    stats = result['paired_summaries']['cem']['object_pose_cost']
    assert stats['mean']==pytest.approx(np.mean(values))
    assert stats['sample_sd']==pytest.approx(np.std(values,ddof=1))
    assert stats['left_wins']+stats['ties']+stats['left_losses']==4


def test_saved_load_artifact_audit_and_arithmetic_request_no_execution(monkeypatch):
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    def forbidden(*args,**kwargs): pytest.fail('neural or physics execution requested')
    for name in ('require_cuda_allocation','load_compatible_pretrained','collect_nonterminal_boundary_contact_roots'):
        monkeypatch.setattr(base,name,forbidden)
    monkeypatch.setattr(base.gym,'make',forbidden)
    monkeypatch.setattr(m.physical,'run',forbidden)
    monkeypatch.setattr(m.planner,'cuda',forbidden)
    monkeypatch.setattr(m.planner.CudaObjective,'evaluate',forbidden)
    reader = m.evidence.EvidenceReader()
    data = m.load_inputs(m.PROJECT,reader)
    m.audit_artifacts(data,reader)
    result = m.compare_rows(m.build_rows(data))
    assert result['branches']==20
    reader.unchanged()
    assert len(reader.hashes)>900


def test_missing_input_retains_invalid_report_and_output_is_exclusive(tmp_path):
    output = tmp_path/m.TAG
    result = m.run(tmp_path,output)
    assert result['status']=='invalid_evidence' and not result['evidence_integrity_valid']
    assert not result['P2_admission'] and not result['composition_admission']
    assert (output/'report.json').is_file() and (output/'report.md').is_file()
    with pytest.raises(FileExistsError): m.run(tmp_path,output)


def test_actual_saved_audit_roundtrip_source_seal_and_CLI_failure(bundle,tmp_path):
    output = tmp_path/m.TAG
    result = m.run(m.PROJECT,output)
    assert result['evidence_integrity_valid'],result.get('integrity_errors')
    assert result['source_verification']['status']=='verified'
    assert json.loads((output/'report.json').read_text())==result
    text = (output/'report.md').read_text()
    assert 'MATH backend' in text and 'not official MPPI' in text and 'composition_admission=false' in text
    assert '156 autograd.grad' in text and 'No equal total budget' in text
    assert m.main(['--project',str(tmp_path),'--output',str(tmp_path/(m.TAG+'_missing'))])==1

