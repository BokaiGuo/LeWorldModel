"""Saved true calibration artifacts and array arithmetic only; no neural/physics."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.spectral_reachability import p2_sra_matched_comparison_20261027_v1 as m


@pytest.fixture(scope='module')
def bundle():
    assert (m.PROJECT/'results'/m.SRA_RUN/'report.json').is_file(), 'main-thread SRA physical required'
    reader = m.evidence.EvidenceReader()
    value = m.load_inputs(m.PROJECT,reader)
    m.audit_artifacts(value,reader)
    reader.unchanged()
    return value


def test_real_four_arm_complete_calibration_and_cost_boundary(bundle):
    result = m.compare_rows(m.build_rows(bundle))
    assert result['paired_roots'] == 4 and result['branches'] == 16
    assert result['official_successes'] == {a:0 for a in m.ARMS}
    assert not any(result[k] for k in ('P1_admission','P2_admission','composition_admission',
        'fair_total_budget_comparison','prediction_budget_matched','official_MPPI','physical_efficacy_verified'))
    assert result['new_neural_calls'] == result['new_physical_steps'] == 0
    costs = result['cost_accounting']
    assert costs['four_root_total_predict_samples'] == {a:352 if a==m.SRA_ARM else 336 for a in m.ARMS}
    assert costs['sra_additional_predict_samples_four_roots'] == 16
    assert costs['sra_jacobian_calls'] == 4 and costs['sra_svd_calls'] == 8
    assert costs['sra_scalar_output_vjp_directions'] == 768
    assert costs['per_root']['cem']['svd_calls'] is None
    assert len(result['paired_state_error_summaries']['cem']) == 7
    assert 'no CI' in result['uncertainty_scope']
    assert 'No composition arm' in result['composition_scope']


def test_stable_root_join_ignores_all_saved_result_array_orders(bundle):
    data = deepcopy(bundle)
    data['compilation']['rows'].reverse()
    for group in (*data['groups'],data['sra']):
        group['plans'].reverse()
        group['planning']['rows'].reverse()
        group['report']['rows'].reverse()
        group['report']['encodings'].reverse()
        if group is not data['sra']:
            for row in group['planning']['rows']: row['rows'].reverse()
    original = m.compare_rows(m.build_rows(bundle))
    assert m.compare_rows(m.build_rows(data)) == original
    assert m.compare_rows(m.build_rows(bundle)[::-1]) == original


@pytest.mark.parametrize('defect',['missing','duplicate','wrong_root','wrong_arm','root_index','goal',
    'context','state','action','step_order','selection','pose','official','encoding','encoding_nan',
    'missing_encoding','encode_root','count','abort','budget','budget_root','encoder_work',
    'predict_score','planning_total','jacobian','svd','vjp','candidate','bank','winner','provenance','fixture','overhead'])
def test_sra_saved_evidence_mutations_reject(bundle,defect):
    data = deepcopy(bundle)
    group = data['sra']
    branch = group['report']['rows'][0]
    row = group['planning']['rows'][0]
    result = row['result']
    if defect == 'missing': group['report']['rows'].pop()
    if defect == 'duplicate': group['report']['rows'][-1] = deepcopy(branch)
    if defect == 'wrong_root': branch['root_sha256'] = '0'*64
    if defect == 'wrong_arm': branch['method'] = 'cem'
    if defect == 'root_index': branch['root_index'] = 999
    if defect == 'goal': branch['context_identity']['goal_rgb_sha256'] = '0'*64
    if defect == 'context': branch['context_identity']['history_rgb_sha256'] = '0'*64
    if defect == 'state': branch['final_state'][0] += 1
    if defect == 'action': branch['trace'][0]['command'][0] += .1
    if defect == 'step_order': branch['trace'][0]['primitive_step'] = 2
    if defect == 'selection': branch['selection_sha256'] = '0'*64
    if defect == 'pose': branch['pose']['pose_cost'] += .01
    if defect == 'official': branch['official']['official_success'] = True
    if defect == 'encoding': group['report']['encodings'][0]['terminal_vector'].pop()
    if defect == 'encoding_nan': group['report']['encodings'][0]['terminal_vector'][0] = float('nan')
    if defect == 'missing_encoding': group['report']['encodings'].pop()
    if defect == 'encode_root': group['report']['encodings'][0]['root_sha256'] = '0'*64
    if defect == 'count': branch['completed_commands'] = 9
    if defect == 'abort': branch['status'] = 'observation_unavailable'
    if defect == 'budget': result['planning_predict_samples'] = 79
    if defect == 'budget_root': row['actual_prediction_budget']['binding']['root_id'] = '0'*64
    if defect == 'encoder_work': row['action_encoder_processed_context_token_samples'] -= 1
    if defect == 'predict_score': result['selected_model_cost'] += 1
    if defect == 'planning_total': group['planning']['actual_predict_calls'] = 336
    if defect == 'jacobian': row['compile_receipt']['completed_jacobian_calls'] = 0
    if defect == 'svd': row['compile_receipt']['completed_svd_calls'] = 0
    if defect == 'vjp': row['compile_receipt']['completed_scalar_output_vjp_directions'] = 0
    if defect == 'candidate': result['attempts'][0]['commands'][0][0][0] += .1
    if defect == 'bank': group['banks'][row['root_sha256']]['logits'][0][0] += 1
    if defect == 'winner': result['selected_candidate_id'] = 'random_00'
    if defect == 'provenance': result['witness']['receipt']['provenance']['source_sha256'] = '0'*64
    if defect == 'fixture': result['fixture_only'] = True
    if defect == 'overhead': group['report']['planning_accounting']['per_root']['total_predict_samples'] = 84
    with pytest.raises(ValueError): m.build_rows(data)


@pytest.mark.parametrize('field',['root_binding_sha256','metric_sha256','goal_vectors_sha256','physical_identity','action_domain'])
def test_cross_arm_bindings_not_array_position_must_match(bundle,field):
    rows = m.build_rows(bundle)
    rows[-1][field] = 'mismatch'
    with pytest.raises(ValueError,match='cross-arm'): m.compare_rows(rows)


@pytest.mark.parametrize('field',['planning_predict_samples','compilation_predict_samples',
    'total_predict_samples','action_encoder_calls','processed_context_tokens','svd_calls'])
def test_cannot_mask_sra_as_equal_total_budget(bundle,field):
    rows = m.build_rows(bundle)
    row = next(r for r in rows if r['arm']==m.SRA_ARM)
    row['budget'][field] = 84
    with pytest.raises(ValueError): m.compare_rows(rows)


def test_exact_difference_direction_and_no_pooled_state_units(bundle):
    rows = m.build_rows(bundle)
    result = m.compare_rows(rows)
    for pair in result['rows']:
        for arm in m.BASE_ARMS:
            for metric in m.METRICS:
                assert pair['sra_minus'][arm][metric] == pair['arms'][m.SRA_ARM][metric]-pair['arms'][arm][metric]
    costs = result['paired_summaries']['cem']['object_pose_cost']
    values = [p['sra_minus']['cem']['object_pose_cost'] for p in result['rows']]
    assert costs['mean'] == pytest.approx(np.mean(values))
    assert costs['sample_sd'] == pytest.approx(np.std(values,ddof=1))
    assert costs['left_wins']+costs['ties']+costs['left_losses'] == 4
    assert 'raw angle not circular' in result['state_error_scope']


@pytest.mark.parametrize('defect',['gate','branch','encoding','journal','overhead','admission'])
def test_durable_artifacts_or_claim_disagreement_refused(bundle,monkeypatch,defect):
    data = deepcopy(bundle)
    group = data['sra']
    if defect == 'gate': group['report']['gates'] = {}
    if defect == 'journal': group['report']['rows'][0]['journal_sha256'] = '0'*64
    if defect == 'overhead': group['report']['planning_accounting']['per_root']['total_predict_samples'] = 84
    if defect == 'admission': group['report']['P2_admission'] = True
    reader = m.evidence.EvidenceReader()
    original = reader.load
    def altered(path,*args):
        value = original(path,*args)
        if m.SRA_RUN in str(path):
            if defect=='branch' and str(path).endswith('.physical.json'): value['final_state'][0] += 1
            if defect=='encoding' and str(path).endswith('.encoding.json'): value['terminal_vector'][0] += 1
        return value
    monkeypatch.setattr(reader,'load',altered)
    with pytest.raises(ValueError): m.audit_artifacts(data,reader)


def test_real_load_and_artifacts_forbid_all_execution(monkeypatch):
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    def forbidden(*a,**kw): pytest.fail('neural/planning/physics execution requested')
    for name in ('require_cuda_allocation','load_compatible_pretrained','collect_nonterminal_boundary_contact_roots'):
        monkeypatch.setattr(base,name,forbidden)
    monkeypatch.setattr(base.gym,'make',forbidden)
    monkeypatch.setattr(m.sra,'run',forbidden)
    monkeypatch.setattr(m.sra.sampling,'cuda',forbidden)
    reader = m.evidence.EvidenceReader()
    data = m.load_inputs(m.PROJECT,reader)
    m.audit_artifacts(data,reader)
    m.compare_rows(m.build_rows(data))
    reader.unchanged()
    assert len(reader.hashes) > 650


def test_missing_input_retains_invalid_report_and_no_overwrite(tmp_path):
    output = tmp_path/m.TAG
    result = m.run(tmp_path,output)
    assert not result['evidence_integrity_valid'] and result['status']=='invalid_evidence'
    assert not result['P2_admission'] and not result['composition_admission']
    assert (output/'report.json').is_file() and (output/'report.md').is_file()
    with pytest.raises(FileExistsError): m.run(tmp_path,output)


def test_actual_audit_output_and_source_seal(bundle,tmp_path):
    output = tmp_path/m.TAG
    result = m.run(m.PROJECT,output)
    assert result['evidence_integrity_valid'], result.get('integrity_errors')
    assert result['source_verification']['status']=='verified'
    assert json.loads((output/'report.json').read_text()) == result
    text = (output/'report.md').read_text()
    assert 'not official MPPI' in text and 'No equal-total-budget' in text
    assert '40 complete H2' in text and 'composition_admission=false' in text
    assert m.main(['--project',str(tmp_path),'--output',str(tmp_path/(m.TAG+'_missing'))]) == 1
