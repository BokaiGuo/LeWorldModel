"""Known saved calibration evidence / arithmetic only; never neural or physics."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.spectral_reachability import p2_mppi_matched_comparison_20261027_v1 as m


@pytest.fixture(scope='module')
def bundle():
    if not (m.PROJECT/'results'/m.MPPI_RUN/'report.json').is_file():
        pytest.skip('explicit main-thread physical report unavailable')
    reader = m.evidence.EvidenceReader()
    data = m.load_inputs(m.PROJECT,reader)
    reader.unchanged()
    return data


def test_actual_three_arm_saved_comparison(bundle):
    rows = m.build_rows(bundle)
    result = m.compare_rows(rows)
    assert result['paired_roots'] == 4 and result['branches'] == 12
    assert result['official_successes'] == {a:0 for a in m.ARMS}
    assert not result['P2_admission'] and not result['official_MPPI']
    assert not result['fair_total_budget_comparison']
    assert result['new_neural_calls'] == result['new_physical_steps'] == 0
    pose = result['paired_summaries']['cem']['object_pose_cost']
    assert (pose['left_wins'],pose['ties'],pose['left_losses']) == (1,0,3)
    assert np.isclose(pose['mean'],38.688984526301034,rtol=0,atol=1e-10)
    assert result['paired_summaries']['random']['object_pose_cost']['left_wins'] == 2
    assert len(result['paired_state_error_summaries']['cem']) == 7
    assert 'no CI' in result['uncertainty_scope']


def test_all_array_orders_are_irrelevant_to_root_join(bundle):
    changed = deepcopy(bundle)
    changed['compilation']['rows'].reverse()
    for group in changed['groups']:
        for field in ('rows','encodings'): group['report'][field].reverse()
        group['plans'].reverse()
        group['planning']['rows'].reverse()
        group['planning_protocol']['prepared'].reverse()
        for row in group['planning']['rows']: row['rows'].reverse()
    original = m.compare_rows(m.build_rows(bundle))
    assert m.compare_rows(m.build_rows(changed)) == original
    assert m.compare_rows(m.build_rows(bundle)[::-1]) == original


@pytest.mark.parametrize('defect',['missing','duplicate','wrong_root','wrong_arm','goal','context','state',
    'action','step_order','selection','pose','official','encoding','missing_encoding','encode_root',
    'count','abort','budget','budget_root','encoder_work','predict_score','planning_total','planning_calls'])
def test_saved_evidence_mutations_reject(bundle,defect):
    data = deepcopy(bundle)
    group = data['groups'][0]
    branch = group['report']['rows'][0]
    arm = group['planning']['rows'][0]['rows'][0]
    if defect == 'missing': group['report']['rows'].pop()
    if defect == 'duplicate': group['report']['rows'][-1] = deepcopy(branch)
    if defect == 'wrong_root': branch['root_sha256'] = '0'*64
    if defect == 'wrong_arm': branch['method'] = 'cem'
    if defect == 'goal': branch['context_identity']['goal_rgb_sha256'] = '0'*64
    if defect == 'context': branch['context_identity']['history_rgb_sha256'] = '0'*64
    if defect == 'state': branch['final_state'][0] += 1
    if defect == 'action': branch['trace'][0]['command'][0] += .01
    if defect == 'step_order': branch['trace'][0]['primitive_step'] = 2
    if defect == 'selection': branch['selection_sha256'] = '0'*64
    if defect == 'pose': branch['pose']['pose_cost'] += .01
    if defect == 'official': branch['official']['official_success'] = True
    if defect == 'encoding': group['report']['encodings'][0]['terminal_vector'][0] = float('nan')
    if defect == 'missing_encoding': group['report']['encodings'].pop()
    if defect == 'encode_root': group['report']['encodings'][0]['root_sha256'] = '0'*64
    if defect == 'count': branch['completed_commands'] = 9
    if defect == 'abort': branch['status'] = 'observation_unavailable'
    if defect == 'budget': arm['planning_predict_samples'] = 79
    if defect == 'budget_root': arm['actual_prediction_budget']['binding']['root_id'] = '0'*64
    if defect == 'encoder_work': arm['action_encoder_work']['processed_context_tokens'] -= 1
    if defect == 'predict_score': arm['selected_model_cost'] += 1
    if defect == 'planning_total': group['planning']['actual_predict_samples'] = 335
    if defect == 'planning_calls': group['planning']['actual_predict_calls'] = 335
    with pytest.raises(ValueError): m.build_rows(data)


@pytest.mark.parametrize('field',['root_binding_sha256','metric_sha256','goal_vectors_sha256','physical_identity','budget','action_domain'])
def test_cross_arm_bindings_must_match(bundle,field):
    rows = m.build_rows(bundle)
    rows[0][field] = 'mismatch'
    with pytest.raises(ValueError,match='cross-arm'): m.compare_rows(rows)


def test_explicit_difference_direction_and_descriptive_uncertainty():
    stats = m.describe([-3.,-1.,1.,3.])
    assert stats['mean'] == stats['median'] == 0
    assert stats['sample_sd'] == pytest.approx(np.std([-3,-1,1,3],ddof=1))
    assert stats['left_wins'] == stats['left_losses'] == 2
    assert m.describe([1.,1.,1.,1.],higher_better=True)['left_wins'] == 4
    with pytest.raises(ValueError): m.describe([1.,2.,3.])
    with pytest.raises(ValueError): m.describe([0.,1.,2.,float('nan')])


def test_all_persisted_artifacts_verified_without_execution(bundle,monkeypatch):
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    for name in ('require_cuda_allocation','load_compatible_pretrained','collect_nonterminal_boundary_contact_roots'):
        monkeypatch.setattr(base,name,lambda *a,**kw:pytest.fail('execution requested'))
    monkeypatch.setattr(base.gym,'make',lambda *a,**kw:pytest.fail('environment creation'))
    reader = m.evidence.EvidenceReader()
    fresh = m.load_inputs(m.PROJECT,reader)
    for group in fresh['groups']: m.audit_artifacts(group,reader)
    reader.unchanged()
    assert len(reader.hashes) > 450


@pytest.mark.parametrize('defect',['gate','branch','encoding','journal'])
def test_artifact_disagreement_rejects(bundle,monkeypatch,defect):
    group = deepcopy(bundle['groups'][0])
    if defect == 'gate': group['report']['gates'] = {}
    reader = m.evidence.EvidenceReader()
    original = reader.load
    def altered(path,*args):
        value = original(path,*args)
        if defect == 'branch' and str(path).endswith('.physical.json'): value['final_state'][0] += 1
        if defect == 'encoding' and str(path).endswith('.encoding.json'): value['terminal_vector'][0] += 1
        return value
    monkeypatch.setattr(reader,'load',altered)
    if defect == 'journal': group['report']['rows'][0]['journal_sha256'] = '0'*64
    with pytest.raises(ValueError): m.audit_artifacts(group,reader)


def test_output_is_exclusive_and_missing_input_retains_invalid_report(tmp_path):
    output = tmp_path/'new'
    result = m.run(tmp_path,output)
    assert result['status'] == 'invalid_evidence' and not result['P2_admission']
    assert (output/'report.json').is_file() and (output/'report.md').is_file()
    with pytest.raises(FileExistsError): m.run(tmp_path,output)


def test_actual_run_and_source_seal_in_new_output(bundle,tmp_path):
    output = tmp_path/'audit'
    result = m.run(m.PROJECT,output)
    assert result['status'] == 'saved_calibration_comparison_completed'
    assert result['source_verification']['status'] == 'verified'
    saved = json.loads((output/'report.json').read_text())
    assert saved == result
    text = (output/'report.md').read_text()
    assert 'not official MPPI' in text and '40 complete H2' in text
