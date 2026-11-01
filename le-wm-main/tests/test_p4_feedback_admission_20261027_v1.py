"""Saved-data arithmetic/receipt tests only; never model/simulator experiments."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import p4_feedback_admission_20261027_v1 as p

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT/'results'/p.SOURCE_RUN


@pytest.fixture(scope='module')
def saved_report():
    return p.strict_json((FOLDER/'report.json').read_bytes())


@pytest.fixture(scope='module')
def round_fixture(saved_report):
    t = p.Inputs()
    compilation = t.json(ROOT/'results/pusht_d2p_metric_program_preflight4_v1_20261026_cuda/report.json')
    prefix = FOLDER/'root_0.actual.round_1'
    previous = FOLDER/'root_0.actual.round_0'
    return dict(decision=t.json(str(prefix)+'.decision.json'),
        rgb=t.npy(str(prefix)+'.rgb.npy',(3,224,224,3),np.uint8),
        between=t.npy(str(prefix)+'.between.npy',(2,5,2)),
        terminal=t.npy(str(prefix)+'.terminal.npy',(224,224,3),np.uint8),
        execution=t.json(str(prefix)+'.execution.json'),
        bank=np.asarray(saved_report['protocol']['banks'][0]),compiled=compilation['rows'][0]['compiled'],
        expected_schedule=p.schedule(0)[40:],
        previous=t.json(str(previous)+'.decision.json'),
        previous_rgb=t.npy(str(previous)+'.rgb.npy',(3,224,224,3),np.uint8),
        previous_between=t.npy(str(previous)+'.between.npy',(2,5,2)),
        previous_terminal=t.npy(str(previous)+'.terminal.npy',(224,224,3),np.uint8))


@pytest.fixture(scope='module')
def audit_result():
    return p.audit(FOLDER/'report.json')


def test_import_is_model_and_simulator_free():
    code = ('import sys; from experiments.spectral_reachability import p4_feedback_admission_20261027_v1; '
            'assert not any(n in sys.modules for n in ("torch","gymnasium","stable_worldmodel"))')
    done = subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,capture_output=True,text=True)
    assert done.returncode == 0, done.stderr


@pytest.mark.parametrize('raw', ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}'])
def test_strict_json_rejects_duplicate_or_nonfinite(raw):
    with pytest.raises(ValueError): p.strict_json(raw)


def test_normalization_time_major_no_div100():
    raw = np.arange(20,dtype=np.float32).reshape(2,5,2)/20
    expected = ((raw-p.MEAN)/p.STD).reshape(2,10)
    assert np.array_equal(p.normalize(raw),expected)
    assert not np.array_equal(p.normalize(raw),p.normalize(raw/100))
    assert not np.array_equal(p.normalize(raw),p.normalize(raw.transpose(0,2,1).reshape(2,5,2)))


@pytest.mark.parametrize('raw', [np.ones((2,2)),np.full((5,2),1.01),np.full((5,2),np.nan),np.ones((5,2),bool)])
def test_bad_packet_fails(raw):
    with pytest.raises(ValueError): p.normalize(raw)


def test_full_actual_interface_saved_witness(round_fixture):
    result = p.round_interface(**round_fixture)
    assert result['rgb_history_verified'] and result['normalization_verified']
    assert result['full_packet_completed'] and result['observed_context_used']
    assert result['model_cost_recomputed'] == pytest.approx(1.4384439866036205)
    assert result['new_model_replay'] is False and result['rgb_reencoding_verified'] is False


@pytest.mark.parametrize('fault', ['frame_order','imagined_endpoint','stack_hash','between_shift','extra_div100',
    'action_order','context_hash','wrong_source','wrong_flag','rollout1','mid_shift','end_shift',
    'terminal_history','selected_command','executed_command','horizon','score','replay_bool',
    'trace','attempt','terminal','guard_root','guard_hidden','round_without_previous','nan'])
def test_interface_mutations_fail_closed(round_fixture, fault):
    data = deepcopy(round_fixture)
    d, ex = data['decision'], data['execution']
    if fault == 'frame_order': data['rgb'] = data['rgb'][::-1].copy()
    elif fault == 'imagined_endpoint': data['previous_terminal'] = np.zeros_like(data['previous_terminal'])
    elif fault == 'stack_hash': d['observed']['stack_sha256'] = 'bad'
    elif fault == 'between_shift': data['previous_between'][1] += .01
    elif fault == 'extra_div100': data['between'] /= 100
    elif fault == 'action_order': d['observed']['encoded_context']['action_history'][0].reverse()
    elif fault == 'context_hash': d['planning_context_sha256'] = 'bad'
    elif fault == 'wrong_source':
        d['planning_context'] = data['previous']['model_next_context']
        d['planning_context_sha256'] = p.content_sha(d['planning_context'])
    elif fault == 'wrong_flag': d['observed_context_used_for_selection'] = False
    elif fault == 'rollout1': d['selected']['actions'] = d['selected']['actions'][:5]
    elif fault == 'mid_shift': d['model_next_context']['latent_history'][0][0] += .1
    elif fault == 'end_shift': d['selected']['terminal']['latent_history'][1][0] += .1
    elif fault == 'terminal_history': d['selected']['terminal']['action_history'].reverse()
    elif fault == 'selected_command': d['selected']['actions'][0][0] += .1
    elif fault == 'executed_command': d['executed_packet'][0][0] += .1
    elif fault == 'horizon': d['search_horizon_tokens'] = 1
    elif fault == 'score': d['selected']['cost'] += .01
    elif fault == 'replay_bool': d['replay_exact'] = [1,1]
    elif fault == 'trace': ex['trace'][0]['command'][0] += .1
    elif fault == 'attempt': ex['attempt_before_step'].pop()
    elif fault == 'terminal': data['terminal'][0,0,0] ^= 1
    elif fault == 'guard_root': ex['guard']['root_id'] = 'other'
    elif fault == 'guard_hidden': ex['guard']['observations'][0]['canonical_agent_pixels'] = 0
    elif fault == 'round_without_previous': data['previous'] = None
    else: d['model_next_context']['latent_history'][-1][0] = float('nan')
    with pytest.raises((ValueError, TypeError, KeyError)):
        p.round_interface(**data)


def test_real_accounting_exact_per_round_units(saved_report):
    for row in saved_report['rows']:
        rounds = p.accounting(row)
        assert len(rounds) == 2
        for item in rounds:
            assert item['planning_predict_calls'] == item['planning_predict_samples'] == 80
            assert item['evaluation_predict_calls'] == item['evaluation_predict_samples'] == 4
            assert item['processed_context_tokens'] == 252
            assert item['rgb_encode_frames'] == 3 and item['physical_step_returns'] == 5


@pytest.mark.parametrize('fault', ['missing_scope','duplicate_scope','horizon','sample','failed','bool_complete',
    'round','history_length','fake_bool_hist','method','missing_event','event_phase','event_failure',
    'event_tokens','encode_frames','encode_hook','action_hook','physics','fake_total','negative_wall'])
def test_accounting_does_not_trust_branch_totals_or_gates(saved_report,fault):
    row = deepcopy(saved_report['rows'][0])
    scope = row['predict'][0]; event = row['resource_events'][1]
    if fault == 'missing_scope': row['predict'].pop()
    elif fault == 'duplicate_scope': row['predict'][-1] = row['predict'][0]
    elif fault == 'horizon': scope['horizon_tokens'] = 1
    elif fault == 'sample': scope['predict_samples'] = 1
    elif fault == 'failed': scope['failed_predict_calls'] = 1
    elif fault == 'bool_complete': scope['completed'] = 1
    elif fault == 'round': scope['round'] = 1
    elif fault == 'history_length': scope['context_lengths'] = {'2':2}
    elif fault == 'fake_bool_hist': row['predict'][41]['context_lengths'] = {'3':True}
    elif fault == 'method': scope['method'] = 'other'
    elif fault == 'missing_event': row['resource_events'].pop()
    elif fault == 'event_phase': event['phase'] = 'evaluation'
    elif fault == 'event_failure': event['completed'] = False
    elif fault == 'event_tokens': event['tokens'] = 1
    elif fault == 'encode_frames': row['resource_events'][0]['frames'] = 2
    elif fault == 'encode_hook': row['image_encode_frames'] = 3
    elif fault == 'action_hook': row['action_encoder_work']['calls'] = 160
    elif fault == 'physics': row['physics_steps'] = 9
    elif fault == 'fake_total': row['predict_planning_samples'] = 80
    else: event['wall_seconds'] = -1
    with pytest.raises(ValueError): p.accounting(row)


def test_census_candidate_argmin_and_failure_retention(round_fixture):
    decision = round_fixture['decision']
    events = [p.strict_json(line) for line in (FOLDER/'root_0.actual.candidates.jsonl').read_bytes().splitlines()]
    events = [e for e in events if e['round'] == 1]
    ranked = p.audit_candidates(events,decision)
    assert ranked['winner_path'] == decision['selected']['path']
    assert ranked['margin_to_second'] >= 0
    for kind in ('drop','swap','failed','wrong_winner','nan'):
        values, d = deepcopy(events), deepcopy(decision)
        if kind == 'drop': values.pop()
        elif kind == 'swap': values[0],values[1] = values[1],values[0]
        elif kind == 'failed': values[0]['completed'] = False
        elif kind == 'wrong_winner': d['selected']['path'] = [0,0]
        else: values[0]['cost'] = float('nan')
        with pytest.raises(ValueError): p.audit_candidates(values,d)


def test_actual_independent_audit_complete_and_scientifically_blocked(audit_result):
    assert audit_result['audit_passed'], audit_result['errors']
    assert len(audit_result['rows']) == 8 and len(audit_result['paired_root_rounds']) == 8
    assert len(audit_result['source_verification']) == 3
    totals = audit_result['totals']
    assert totals['planning_predict_samples'] == totals['planning_predict_calls'] == 1280
    assert totals['evaluation_predict_samples'] == totals['evaluation_predict_calls'] == 64
    assert totals['processed_context_tokens'] == 4032
    assert totals['rgb_encode_calls'] == 16 and totals['rgb_encode_frames'] == 48
    assert totals['physical_step_attempts'] == totals['physical_step_returns'] == 80
    assert totals['action_encoder_samples'] == totals['action_encoder_calls'] == 1344
    assert audit_result['new_predict_calls'] == audit_result['new_image_encode_calls'] == audit_result['new_physics_steps'] == 0
    assert audit_result['P1_admission'] is audit_result['P4_admission'] is False
    assert audit_result['scientific_status'] == 'blocked_upstream'


def test_different_context_same_action_does_not_imply_null_effect(audit_result):
    pairs = [r for r in audit_result['paired_root_rounds'] if r['round'] == 1]
    assert all(r['planning_context_max_abs'] > 0 for r in pairs)
    assert sum(r['selected_paths_equal'] for r in pairs) == 3
    assert all(r['same_policy_implies_feedback_ineffective'] is False for r in pairs)
    outcome = audit_result['saved_outcome_pairs']
    assert [r['pose_cost_actual_minus_imagined'] == 0 for r in outcome] == [True,True,True,False]
    assert outcome[3]['pose_cost_actual_minus_imagined'] > 0
    assert not audit_result['feedback_ineffective'] and not audit_result['feedback_harm_generalized']


@pytest.mark.parametrize('passed',[True,False])
def test_admission_cannot_be_opened_by_engineering_pass(passed):
    result = p.boundary(passed)
    assert result['P4_admission'] is result['P1_admission'] is result['efficacy_verified'] is False
    assert len(result['missing_requirements']) == 5
    assert result['next_executable_gap']['requires_new_gpu'] is False
    assert result['next_executable_gap']['unlocks_P4'] is False


def test_read_hashes_detect_changes_and_pickle_arrays_blocked(tmp_path):
    path = tmp_path/'input.json'
    p.write(path,{'x':1})
    tracker = p.Inputs(); assert tracker.json(path) == {'x':1}
    with path.open('a') as stream: stream.write(' ')
    with pytest.raises(ValueError,match='changed'): tracker.stable()
    with (tmp_path/'object.npy').open('xb') as stream: np.save(stream,np.array([{}],dtype=object))
    with pytest.raises(ValueError): p.Inputs().npy(tmp_path/'object.npy',(1,))


def test_exclusive_new_output_and_failure_report(tmp_path,monkeypatch,audit_result):
    monkeypatch.setattr(p,'audit',lambda path:deepcopy(audit_result))
    output = tmp_path/p.TAG
    result = p.run(report=FOLDER/'report.json',output=output)
    assert result['audit_passed'] and (output/'report.md').exists()
    assert (output/'source_seal/manifest.json').exists()
    with pytest.raises(FileExistsError): p.run(report=FOLDER/'report.json',output=output)
    def fail(path): raise ValueError('missing scientific or source input')
    monkeypatch.setattr(p,'audit',fail)
    failed = p.run(report=tmp_path/'missing.json',output=tmp_path/(p.TAG+'_failed'))
    assert failed['status'] == 'integrity_fail' and not failed['P4_admission']


def test_input_read_only_and_saved_hashes_match(audit_result):
    assert audit_result['all_read_inputs_unchanged']
    assert len(audit_result['input_sha256']) > 300
    assert audit_result['input_sha256'][str(FOLDER/'report.json')] == p.Inputs().hash(FOLDER/'report.json')
