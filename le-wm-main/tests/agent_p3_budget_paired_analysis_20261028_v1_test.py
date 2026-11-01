"""Saved arithmetic/RAM boundary tests, never scientific or physics evidence."""
from copy import deepcopy
from pathlib import Path

import pytest

from experiments.spectral_reachability import agent_p3_budget_paired_analysis_20261028_v1 as m


@pytest.fixture(scope='module')
def actual():
    path = m.ROOT/'results/agent_p3_budget_physical_handoff_20261028_v1_prepare_MAIN01.json'
    assert m.sha(path) == m.HANDOFF_SHA
    p = m.bridge.READ(path, 64*1024**2)
    r = m.bridge.READ(p['authority']['host_report'], 256*1024**2)
    receipt = m.bridge.READ(p['authority']['strict_receipt'], 256*1024**2)
    assert m.sha(p['authority']['host_report']) == p['authority']['report_sha256']
    assert m.sha(p['authority']['strict_receipt']) == p['authority']['receipt_sha256']
    # Arithmetic fixture only: full upstream validation is exercised by actual CLI.
    return p, r, receipt


@pytest.fixture(scope='module')
def result(actual): return m.assemble(*actual)


def test_actual_full_denominator_and_unequal_compute(result):
    assert len(result['rows']) == 48 and len(result['pairs']) == 24
    assert result['independent_known_roots'] == 2
    assert result['modes']['fixed']['measured_task_slots'] == 23
    assert result['modes']['adaptive']['measured_task_slots'] == 18
    assert result['modes']['fixed']['resources']['completed']['predict_API'] == 391
    assert result['modes']['adaptive']['resources']['completed']['predict_API'] == 634
    assert result['resource_delta_adaptive_minus_fixed']['completed']['predict_API'] == 243
    assert result['actual_vectors_equal'] is False
    assert result['resource_savings']['uniform_actual_savings'] is False
    assert all(result[k] is False for k in m.FLAGS)


def test_missing_is_not_zero_and_pairs_not_selected(result):
    missing = [r for r in result['rows'] if r['final'] is None]
    assert len(missing) == 7
    assert sum(r['route'] == 'paid_refused' for r in missing) == 5
    assert sum(r['route'] == 'not_reached' for r in missing) == 2
    assert all(r['initial'] is r['improvement'] is r['own_returned'] is None for r in missing)
    assert all(r['resources'] is not None for r in missing if r['route'] == 'paid_refused')
    assert all(r['resources'] is None for r in missing if r['route'] == 'not_reached')
    assert sum(p['task_delta_adaptive_minus_fixed'] is not None for p in result['pairs']) == 18
    assert result['modes']['adaptive']['task']['native_success']['full_denominator_mean'] is None
    assert result['modes']['fixed']['task']['native_success']['full_denominator_mean'] is None


def test_two_root_clusters_not_24_independent_roots(result):
    assert [c['root_slot'] for c in result['root_clusters']] == [0, 1]
    assert sum(c['child_denominator'] for c in result['root_clusters']) == 24
    assert sum(c['independent_root_count'] for c in result['root_clusters']) == 2
    assert result['equal_root_weight_paired_delta']['joint_xy_error']['denominator'] == 2


def test_earlystop_and_historical_cost_not_new(result):
    rows = [r for r in result['rows'] if r['final'] is not None]
    assert sum(r['native_earlystop'] for r in rows) == 4
    # Two terminal returns are at step5, two genuinely stop before step5.
    assert sum(r['own_returned'] < 5 for r in rows) == 2
    assert all(r['own_returned'] <= 5 for r in rows)
    assert all(r['physical_reference']['fresh'] is False for r in rows)
    assert result['new_neural_calls'] == result['new_physics_calls'] == 0
    assert result['wall']['fixed_seconds'] is result['wall']['adaptive_seconds'] is None
    assert all(r['new_physics_resources'] is None for r in rows)


def test_native7_reward_is_not_native_joint_success():
    x = m.task_metrics(dict(state=[0,0,0,0,0,1000,1000], goal=[0]*7))
    assert x['native_success'] is True and x['native7_distance'] > 1000
    assert x['joint_xy_error'] == 0 and x['object_position_error'] == 0
    assert x['native7_reward'] == -x['native7_distance']
    y = m.task_metrics(dict(state=[25,0,0,0,0,0,0], goal=[0]*7))
    assert y['native_success'] is False and y['object_position_error'] == 0


@pytest.mark.parametrize('change', ['denominator', 'science', 'drop', 'order', 'root', 'context',
    'goal', 'source', 'command', 'selection', 'metric', 'earlystop', 'resources', 'missing', 'paid'])
def test_tamper_rejected(actual, change):
    # Copy only small affected handoff and ledger subtrees, not all 90MB arrays.
    original, report, full = actual
    p = dict(original, handoff=deepcopy(original['handoff']))
    receipt = dict(full)
    rows = p['handoff']['rows']; r = rows[0]
    if change == 'denominator': p['handoff']['child_denominator'] = 23
    elif change == 'science': p['P3_admission'] = True
    elif change == 'drop': rows.pop()
    elif change == 'order': rows[0], rows[1] = rows[1], rows[0]
    elif change == 'root': r['runner_plan']['child']['root_slot'] = 3
    elif change == 'context': r['runner_plan']['child']['context_sha256'] = '0'*64
    elif change == 'goal': r['goal_sha256'] = '0'*64
    elif change == 'source': r['runner_plan']['child']['ancestry']['source_report_sha256'] = '0'*64
    elif change == 'command': r['command']['bytes_hex'] = '00'
    elif change == 'selection': r['selection_digest'] = '0'*64
    elif change == 'metric': r['reuse_evidence']['full_old_row']['final_metrics']['native_success'] = 'false'
    elif change == 'earlystop': r['reuse_evidence']['full_old_row']['own_returned'] = 0
    elif change == 'resources':
        receipt['actual_resources'] = deepcopy(full['actual_resources'])
        receipt['actual_resources']['fixed']['completed']['predict_API'] -= 1
    elif change == 'missing':
        missing = next(r for r in rows if r['route'] == 'paid_refused')
        missing['route'] = 'reused_saved'
    else: r['paid_status'] = 'not_reached'
    with pytest.raises((ValueError, TypeError, KeyError)): m.assemble(p, report, receipt)


def test_known_missing_semantics():
    assert m.known([None, None]) == dict(denominator=2, known=0, missing=2,
        known_sum=None, known_only_mean=None, full_denominator_mean=None)
    assert m.known([0, None])['known_only_mean'] == 0
    assert m.known([0, None])['full_denominator_mean'] is None
    with pytest.raises(ValueError): m.known([float('nan')])


@pytest.mark.parametrize('x', [-1, True, 1.5])
def test_bad_resource_counter(x):
    with pytest.raises(ValueError): m.resource_total([dict(attempted={'a':x}, completed={'a':0})])


def test_wrong_SHA_fails_before_saved_consumer(monkeypatch, tmp_path):
    path = tmp_path/'wrong.json'; path.write_text('{}')
    def forbidden(*a, **kw): raise AssertionError('must not consume on wrong SHA')
    monkeypatch.setattr(m.bridge, 'verify', forbidden)
    with pytest.raises(ValueError, match='external handoff SHA'): m.material(str(path), '0'*64)


def test_verify_analysis_tamper_fails_before_reconstruction(monkeypatch, tmp_path):
    path = tmp_path/'bad.json'; path.write_text('{}')
    def forbidden(*a, **kw): raise AssertionError('must not reconstruct wrong SHA')
    monkeypatch.setattr(m, 'material', forbidden)
    with pytest.raises(ValueError, match='external actual analysis SHA'): m.verify(str(path), '0'*64)


def test_exclusive_writer_and_size_cap(tmp_path):
    path = tmp_path/'new.json'
    m.bridge.physical.write_new(path, {'RAM': True}, cap=1024)
    with pytest.raises(ValueError): m.bridge.physical.write_new(path, {'RAM': False}, cap=1024)
    with pytest.raises(ValueError): m.bridge.physical.write_new(tmp_path/'large.json', {'RAM': 'x'*20}, cap=8)


def test_paired_all24_and_reward_gain_direction(result):
    assert result['paired_task_delta']['joint_xy_error']['denominator'] == 24
    assert result['paired_task_delta']['joint_xy_error']['known'] == 18
    assert result['paired_task_delta']['joint_xy_error']['missing'] == 6
    assert result['paired_task_delta']['joint_xy_error']['full_denominator_mean'] is None
    for r in result['rows']:
        if r['final'] is not None:
            assert r['improvement']['native7_reward'] == r['final']['native7_reward']-r['initial']['native7_reward']


def test_freeze_pins_and_no_execution_entry():
    m.bridge.contract.check_sources(m.PINS)
    text = Path(m.__file__).read_text()
    assert 'import torch' not in text and 'native_factory(' not in text
    assert '.consume(' not in text and 'run_verified(' not in text
