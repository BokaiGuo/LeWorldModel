"""RAM/saved arithmetic only; tests are not native task or CUDA evidence."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p3_task_resource_decomposition_20261028_v1 as m


@pytest.fixture(scope='module')
def actual():
    path = m.ROOT/'results/agent_p3_budget_paired_analysis_20261028_v1_saved01.json'
    assert m.sha(path) == m.PAIRED_SHA
    a = m.bridge.READ(path, m.paired.CAP)
    assert m.sha(a['prepared']) == a['prepare_sha256']
    p = m.bridge.READ(a['prepared'], 64*1024**2)
    authority = p['authority']
    assert m.sha(authority['host_report']) == authority['report_sha256']
    assert m.sha(authority['strict_receipt']) == authority['receipt_sha256']
    report = m.bridge.READ(authority['host_report'], 256*1024**2)
    receipt = m.bridge.READ(authority['strict_receipt'], 256*1024**2)
    hp = m.bridge.READ(report['prepared'], 64*1024**2)
    # Arithmetic ABI regression, NOT a replacement for the actual CLI full closure.
    return a, p, hp, report, receipt


@pytest.fixture(scope='module')
def result(actual): return m.assemble(*actual)


def test_native_squared_axes_not_wrapped_angle():
    goal = [0]*7; state = [3,4,6,8,2*np.pi,5,12]
    x = m.decompose(dict(state=state, goal=goal))
    assert x['agent_xy_squared'] == 25 and x['object_xy_squared'] == 100
    assert x['joint_xy_squared'] == 125 and x['velocity_squared'] == 169
    assert x['raw_angle_squared'] == (2*np.pi)**2
    assert m.roundoff(x['native7_squared'], 125+(2*np.pi)**2+169)
    # native angle success term/wrapped angle is NOT the fifth native7 component.
    assert x['raw_angle_squared'] > 39


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), True])
def test_bad_native_numbers_rejected(bad):
    with pytest.raises(ValueError): m.decompose(dict(state=[0,0,0,0,0,0,bad], goal=[0]*7))


def test_missing_native_dimension_rejected():
    with pytest.raises(ValueError): m.decompose(dict(state=[0]*6, goal=[0]*7))


def test_nonzero_goal_velocity_is_not_zero_target():
    x = m.decompose(dict(state=[0,0,0,0,0,9,11], goal=[0,0,0,0,0,6,7]))
    assert x['velocity_squared'] == 25
    assert x['residual7'][5:] == [3,4]


def test_algebraic_velocity_offset_example_not_latent_prediction():
    f = m.decompose(dict(state=[3,0,0,0,0,10,0], goal=[0]*7))
    a = m.decompose(dict(state=[5,0,0,0,0,1,0], goal=[0]*7))
    d = {k:a[k]-f[k] for k in m.SQUARED}
    assert d['native7_squared'] == -83 and d['joint_xy_squared'] == 16 and d['velocity_squared'] == -99
    assert m.roundoff(d['native7_squared'], sum(d[k] for k in m.COMPONENTS))


def test_actual_all48_all24_two_root_clusters(result):
    assert len(result['rows']) == 48 and len(result['pairs']) == 24
    assert result['all_pairs']['known_pairs'] == 18 and result['all_pairs']['missing_pairs'] == 6
    assert [c['root_slot'] for c in result['root_clusters']] == [0,1]
    assert [c['summary']['known_pairs'] for c in result['root_clusters']] == [6,12]
    assert sum(c['independent_root_count'] for c in result['root_clusters']) == 2
    assert result['independent_known_roots'] == 2
    assert all(result[k] is False for k in m.FLAGS)


def test_actual_native_recompute_every_measured_pair(result):
    for r in result['rows']:
        if r['final'] is not None:
            for stage in ('initial','final'):
                x = r[stage]
                assert m.roundoff(x['native7_squared'], sum(x[k] for k in m.COMPONENTS))
                assert m.roundoff(x['native7_squared'], x['native7_distance']**2)
                assert m.roundoff(x['joint_xy_squared'], x['joint_xy_error']**2)
    for p in result['pairs']:
        if p['squared_delta_adaptive_minus_fixed'] is not None:
            x = p['squared_delta_adaptive_minus_fixed']
            assert m.roundoff(x['native7_squared'], sum(x[k] for k in m.COMPONENTS))
    x = result['all_pairs']['squared_delta_adaptive_minus_fixed']
    assert m.roundoff(x['native7_squared']['known_sum'], sum(x[k]['known_sum'] for k in m.COMPONENTS))


def test_actual_distance_axis_contradiction_kept(result):
    dist = result['all_pairs']['distance_delta_adaptive_minus_fixed']
    assert dist['native7_distance']['known_only_mean'] < 0
    assert dist['joint_xy_error']['known_only_mean'] > 0
    sq = result['all_pairs']['squared_delta_adaptive_minus_fixed']
    assert sq['native7_squared']['known_only_mean'] < 0
    assert sq['joint_xy_squared']['known_only_mean'] > 0
    assert sq['velocity_squared']['known_only_mean'] < 0


def test_missing_and_earlystop_preserved(result):
    missing = [r for r in result['rows'] if r['final'] is None]
    assert len(missing) == 7
    assert all(r['initial'] is r['initial_minus_final_squared'] is None for r in missing)
    assert sum(r['route'] == 'paid_refused' for r in missing) == 5
    assert sum(r['route'] == 'not_reached' for r in missing) == 2
    assert all(p['squared_delta_adaptive_minus_fixed'] is None for p in result['pairs']
        if p['distance_delta_adaptive_minus_fixed'] is None)
    assert result['all_pairs']['squared_delta_adaptive_minus_fixed']['native7_squared']['full_denominator_mean'] is None
    assert sum(r['native_earlystop'] is True for r in result['rows']) == 4
    assert sum(r['own_returned'] is not None and r['own_returned'] < 5 for r in result['rows']) == 2


def test_Q_is_latent_not_native_and_horizons_separate(result):
    selected = [r['model_Q'] for r in result['rows'] if r['model_Q']['selected_status'] == 'selected']
    assert len(selected) == 41
    assert {q['selected_horizon_packets'] for q in selected} == {1,3}
    assert sum(q['same_five_step_model_and_physical_horizon'] is False for q in selected) == 2
    assert all(q['native_goal_prediction'] is q['native_joint_xy_prediction'] is q['native_velocity_prediction'] is None
               for q in selected)
    assert all(q['Q_is_native_success'] is q['Q_is_native_reward'] is False for q in selected)
    assert all(p['Q_predicted_native_jointXY_delta'] is p['Q_predicted_native_velocity_delta'] is None for p in result['pairs'])
    assert all(q['prefix5_latent_Q_gain'] == q['parent_latent_Q']-q['prefix5_latent_Q'] for q in selected)
    assert result['all_pairs']['native_prediction_comparison']['velocity'] == 'unknown'


def test_resources_retained_actual_not_cap(result):
    assert result['actual_resources']['fixed']['completed']['predict_API'] == 391
    assert result['actual_resources']['adaptive']['completed']['predict_API'] == 634
    assert result['resource_delta_adaptive_minus_fixed']['completed']['predict_API'] == 243
    assert result['actual_vectors_equal'] is False
    assert result['actual_resource_coverage']['fixed'] == dict(measured_slots=23, missing_slots=1, slot_denominator=24)
    assert result['FLOPs'] is None
    assert result['new_neural_calls'] == result['new_physics_calls'] == 0


@pytest.mark.parametrize('tamper', ['drop','roots','science','pair','resource','state','goal','raw','cost','native_mapping'])
def test_saved_tamper_fails(actual,tamper):
    a,p,hp,report,receipt = actual
    a = deepcopy(a); p = dict(p, handoff=deepcopy(p['handoff'])); hp = dict(hp, wiring=dict(hp['wiring']))
    # Only copy a single compiler row for mutation, not all 90MiB arrays.
    receipt = dict(receipt, full_rows=list(receipt['full_rows']))
    receipt['full_rows'][0] = deepcopy(receipt['full_rows'][0])
    if tamper == 'drop': a['rows'].pop()
    elif tamper == 'roots': a['independent_known_roots'] = 24
    elif tamper == 'science': a['P3_admission'] = True
    elif tamper == 'pair': a['pairs'][0]['child_id'] = 'bad'
    elif tamper == 'resource': a['resource_delta_adaptive_minus_fixed']['completed']['predict_API'] -= 1
    elif tamper == 'state': p['handoff']['rows'][0]['reuse_evidence']['full_old_row']['last_observation']['state'][5] += 1
    elif tamper == 'goal': p['handoff']['rows'][0]['reuse_evidence']['full_old_row']['last_observation']['goal'][5] += 1
    elif tamper == 'raw': receipt['full_rows'][0]['compiler']['selected']['raw5'][0][0] += .001
    elif tamper == 'cost': receipt['full_rows'][0]['compiler']['selected']['terminal_cost'] += .01
    else:
        hp['wiring']['tasks'] = deepcopy(hp['wiring']['tasks'])
        hp['wiring']['tasks'][0]['native_prediction_mapping'] = 'unbound invented decoder'
    with pytest.raises((ValueError, KeyError, TypeError)): m.assemble(a,p,hp,report,receipt)


def test_wrong_external_SHA_stops_before_full_consumer(monkeypatch,tmp_path):
    p = tmp_path/'bad.json'; p.write_text('{}')
    def forbidden(*a, **kw): raise AssertionError('must fail before full source consumer')
    monkeypatch.setattr(m.paired, 'verify', forbidden)
    with pytest.raises(ValueError,match='external actual complete saved01 SHA'): m.material(str(p),'0'*64)
    with pytest.raises(ValueError,match='external decomposition SHA'): m.verify(str(p),'0'*64)


def test_freeze_pins_and_no_execution():
    m.bridge.contract.check_sources(m.PINS)
    source = Path(m.__file__).read_text()
    assert 'import torch' not in source and 'native_factory(' not in source and 'run_verified(' not in source
    assert 'host.consume(' not in source


def test_new_writer_size_and_reentry(tmp_path):
    p = tmp_path/'new.json'; m.bridge.physical.write_new(p,dict(RAM=True),cap=1024)
    with pytest.raises(ValueError): m.bridge.physical.write_new(p,dict(RAM=False),cap=1024)
    with pytest.raises(ValueError): m.bridge.physical.write_new(tmp_path/'large.json',dict(RAM='a'*30),cap=8)
