"""RAM-only fixed library support and witness/serialization regression tests."""
from copy import deepcopy
import json
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p1_common_library_support_20261028_v1 as m


def native_metrics(state, goal):
    s, g = np.asarray(state), np.asarray(goal); d = s-g
    distance = float(np.linalg.norm(d)); angle = abs(float(d[4]))
    success = bool(np.linalg.norm(d[:4]) < 20 and min(angle, 2*np.pi-angle) < np.pi/9)
    return m.panel.base.task_metrics(s, g, [s[2], s[3], s[4]], [256., 256., 0.], success, distance, -distance)


def rows():
    zero = np.zeros((5, 2), '<f4'); uz = np.arange(10, dtype='<f4').reshape(5, 2)*np.float32(.01)+np.float32(.1)
    us = np.full((5, 2), .2, '<f4'); bank = (zero, uz, -uz, us, -us)
    result = []
    for i in range(40):
        task, rem = divmod(i, 10); role, word = divmod(rem, 5)
        goal = [10., 0., 0., 0., np.pi/2, 0., 0.]
        endpoint = [float(role+1+(role+1)*(0, 1, -1, 2, -2)[word]), 0., 0., 0., 0., 0., 0.]
        metrics = native_metrics(endpoint, goal)
        result.append(dict(index=i, root_slot=task//2, condition=('off', 'on')[task%2], role=('Z', 'S')[role],
            library_index=word, library_label=m.teacher.LIBRARY[word], task_id=str(task), plan_digest=str(i), job_digest=str(i),
            goal192=[0.]*192, goal_binding=dict(task=task), outline_pose=[256.,256.,0.], physical_goal=goal,
            physical_metrics=metrics, returned_reward=-metrics['native_distance'], returned_state=endpoint,
            predicted_Q=float(word+1), actual_Q=None if i == 12 else float(5-word), suffix_raw_bits=bank[word].tobytes().hex(),
            boundary_trace={'10': dict(state=[0.]*7), '15': dict(state=[float(role),0.,0.,0.,0.,0.,0.]), '20': dict(state=endpoint)}))
    return result


def test_complete40_fixed_eight_supports_and_deployment_cells():
    a = m.assemble(rows())
    assert len(a['rows']) == 40 and len(a['tasks']) == 4 and a['known_roots'] == 2
    assert a['actual_latent_available'] == 39 and a['rows'][12]['actual_Q'] is None
    for task in a['tasks']:
        assert [len(r['spaces']['joint_xy']['finite_responses']) for r in task['roles']] == [5, 5]
        assert task['support_comparison']['joint_xy']['H_Z'] == 2.
        assert task['support_comparison']['joint_xy']['H_S'] == 4.
        assert task['support_comparison']['joint_xy']['H_S_minus_H_Z'] == 2.
        assert [c['library_index'] for c in task['fixed_deployment_and_cross_terms']] == [1,3,3,1]
        assert [c['role'] for c in task['fixed_deployment_and_cross_terms']] == ['Z','S','Z','S']
        assert not any(c['chosen_by_physical_outcome'] for c in task['fixed_deployment_and_cross_terms'])


def test_original_direction_not_postprefix_residual_and_angle_separate():
    r = rows(); r[0]['boundary_trace']['15']['state'][0] = 100.
    support = m.finite_support(r[:5], [0.]*7, r[0]['physical_goal'], 'joint_xy')
    assert support['original_root_to_goal_unit_direction'] == [1.,0.,0.,0.]
    assert support['support']['value'] == 2.
    angle = m.finite_support(r[:5], [0.]*7, r[0]['physical_goal'], 'object_angle_wrapped')
    assert angle['original_root_to_goal_unit_direction'] == [1.] and angle['support']['value'] == 0.
    assert angle['support']['attaining_library_indices'] == [0,1,2,3,4]


def test_zero_direction_is_unknown_not_zero_support():
    r = rows(); s = m.finite_support(r[:5], [0.]*7, r[0]['physical_goal'], 'object_xy')
    assert s['original_root_to_goal_unit_direction'] is None
    assert [x['projection'] for x in s['finite_responses']] == [None]*5
    assert s['support']['value'] is None and s['support']['reason'] == 'zero_original_root_goal_direction'


@pytest.mark.parametrize('missing', list(range(5)))
def test_any_required_endpoint_missing_never_subset_max(missing):
    r = rows()[:5]; r[missing]['returned_state'] = None
    s = m.finite_support(r, [0.]*7, r[0]['physical_goal'], 'joint_xy')
    assert len(s['finite_responses']) == 5 and s['support']['value'] is None
    assert s['support']['attaining_library_indices'] is None
    assert s['support']['available'] == (0 if missing == 0 else 4)


def test_angle_cut_wraps_response_not_linear_raw_subtraction():
    r = rows()[:5]; root = [0.,0.,0.,0.,3.,0.,0.]; goal = [0.,0.,0.,0.,-3.,0.,0.]
    r[0]['returned_state'][4] = 3.; r[1]['returned_state'][4] = -3.
    s = m.finite_support(r, root, goal, 'object_angle_wrapped')
    assert s['original_root_to_goal_unit_direction'] == [1.]
    assert s['finite_responses'][1]['projection'] == float(np.arctan2(np.sin(-6.),np.cos(-6.)))


def test_exact_all_ties_and_small_nonzero_preserved():
    assert m.complete_envelope([0.]*5)['attaining_library_indices'] == list(range(5))
    e = m.complete_envelope([0.,1e-300,0.,1e-300,0.])
    assert e['value'] == 1e-300 and e['attaining_library_indices'] == [1,3]
    assert m.complete_envelope([0.,None,2.,3.,4.])['value'] is None


@pytest.mark.parametrize('change', ['zero_sign','inverse','reverse_time','cross_role','clip','nan','wrong_size'])
def test_raw_action_authority_mutations_rejected(change):
    r = rows()[:10]; a = np.frombuffer(bytes.fromhex(r[1]['suffix_raw_bits']), '<f4').reshape(5,2).copy()
    if change == 'zero_sign': r[0]['suffix_raw_bits'] = (-np.zeros((5,2),'<f4')).tobytes().hex()
    elif change == 'inverse': r[2]['suffix_raw_bits'] = a.tobytes().hex()
    elif change == 'reverse_time': r[2]['suffix_raw_bits'] = (-a[::-1]).tobytes().hex()
    elif change == 'cross_role': r[6]['suffix_raw_bits'] = (a+np.float32(.01)).tobytes().hex()
    elif change == 'clip': a[0,0] = 1.1; r[1]['suffix_raw_bits'] = a.tobytes().hex()
    elif change == 'nan': a[0,0] = np.nan; r[1]['suffix_raw_bits'] = a.tobytes().hex()
    else: r[1]['suffix_raw_bits'] = '00'
    with pytest.raises(ValueError): m.raw_library(r)


@pytest.mark.parametrize('change', ['drop','reorder','word','role','root','goal','task','outline','start','root_after','fill12','other_missing','endpoint'])
def test_fixed_roster_and_native_task_join_fail_closed(change):
    r = rows()
    if change == 'drop': r.pop()
    elif change == 'reorder': r[0],r[1] = r[1],r[0]
    elif change == 'word': r[1]['library_label'] = 'other'
    elif change == 'role': r[1]['role'] = 'S'
    elif change == 'root': r[1]['root_slot'] = 1
    elif change == 'goal': r[1]['physical_goal'][0] = 1.
    elif change == 'task': r[1]['task_id'] = 'other'
    elif change == 'outline': r[1]['outline_pose'][0] = 1.
    elif change == 'start': r[1]['boundary_trace']['15']['state'][0] = 1.
    elif change == 'root_after': r[1]['boundary_trace']['10']['state'][0] = 1.
    elif change == 'fill12': r[12]['actual_Q'] = 0.
    elif change == 'other_missing': r[0]['actual_Q'] = None
    else: r[1]['returned_state'][0] = 1.  # boundary state shares object: detach for explicit mutation
    if change == 'endpoint': r[1]['boundary_trace']['20']['state'] = [99.]*7
    with pytest.raises(ValueError): m.assemble(r)


def test_raw_library_duplicates_preserved_no_dedup_or_refund():
    r = rows()[:10]
    for off in (0,5):
        r[off+3]['suffix_raw_bits'] = r[off+1]['suffix_raw_bits']
        r[off+4]['suffix_raw_bits'] = r[off+2]['suffix_raw_bits']
    bank = m.raw_library(r)
    assert len(bank) == 5 and bank[1]['raw_bits'] == bank[3]['raw_bits']


def test_own_zero_gain_success_and_Q_kept_separate_missing12_not_missing_physics():
    a = m.assemble(rows()); row = a['rows'][12]
    assert row['own_zero_comparison']['actual_latent_Q_gain'] is None
    assert row['own_zero_comparison']['physical_metrics_higher_is_better']['all_xy_error'] is not None
    assert a['tasks'][1]['roles'][0]['spaces']['joint_xy']['support']['value'] == 2.
    row = a['rows'][1]; g = row['own_zero_comparison']
    assert g['physical_metrics_higher_is_better']['all_xy_error'] == 1.
    assert g['predicted_Q_gain'] == -1. and g['actual_latent_Q_gain'] == 1.
    assert not g['model_Q_is_physical_gain']


def test_missing_metrics_preserve_fixed_native_envelope_null():
    r = rows(); r[1]['physical_metrics']['object_xy_error'] = None
    a = m.assemble(r)
    e = a['tasks'][0]['roles'][0]['native_metric_zero_relative_evaluation_upper_envelopes']['object_xy_error']
    assert e['value'] is None and e['available'] == 4 and e['fixed_slots'] == 5
    assert a['tasks'][0]['roles'][0]['spaces']['joint_xy']['support']['value'] == 2.


def test_finite_values_required():
    with pytest.raises(ValueError): m.complete_envelope([0.,0.,float('nan'),0.,0.])
    with pytest.raises(ValueError): m.complete_envelope([0.]*4)


def test_full_json_roundtrip_verify_and_rehashed_tampering(tmp_path):
    a = dict(schema=m.NAME+'.analysis', **m.assemble(rows()))
    p = tmp_path/'report.json'; p.write_bytes(m.canonical(a))
    assert json.loads(p.read_text()) == a and m.verify(p, m.sha(p), a)['verified']
    damaged = deepcopy(a); damaged['tasks'][0]['support_comparison']['joint_xy']['H_Z'] = 99.
    p.write_bytes(m.canonical(damaged))
    with pytest.raises(ValueError): m.verify(p, m.sha(p), a)
    with pytest.raises(ValueError): m.verify(p, '0'*64, a)


def test_exclusive_output_no_overwrite_or_MAIN_write(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'ROOT', tmp_path); (tmp_path/'results').mkdir()
    p = tmp_path/'results'/(m.NAME+'_test.json')
    m.write(p, dict(ok=True))
    with pytest.raises(FileExistsError): m.write(p, {})
    with pytest.raises(ValueError): m.write(tmp_path/'results/MAIN.json', {})
