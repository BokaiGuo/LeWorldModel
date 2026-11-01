"""Independent arithmetic checks on real saved P3 evidence; no runtime imports.

Run from le-wm-main. These tests consume the pinned calibration artifacts;
they are not mock efficacy experiments and never invoke a model or simulator.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
PREFIX = RESULTS / 'p3_joint_prefix_cuda_encoding_20261027_v1_cuda_main01'
PREPARED = RESULTS / 'p3_joint_prefix_cuda_encoding_20261027_v1_prepared01'
NUMERIC = RESULTS / 'p3_joint_prefix_cuda_encoding_20261027_v1_numerical_main01/report.json'
RECEIPT_SHA = 'ca6cdef177b2b8fe5e9739f2a07d2254b1183730ff9c340f52d96707f322a802'
NUMERIC_SHA = '147c3cca6a66789fbd5052fc4f4017b4e0a6b63c59038cfac1fbc19c01da77de'
PHYSICAL_SHA = '6c3659e199a8151fee2fa2c020a0cfe63db00b691b947cdbeb762c3c18d6004a'
PLANNER_SHA = '23eae72d18fa02e9997859311bef282fc4f5e3478790d846524a3fa6f6b70950'
METRICS = ('old_Q', 'goal_patch_Q')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path, expected=None):
    path = Path(path)
    assert path.stat().st_size < 24_000_000
    if expected is not None:
        assert sha(path) == expected
    return json.loads(path.read_text())


def indexed(rows, fields):
    out = {tuple(r[k] for k in fields): r for r in rows}
    assert len(out) == len(rows), 'duplicate identity'
    return out


def scores(actual, predicted, goal, q):
    a, p, g = (np.asarray(x, dtype=np.float64) for x in (actual, predicted, goal))
    q = np.asarray(q, dtype=np.float64)
    assert a.shape == p.shape == g.shape == (192,) and q.shape == (192, 192)
    assert all(np.isfinite(x).all() for x in (a, p, g, q))
    return dict(actual=float(np.mean((q @ (a-g))**2)),
                predicted=float(np.mean((q @ (p-g))**2)),
                q_mse=float(np.mean((q @ (a-p))**2)),
                native_mse=float(np.mean((a-p)**2)))


def pose(state, goal):
    s, g = np.asarray(state), np.asarray(goal)
    angle = abs((s[4]-g[4]+np.pi) % (2*np.pi)-np.pi)
    return float(np.linalg.norm(s[2:4]-g[2:4]) + 20*angle)


def assert_scores(value, claimed, metric):
    assert value['actual'] == pytest.approx(claimed['actual_Q_costs'][metric], abs=1e-12)
    assert value['predicted'] == pytest.approx(claimed['predicted_Q_costs'][metric], abs=1e-12)
    assert value['q_mse'] == pytest.approx(claimed['projected_prediction_mse'][metric], abs=1e-12)
    assert value['native_mse'] == pytest.approx(claimed['native_prediction_mse'], abs=1e-12)
    assert value['actual']-value['predicted'] == pytest.approx(
        claimed['actual_minus_predicted_same_Q_cost'][metric], abs=1e-12)


@lru_cache(maxsize=1)
def evidence():
    prepared = read(PREPARED/'prepare.json', 'ec969feae661973b845eb38677dd4d0a90d22cd990bda65d54569966c481504e')
    protocol = read(PREPARED/'protocol.json', prepared['protocol_sha256'])
    receipt = read(PREFIX/'receipt.json', RECEIPT_SHA)
    numeric = read(NUMERIC, NUMERIC_SHA)
    source = read(protocol['goal_q_source'], protocol['goal_q_source_sha256'])
    physical = read(protocol['physical_report'], PHYSICAL_SHA)
    planner = read(protocol['planner_report'], PLANNER_SHA)
    cp = read(RESULTS/'p3_joint_horizon_goal_metric_scheduler_20261027_v1_prepared01/protocol.json',
              'dcad803f75c1d0824358960995064793330ea769e3352b7b5656e1c9cfa9479e')
    plans = indexed(source['plans'], ('root_sha256', 'method'))
    rows = indexed(physical['rows'], ('root_sha256', 'method'))
    terminal = indexed(physical['encodings'], ('root_sha256', 'method'))
    roots = indexed(planner['roots'], ('root_sha256',))
    records = {r['packet']['root_sha256']: r for r in cp['records']}
    timelines = indexed(numeric['trajectories'], ('root_sha256', 'arm'))
    claims = indexed(numeric['rows'], ('frame_id',))
    frames = indexed(protocol['frames'], ('root_sha256', 'arm', 'primitive_step'))
    latent_rows = indexed(receipt['rows'], ('frame_id',))
    assert len(plans) == len(rows) == len(terminal) == len(timelines) == 16
    assert len(frames) == len(latent_rows) == len(claims) == 17
    points = {}
    for key, plan in plans.items():
        arm = roots[key[0],]['arms'][key[1]]
        record = records[key[0]]
        h = plan['planned_horizon_tokens']
        assert h in (2, 3) and len(arm['segments']) == h
        assert arm['winner']['actions'] == plan['commands']
        assert arm['winner']['context'] == arm['whole_replay_context'] == arm['segmented_replay_context'] == plan['selected_predicted_context']
        assert arm['whole_and_segmented_exact'] is True
        before = plan['predictor_context']
        assert before == record['packet']['root_context']
        generated = []
        mean = np.asarray(cp['normalization']['mean'], np.float32)
        std = np.asarray(cp['normalization']['std'], np.float32)
        for i, after in enumerate(arm['segments']):
            raw = np.asarray(plan['commands'][5*i:5*i+5], np.float32)
            token = ((raw-mean)/std).reshape(1, 10)
            expected = np.concatenate((np.asarray(before['action_history'], np.float32), token))[-2:]
            np.testing.assert_array_equal(after['action_history'], expected)
            np.testing.assert_array_equal(after['latent_history'][:2], np.asarray(before['latent_history'], np.float32)[1:])
            generated.append(after['latent_history'][-1])
            before = after
        assert before == plan['selected_predicted_context']
        assert before['latent_history'] == (plan['predictor_context']['latent_history'] + generated)[-3:]
        metrics = source['root_metrics'][key[0]]
        assert metrics == dict(goal=record['packet']['goal'], metrics=record['metrics'])
        checkpoints = timelines[key]['checkpoints']
        assert [p['primitive_step'] for p in checkpoints] == list(range(5, 5*h+1, 5))
        for packet_index in range(h):
            step = 5*(packet_index+1)
            if packet_index < h-1:
                f = frames[key[0], key[1], step]
                r = latent_rows[f['frame_id'],]
                assert f['predicted_vector'] == generated[packet_index]
                assert f['predicted_history_index'] == 3-h+packet_index
                assert f['predicted_vector'] == before['latent_history'][3-h+packet_index]
                assert sha(r['latent_path']) == r['latent_sha256']
                actual = np.load(r['latent_path'], allow_pickle=False)
                assert actual.dtype == np.float32 and actual.shape == (192,)
                assert sha(f['rgb_path']) == f['rgb_file_sha256']
                state = rows[key]['trace'][step-1]['state']
                assert state == f['actual_state']
                claim = claims[f['frame_id'],]
            else:
                actual = terminal[key]['terminal_vector']
                assert terminal[key]['selection_sha256'] == plan['selection_sha256']
                state = rows[key]['final_state']
                claim = checkpoints[-1]
            for metric in METRICS:
                result = scores(actual, generated[packet_index], metrics['goal'], metrics['metrics'][metric]['projection'])
                assert_scores(result, claim, metric)
                assert_scores(result, checkpoints[packet_index], metric)
                if metric == plan['metric_label']:
                    points[plan['root_index'], metric, plan['schedule'], step] = dict(
                        **result, pose=pose(state, rows[key]['context_identity']['goal']))
        assert timelines[key]['first_saved_nonzero_native_error_step'] == 5
    return protocol, receipt, numeric, points


def delta(points, root, metric, step, field, uniform_step=None):
    return (points[root, metric, 'dynamic', step][field] -
            points[root, metric, 'uniform', step if uniform_step is None else uniform_step][field])


def test_full_saved_census_chronology_and_all_33_two_q_arithmetic():
    p, r, n, points = evidence()
    assert len(points) == 33
    assert len(p['frames']) == 17
    assert sum(f['primitive_step'] == 5 for f in p['frames']) == 16
    assert [(f['root_index'], f['arm']) for f in p['frames'] if f['primitive_step'] == 10] == [(1, 'goal_patch_Q__uniform')]
    assert r['counts'] == {k: dict(attempted=17, completed=17) for k in ('encoder', 'existing_projector')}
    assert len(r['entries']) == 34 and all(e['completed'] for e in r['entries'])
    assert r['errors'] == [] and r['model_unchanged'] is r['source_verified'] is True
    assert n['new_neural_calls'] == n['new_physical_steps'] == 0
    assert n['inherited_terminal_encoder_samples'] == 16
    assert n['P1_admission'] is n['P3_admission'] is False


@pytest.mark.parametrize('root,step,predicted,actual,physical', [
    (1, 5, .2886732285007746, .18011909514904434, 7.520712994259455),
    (1, 10, -.017206227752833758, .20974681451552146, 71.85710171475712),
    (2, 5, -.17900066163275197, -.10651306878952882, 4.509433390926489),
    (2, 10, -.10960833340378606, -.023677502867109013, 11.635172243906013),
])
def test_harmed_old_q_roots_saved_time_decomposition(root, step, predicted, actual, physical):
    points = evidence()[3]
    for field, expected in [('predicted', predicted), ('actual', actual), ('pose', physical)]:
        assert delta(points, root, 'old_Q', step, field) == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize('root,metric', [(0, 'old_Q'), (0, 'goal_patch_Q'), (2, 'goal_patch_Q'), (3, 'old_Q'), (3, 'goal_patch_Q')])
def test_other_five_same_h_pairs_are_exact_ties(root, metric):
    points = evidence()[3]
    for step in (5, 10):
        assert all(delta(points, root, metric, step, f) == 0 for f in ('actual', 'predicted', 'pose'))


def test_mixed_h_is_separate_not_a_matched_terminal_comparison():
    points = evidence()[3]
    assert (1, 'goal_patch_Q', 'uniform', 15) in points
    assert (1, 'goal_patch_Q', 'dynamic', 15) not in points
    assert delta(points, 1, 'goal_patch_Q', 5, 'actual') == pytest.approx(.0256083060889125)
    assert delta(points, 1, 'goal_patch_Q', 10, 'actual', uniform_step=15) == pytest.approx(.0010634518711231528)


def test_counterexample_wrong_time_and_wrong_metric_fail_numeric_claim():
    p, _, n, _ = evidence()
    f = next(f for f in p['frames'] if f['root_index'] == 1 and f['arm'] == 'old_Q__dynamic')
    claim = next(r for r in n['rows'] if r['frame_id'] == f['frame_id'])
    for incorrect in (claim['predicted_Q_costs']['old_Q']+.01, claim['predicted_Q_costs']['goal_patch_Q']):
        wrong = dict(actual=claim['actual_Q_costs']['old_Q'], predicted=incorrect,
                     q_mse=claim['projected_prediction_mse']['old_Q'], native_mse=claim['native_prediction_mse'])
        with pytest.raises(AssertionError):
            assert_scores(wrong, claim, 'old_Q')


def test_duplicate_root_arm_records_are_not_silently_overwritten():
    with pytest.raises(AssertionError, match='duplicate identity'):
        indexed([dict(root='a', arm='b'), dict(root='a', arm='b')], ('root', 'arm'))


TF = RESULTS / 'p3_prefix_teacher_forcing_20261027_v1_cuda_main01'
TF_PREP = RESULTS / 'p3_prefix_teacher_forcing_20261027_v1_prepared01'
TF_SHA = 'c5bd1f15a4a8aa7dd60050f8ce5a681aa20c1966e1715cdc3e1df013e9fba499'


@lru_cache(maxsize=1)
def teacher_evidence():
    protocol, receipt, _, _ = evidence()
    report = read(TF/'report.json', TF_SHA)
    prepared = read(TF_PREP/'prepare.json', report['prepare_sha256'])
    tasks = read(TF_PREP/'protocol.json', prepared['protocol_sha256'])
    assert report['status'] == 'teacher_forcing_completed' and report['errors'] == []
    assert report['model_unchanged'] is report['source_verified'] is True
    assert report['model_hash_before'] == report['model_hash_after'] != None
    assert report['new_predict_calls'] == 17
    assert report['new_encoder_calls'] == report['new_AD_calls'] == report['new_physical_steps'] == 0
    assert report['P1_admission'] is report['P3_admission'] is report['training'] is False
    assert len(report['entries']) == 34
    assert [(e['task_id'], e['kind'], e['completed']) for e in report['entries']] == [
        (f'{i:02d}', k, True) for i in range(17) for k in ('action_encoder', 'predict')]
    assert sha(TF/'events.jsonl') == report['journal_sha256']
    assert len(tasks['tasks']) == len(report['rows']) == 17
    events = [json.loads(line) for line in (TF/'events.jsonl').read_text().splitlines()]
    expected_events = []
    for i in range(17):
        expected_events.append(dict(event='task_intent', task_id=f'{i:02d}'))
        for kind in ('action_encoder', 'predict'):
            expected_events.extend([
                dict(event='attempt', task_id=f'{i:02d}', kind=kind, completed=False),
                dict(event='return', task_id=f'{i:02d}', kind=kind, completed=True)])
    assert events == expected_events
    for path, expected in tasks['sources_sha256'].items():
        assert sha(path) == expected
    source = read(protocol['goal_q_source'], protocol['goal_q_source_sha256'])
    planner = read(protocol['planner_report'], PLANNER_SHA)
    physical = read(protocol['physical_report'], PHYSICAL_SHA)
    plans = indexed(source['plans'], ('root_sha256', 'method'))
    roots = indexed(planner['roots'], ('root_sha256',))
    terminal = indexed(physical['encodings'], ('root_sha256', 'method'))
    by_id = indexed(receipt['rows'], ('frame_id',))
    observed = {}
    for f in protocol['frames']:
        observed[f['root_sha256'], f['arm'], f['primitive_step']] = np.load(
            by_id[f['frame_id'],]['latent_path'], allow_pickle=False).tolist()
    expected_census = {(key[0], key[1], b*5) for key, plan in plans.items()
                       for b in range(1, plan['planned_horizon_tokens'])}
    assert {(t['root_sha256'], t['arm'], t['boundary_step']) for t in tasks['tasks']} == expected_census
    return tasks, report, source, roots, plans, observed, terminal


@pytest.mark.parametrize('i', range(17))
def test_actual_teacher_forced_task_causal_input_binding_and_both_q_arithmetic(i):
    tasks, report, source, roots, plans, observed, terminal = teacher_evidence()
    t, row = tasks['tasks'][i], report['rows'][i]
    assert t['task_id'] == row['task_id'] == f'{i:02d}'
    assert all(t[k] == row[k] for k in ('root_sha256', 'arm', 'boundary_step', 'target_step'))
    key = t['root_sha256'], t['arm']
    plan = plans[key]
    arm = roots[key[0],]['arms'][key[1]]
    b = t['boundary_step']//5
    assert t['target_step'] == 5*(b+1)
    expected_context = dict(arm['segments'][b-1])
    expected_context['latent_history'] = (plan['predictor_context']['latent_history'] + [
        observed[key[0], key[1], 5*k] for k in range(1, b+1)])[-3:]
    assert t['context'] == expected_context
    assert t['commands'] == plan['commands'][5*b:5*(b+1)]
    target = (terminal[key]['terminal_vector'] if b+1 == plan['planned_horizon_tokens']
              else observed[key[0], key[1], 5*(b+1)])
    assert t['actual_target'] == target
    assert t['imagined_target'] == arm['segments'][b]['latent_history'][-1]
    saved = read(TF/(t['task_id']+'.context.json'))
    assert saved['latent_history'][:2] == t['context']['latent_history'][1:]
    assert saved['action_history'] == arm['segments'][b]['action_history']
    assert row == read(TF/(t['task_id']+'.result.json'))
    metric = source['root_metrics'][key[0]]
    for q in METRICS:
        for label, prediction in [('teacher_forced', saved['latent_history'][-1]), ('imagined', t['imagined_target'])]:
            result = scores(target, prediction, metric['goal'], metric['metrics'][q]['projection'])
            assert_scores(result, row[label], q)
    own_q = plan['metric_label']
    assert row['teacher_forced']['native_prediction_mse'] < row['imagined']['native_prediction_mse']
    assert row['teacher_forced']['projected_prediction_mse'][own_q] < row['imagined']['projected_prediction_mse'][own_q]
