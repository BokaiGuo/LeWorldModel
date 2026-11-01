"""Independent saved-array audit; no model/simulator imports or execution.

Run as pytest, or execute this file to print recomputed real artifact numbers.
Numerical tolerances are receipt tolerances, NEVER plateau thresholds.
"""
import collections
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SAVED = ROOT / 'results/p1_direct_cuda_calibration_main_20261027_v2_cuda_main01'


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tokens(task, raw):
    return (np.asarray(raw, np.float32).reshape(-1, 10) -
            np.tile(np.asarray(task['action_mean'], np.float32), 5)) / np.tile(np.asarray(task['action_std'], np.float32), 5)


def check_path(task, commands, contexts):
    assert len(contexts) == 2
    previous = task['root_context']
    for k, context in enumerate(contexts):
        assert context['layout'] == 'states_with_between_state_actions_v1'
        assert np.asarray(context['latent_history']).shape == (3, 192)
        assert np.isfinite(context['latent_history']).all()
        assert context['latent_history'][:2] == previous['latent_history'][1:]
        expected = np.concatenate((previous['action_history'], tokens(task, commands)[k:k+1]))[-2:]
        assert np.array_equal(context['action_history'], expected)
        previous = context


def prediction(task, value):
    assert value['task_id'] == digest(task)
    logits = np.asarray(value['logits'])
    assert logits.shape == (2, 10) and np.isfinite(logits).all()
    assert np.all(np.linalg.norm(logits, axis=1) <= task['packet_logit_cap'] + 1e-12)
    raw = np.tanh(logits.astype(np.float32)).reshape(10, 2)
    assert np.array_equal(raw, value['commands']) and np.max(np.abs(raw)) <= 1
    assert np.all(np.linalg.norm(raw.astype(float).reshape(2, 10), axis=1) <= task['packet_logit_cap'] + 1e-7)
    check_path(task, raw, value['contexts'])
    assert value['whole_replay'] == value['contexts'][-1]
    assert value['segmented_replay'] == value['contexts']
    residual = np.asarray(task['projection']) @ (np.asarray(value['contexts'][-1]['latent_history'][-1]) - task['goal'])
    assert value['cost'] == float(np.mean(residual * residual))


def project(value, cap):
    value = np.asarray(value, float).copy()
    for row in value:
        norm = np.linalg.norm(row)
        if norm > cap:
            row *= cap / norm
    value = value.astype(np.float32)
    for row in value:
        if np.linalg.norm(row.astype(float)) > cap:
            row *= np.float32(1 - 4 * np.finfo(np.float32).eps)
    return value.astype(float)


def audit_component(task, report):
    assert report['status'] == 'direct_component_completed_not_admitted'
    assert report['error'] is None and report['secondary_errors'] == []
    assert report['task_id'] == digest(task)
    c = report['configuration']
    assert digest(c) == report['configuration_sha256']
    assert (c['B1'], c['B2'], len(c['initial_logits']), c['extra_line_search_trials']) == (1, 3, 1, 0)
    expected_budget = dict(predict=30, action_encoder=30, jacobian=3, vjp_directions=576,
                           svd=3 if c['method']=='full_H2_GN' else 0,
                           linear_solve=3 if c['method']=='full_H2_GN' else 0, proposal=3, selection=3)
    assert report['budget'] == expected_budget
    assert np.array_equal(task['projection'], np.eye(192))
    assert np.array_equal(task['root_context']['action_history'], tokens(task, task['root_raw_action_history']))
    assert len(report['direct_runs']) == 1
    run = report['direct_runs'][0]
    assert len(run) == 4
    assert run[0]['incumbent']['logits'] == c['initial_logits'][0]
    gradients, supports, spectra = [], [], []
    for i, step in enumerate(run):
        assert step['iteration'] == i
        prediction(task, step['incumbent'])
        if i == 0:
            assert step['derivative'] is None and step['proposal'] is None and not step['accepted']
            continue
        previous = run[i-1]['incumbent']
        proposal = step['proposal']; prediction(task, proposal)
        assert step['accepted'] is (proposal['cost'] < previous['cost'])
        assert step['incumbent'] == (proposal if step['accepted'] else previous)
        derivative = step['derivative']
        assert derivative['task_id'] == digest(task) and derivative['at_logits'] == previous['logits']
        assert derivative['at_terminal'] == previous['contexts'][-1]['latent_history'][-1]
        j = np.asarray(derivative['raw_terminal_jacobian'])
        assert j.shape == (192, 20) and np.isfinite(j).all()
        q = np.asarray(task['projection']); a = q @ j
        residual = q @ (np.asarray(derivative['at_terminal']) - task['goal'])
        gradient = 2 * a.T @ residual / 192
        np.testing.assert_allclose(gradient, derivative['objective_gradient'], rtol=1e-12, atol=1e-14)
        if c['method'] == 'full_H2_GN':
            u, s, vh = np.linalg.svd(a / np.sqrt(192), full_matrices=False)
            delta = -vh.T @ (s * (u.T @ (residual / np.sqrt(192))) / (s*s+c['damping']))
            spectra.append([float(s[0]), float(s[-1])])
        else:
            delta = -gradient
        expected = project(np.asarray(previous['logits']) + c['step_scale']*delta.reshape(2, 10), task['packet_logit_cap'])
        np.testing.assert_allclose(expected, proposal['logits'], rtol=0, atol=2e-7)
        gradients.append(float(np.linalg.norm(gradient)))
        supports.append([float(np.linalg.norm(j[:, :10])), float(np.linalg.norm(j[:, 10:]))])
    events = report['extended_journal']
    counts = collections.Counter()
    for i, event in enumerate(events):
        assert event['index'] == i and event['completed'] and event['samples'] == 1
        counts[event['resource']] += event['samples']
    assert dict(counts) == {k:v for k,v in report['budget'].items() if v}
    costs = [s['incumbent']['cost'] for s in run]
    return dict(method=c['method'], costs=costs, accepted=[s['accepted'] for s in run[1:]],
                B1_to_B2_reduction=costs[1]-costs[3], gradient_norms=gradients,
                logit_J_packet_frobenius=supports, singular_extremes=spectra, resources=dict(counts))


def audit_journal(task, report, path):
    events = [json.loads(line) for line in Path(path).read_text().splitlines()]
    assert events[0]['kind'] == 'direct_started' and events[0]['task'] == task
    assert events[0]['configuration'] == report['configuration']
    assert events[-1] == dict(kind='direct_finished', report=report)
    pending = set()
    for event in events:
        if event['kind'] == 'operation_attempted':
            assert event['event']['index'] not in pending
            pending.add(event['event']['index'])
        elif event['kind'] == 'operation_completed':
            assert event['event']['index'] in pending
            pending.remove(event['event']['index'])
    assert not pending
    attempted = [e['event'] for e in events if e['kind'] == 'operation_attempted']
    completed = [e['event'] for e in events if e['kind'] == 'operation_completed']
    # AD/J wraps nested encoder/predict/VJP operations; completion order is
    # intentionally not dispatch order. Pair by immutable operation index.
    completed = sorted(completed, key=lambda e: e['index'])
    assert completed == report['extended_journal']
    assert len(attempted) == len(completed)
    for a, b in zip(attempted, completed):
        assert not a['completed'] and dict(a, completed=True) == b
    steps = [e['step'] for e in events if e['kind'] == 'trajectory_step']
    assert steps == report['direct_runs'][0]
    predictions = [e['prediction'] for e in events if e['kind'] == 'prediction_complete']
    assert predictions == [steps[0]['incumbent']] + [s['proposal'] for s in steps[1:]]
    paths = [e for e in events if e['kind'] == 'prediction_path']
    packets = [e for e in events if e['kind'] == 'native_packet']
    assert len(paths) == 12 and len(packets) == 24
    for i, p in enumerate(predictions):
        for n, mode in enumerate(('nominal', 'whole_replay', 'segmented_replay')):
            event = paths[3*i+n]
            assert event['mode'] == mode and event['contexts'] == p['contexts']
            for k in range(2):
                v = packets[6*i+2*n+k]
                assert v['packet'] == k and v['mode'] == mode and v['context'] == p['contexts'][k]
    derivatives = [s['derivative'] for s in steps[1:]]
    assert [e['derivative'] for e in events if e['kind'] == 'derivative_complete'] == derivatives
    assert [e['logits'] for e in events if e['kind'] == 'proposal_logits'] == [s['proposal']['logits'] for s in steps[1:]]
    rows = [e for e in events if e['kind'] == 'jacobian_row']
    joint = [e for e in events if e['kind'] == 'joint_logit_jacobian']
    forward = [e for e in events if e['kind'] == 'jacobian_forward_complete']
    forward_packets = [e for e in events if e['kind'] == 'jacobian_forward_packet']
    assert (len(rows), len(joint), len(forward), len(forward_packets)) == (576, 3, 3, 6)
    raw_support = []
    for i, derivative in enumerate(derivatives):
        incumbent = steps[i]['incumbent']
        assert forward[i]['terminal'] == derivative['at_terminal'] and forward[i]['logits'] == derivative['at_logits']
        for k in range(2):
            assert forward_packets[2*i+k]['packet'] == k and forward_packets[2*i+k]['context'] == incumbent['contexts'][k]
        block = rows[192*i:192*(i+1)]
        assert [e['row'] for e in block] == list(range(192))
        raw_j = np.asarray([e['raw_control_derivative'] for e in block])
        assert raw_j.shape == (192, 20) and np.isfinite(raw_j).all()
        expected = raw_j * (1-np.asarray(incumbent['commands']).reshape(20)**2)[None]
        assert np.array_equal(expected, derivative['raw_terminal_jacobian'])
        assert joint[i]['jacobian'] == derivative['raw_terminal_jacobian'] and joint[i]['terminal'] == derivative['at_terminal']
        norms = [float(np.linalg.norm(raw_j[:, :10])), float(np.linalg.norm(raw_j[:, 10:]))]
        assert min(norms) > 0  # support evidence, NOT a scientific sensitivity threshold
        raw_support.append(norms)
    return raw_support


def audit_saved():
    top = read(SAVED / 'report.json')
    assert top['status'] == 'direct_calibration_completed' and top['errors'] == []
    assert top['model_hash_before'] == top['model_hash_after'] and top['model_unchanged']
    assert not top['P1_admission'] and not top['model_task_eligible'] and not top['training']
    result = dict(components=[], resources=collections.Counter())
    source_dir = ROOT / 'results/p1_conditional_calibration2_main_20261027_v1'
    source = read(source_dir / 'sourcepack.json')
    images = [json.loads(s) for s in (SAVED / 'image.events.jsonl').read_text().splitlines()]
    image_values = [v for v in images if v['kind'] == 'native_image_completed']
    assert len(image_values) == 8
    for resource in ('encoder', 'projector'):
        attempted = sum(v['samples'] for v in images if v['kind'] == 'native_image_attempted' and v['resource'] == resource)
        completed = sum(v['counts'][resource] for v in image_values)
        assert attempted == completed == 8
        assert top['actual_image_resources'][resource] == dict(attempted=8, completed=8)
    assert len(top['direct']) == 4
    for index in range(2):
        encoded = read(SAVED / f'task_{index}.json'); task = encoded['task']
        source_row = source['slots'][index]
        assert encoded['root_sha256'] == source_row['metadata']['root_sha256']
        assert encoded['raw_source_sha256'] == source_row['raw_source_sha256']
        assert encoded['plan_id'] == digest(source['plan'])
        assert file_sha(source_dir / source_row['raw_source_file']) == encoded['raw_source_sha256']
        with np.load(source_dir / source_row['raw_source_file'], allow_pickle=False) as raw:
            frames = [*raw['rgb_history'], raw['goal_rgb']]
            for k, frame in enumerate(frames):
                assert hashlib.sha256(frame.tobytes()).hexdigest() == image_values[4*index+k]['rgb_sha256']
            assert np.array_equal(task['root_raw_action_history'], raw['raw_action_history'])
        assert encoded['task_id'] == digest(task)
        assert task['latent_dim'] == 192 and task['horizon'] == 2
        assert task['representation'] == 'native_projected_cls'
        assert task['packet_logit_cap'] == float(.75*np.sqrt(5))
        for k, latent in enumerate(task['root_context']['latent_history'] + [task['goal']]):
            assert latent == image_values[4*index+k]['latent']
        assert np.array_equal(task['root_context']['latent_history'][0],task['root_context']['latent_history'][1])
        assert np.array_equal(task['root_context']['latent_history'][1],task['root_context']['latent_history'][2])
        assert encoded['history_semantics'] == 'causal_padding_not_observed_transitions'
        for method in ('full_H2_GN', 'full_H2_gradient'):
            directory = SAVED / f'root_{index}_{method}'
            reference = next(x for x in top['direct'] if x['root_sha256'] == encoded['root_sha256'] and x['method'] == method)
            assert Path(reference['report_path']) == directory / 'report.json'
            assert file_sha(directory / 'report.json') == reference['report_sha256']
            report = read(directory / 'report.json')
            item = audit_component(task, report)
            item['raw_control_J_packet_frobenius'] = audit_journal(task, report, directory / 'direct_journal.jsonl')
            item['root'] = index
            result['resources'].update(item['resources']);result['components'].append(item)
    for k, n in result['resources'].items():
        assert top['actual_direct_resources'][k] == dict(attempted=n, completed=n)
    assert dict(result['resources']) == dict(predict=120, action_encoder=120, jacobian=12,
        vjp_directions=2304, svd=6, linear_solve=6, proposal=12, selection=12)
    result['report_sha256'] = file_sha(SAVED / 'report.json')
    assert result['report_sha256'] == 'fad2c2ff4f409762fa5925f142f9b4d3edc371645fc4c114bfa69c996483f3d4'
    return result


def test_real_four_components():
    result = audit_saved()
    assert len(result['components']) == 4
    assert result['resources']['predict'] == 120


@pytest.mark.parametrize('corruption', ['gradient', 'history', 'acceptance', 'jacobian', 'Q', 'ball'])
def test_rejects_saved_receipt_corruption(corruption):
    task = read(SAVED / 'task_0.json')['task']
    r = read(SAVED / 'root_0_full_H2_gradient/report.json')
    r = deepcopy(r)
    step = r['direct_runs'][0][1]
    if corruption == 'gradient':
        step['derivative']['objective_gradient'][0] += 0.1
    elif corruption == 'history':
        step['proposal']['contexts'][0]['action_history'][0][0] += 1
    elif corruption == 'acceptance':
        step['accepted'] = not step['accepted']
    elif corruption == 'jacobian':
        step['derivative']['raw_terminal_jacobian'][0][0] += 1
    elif corruption == 'Q':
        task['projection'][0][0] = 0
    else:
        step['proposal']['logits'][0][0] = 100
    with pytest.raises(AssertionError):
        audit_component(task, r)


if __name__ == '__main__':
    print(json.dumps(audit_saved(), indent=2))
