"""Main independent saved arithmetic for the new sixteen-branch physical run.

No runtime/compiler/model/environment imports. Not a neural or physical replay.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[2]
RUN = PROJECT/'results/p3_joint_horizon_goal_metric_physical_20261027_v1_main01'
PREP = PROJECT/'results/p3_joint_horizon_goal_metric_physical_20261027_v1_prepared01'
REPORT_SHA = '6c3659e199a8151fee2fa2c020a0cfe63db00b691b947cdbeb762c3c18d6004a'
PREP_SHA = '1cfc5eb2ddcfc563176a3cce51a0ef90b88e92d298b3cf35d415c2e420317daf'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def index(rows, field='method'):
    out = {(r['root_sha256'], r[field]): r for r in rows}
    require(len(out) == len(rows), 'duplicate root/arm')
    return out


def pose(state, goal):
    s, g = np.asarray(state, float), np.asarray(goal, float)
    require(s.shape == g.shape == (7,) and np.isfinite(s).all() and np.isfinite(g).all(), 'finite seven-state')
    xy = float(np.linalg.norm(s[2:4]-g[2:4]))
    angle = float(abs((s[4]-g[4]+np.pi) % (2*np.pi)-np.pi))
    return dict(xy_distance=xy, angle_distance=angle, pose_cost=xy+20*angle)


def qcost(vector, goal, projection):
    z, g, q = np.asarray(vector, float), np.asarray(goal, float), np.asarray(projection, float)
    require(z.shape == g.shape == (192,) and q.shape == (192, 192) and
            all(np.isfinite(x).all() for x in (z, g, q)), 'native finite Q arrays')
    require(np.allclose(q, q.T, atol=1e-8, rtol=0) and np.allclose(q@q, q, atol=1e-8, rtol=0)
            and np.trace(q) > .5, 'nonzero orthogonal task metric')
    return float(np.mean((q@(z-g))**2))


def official(state, goal):
    # Preserve the installed evaluator's unmodified angular convention.
    # Do not substitute wrapped pose angle for its different metric.
    d = np.asarray(state, float)-np.asarray(goal, float)
    angle = min(abs(float(d[4])), 2*np.pi-abs(float(d[4])))
    position, xy = float(np.linalg.norm(d[:4])), float(np.linalg.norm(d[2:4]))
    return dict(official_position_error=position, agent_position_error=float(np.linalg.norm(d[:2])),
                object_position_error=xy, official_state_distance=float(np.linalg.norm(d)), angle_error=angle,
                official_success=bool(position < 20 and angle < np.pi/9),
                object_only_success=bool(xy < 20 and angle < np.pi/9))


def equal(a, b):
    require(np.isclose(a, b, atol=1e-10, rtol=1e-10), 'saved scalar differs')


def audit():
    inputs = {}
    def read(path, expected=None):
        path = Path(path).resolve()
        require(path.is_relative_to(PROJECT) and path.is_file() and path.stat().st_size < 32*1024**2,
                'bounded project input')
        raw = path.read_bytes()
        inputs[str(path)] = sha(raw)
        require(expected is None or sha(raw) == expected, 'input SHA')
        return raw
    def load(path, expected=None):
        return json.loads(read(path, expected))
    implementation_sha256 = sha(read(__file__))
    report = load(RUN/'report.json', REPORT_SHA)
    receipt = load(PREP/'prepare.json', PREP_SHA)
    protocol = load(PREP/'protocol.json', receipt['protocol_sha256'])
    require(report['consumer_prepare_sha256'] == PREP_SHA and
            report['planner_report_sha256'] == protocol['planner_report_sha256'] and
            report['status'] == 'guarded_calibration_recorded' and report['failures'] == [] and
            report['model_unchanged'] is report['source_verified'] is True, 'complete source-bound execution')
    require(all(report[f'P{i}_admission'] is False for i in range(1, 6)), 'no scientific admission')
    rows, plans, enc = index(report['rows']), index(protocol['plans']), index(report['encodings'])
    require(set(rows) == set(plans) == set(enc) and len(rows) == 16, 'sixteen-arm census')
    physical = load(RUN/'physical_report.json', report['physical_report_sha256'])
    require(physical['rows'] == report['rows'], 'physics persisted before encoding')
    budget = [json.loads(line) for line in read(RUN/'physical_budget.events.jsonl',
              report['physical_budget_journal_sha256']).splitlines()]
    actual = Counter((e['kind'], e['phase'], e['completed']) for e in budget)
    for kind, section in [('env_step', 'steps'), ('env_reset', 'resets')]:
        for phase, saved in report['physical_call_receipt'][section].items():
            require(actual[kind, phase, False] == saved['attempts'] and actual[kind, phase, True] == saved['returns'],
                    'actual phase attempts/returns')
    counts = Counter()
    for key, row in rows.items():
        plan, encoding = plans[key], enc[key]
        require(row['selection_sha256'] == plan['selection_sha256'] == encoding['selection_sha256'] and
                row['context_identity']['goal'] == rows[key[0], 'old_Q__uniform']['context_identity']['goal'] and
                row['context_identity'] == rows[key[0], 'old_Q__uniform']['context_identity'], 'source/reset identity')
        n = 5*plan['planned_horizon_tokens']
        require(row['status'] == 'completed' and row['endpoint_fresh'] and row['full_horizon_completed'] and
                row['physical_env_step_attempts'] == row['physical_env_step_returns'] == n and
                len(row['trace']) == n, 'complete fresh selected path')
        events = [json.loads(line) for line in read(row['journal_path'], row['journal_sha256']).splitlines()]
        require([e['kind'] for e in events] == ['initial']+['attempt', 'step_returned', 'observation']*n,
                'ordered physical journal')
        for i, trace in enumerate(row['trace']):
            require(events[1+3*i]['command'] == trace['command'] == plan['commands'][i], 'commands not reselected')
            require(trace['primitive_step'] == i+1 and not trace['terminated'] and not trace['truncated'], 'complete steps')
            equal(trace['reward'], -official(trace['state'], row['context_identity']['goal'])['official_state_distance'])
        for event in events:
            if 'rgb_path' in event:
                read(event['rgb_path'], event['rgb_file_sha256'])
                with Image.open(event['rgb_path']) as image:
                    rgb = np.asarray(image)
                require(rgb.shape == (224, 224, 3) and rgb.dtype == np.uint8 and
                        np.any(np.all(rgb == (65, 105, 225), axis=-1)), 'official visible RGB')
        require(row['final_state'] == row['trace'][-1]['state'] == events[-1]['state'] and
                row['terminal_rgb_path'] == events[-1]['rgb_path'], 'fresh final-state/RGB binding')
        rebuilt = pose(row['final_state'], row['context_identity']['goal'])
        for k, v in rebuilt.items():
            equal(v, row['pose'][k])
        for k, v in official(row['final_state'], row['context_identity']['goal']).items():
            if type(v) is bool:
                require(row['official'][k] is v, 'independent official task success')
            else:
                equal(v, row['official'][k])
        require(encoding['status'] == 'completed' and encoding['failure'] is None and
                encoding['terminal_rgb_sha256'] == row['terminal_rgb_sha256'], 'new endpoint encoding linkage')
        metric = protocol['root_metrics'][key[0]]
        for q, material in metric['metrics'].items():
            equal(qcost(encoding['terminal_vector'], metric['goal'], material['projection']), encoding['actual_costs'][q])
            predicted = qcost(plan['selected_predicted_context']['latent_history'][-1], metric['goal'], material['projection'])
            equal(predicted, encoding['predicted_costs'][q])
            equal(predicted, plan['predicted_costs'][q])
        counts['selected'] += n
        counts['official_success'] += int(row['official']['official_success'])
        counts['object_only_success'] += int(row['official']['object_only_success'])
    diagnostics = []
    for pair in report['comparisons']:
        left, right = (rows[pair['root_sha256'], pair[side]] for side in ('left', 'right'))
        same = left['planned_horizon_tokens'] == right['planned_horizon_tokens']
        require(pair['identical_initial_context'] and pair['selected_horizons'] ==
                [left['planned_horizon_tokens'], right['planned_horizon_tokens']], 'paired initial identity/H')
        if same:
            equal(right['pose']['pose_cost']-left['pose']['pose_cost'], pair['right_minus_left_pose'])
        else:
            require(pair['right_minus_left_pose'] is None and pair['classification'] == 'full_mixed_H2_H3', 'mixed-H not pooled')
        if pair['contrast'].startswith('schedule_effect__'):
            q = pair['contrast'].split('__', 1)[1]
            a, b = (enc[pair['root_sha256'], pair[side]] for side in ('left', 'right'))
            diagnostics.append(dict(root_sha256=pair['root_sha256'], metric=q, classification=pair['classification'],
                predicted_same_Q_delta=b['predicted_costs'][q]-a['predicted_costs'][q],
                actual_same_Q_delta=b['actual_costs'][q]-a['actual_costs'][q],
                pose_delta_same_H=pair['right_minus_left_pose'],
                scope='Descriptive same-root within-Q; mixed horizons not a matched control comparison'))
    encoding_events = [json.loads(line) for line in read(RUN/'encoding.events.jsonl').splitlines()]
    require(not any(e['kind'] == 'forbidden_predict' for e in encoding_events), 'no endpoint predict')
    for kind in ('encoder', 'projector'):
        entries = [e for e in encoding_events if e['kind'] == kind]
        require(entries == [dict(kind=kind, index=i, completed=done) for i in range(16) for done in (False, True)],
                'sixteen exact endpoint module attempts/returns')
        require(report['encoding_stats'][kind+'_attempts'] == report['encoding_stats'][kind+'_completed'] == 16,
                'encoding report totals')
    for path, expected in inputs.items():
        require(sha(Path(path).read_bytes()) == expected, 'input changed during audit')
    return dict(status='saved_arrays_verified', counts=dict(counts), report_sha256=REPORT_SHA,
                implementation_sha256=implementation_sha256,
                same_Q_schedule_diagnostics=diagnostics,
                comparisons=report['comparisons'], actual_physical_calls=report['physical_call_receipt'],
                inputs=inputs, new_neural_calls=0, new_physical_steps=0, P1_admission=False, P3_admission=False,
                scope='Known four roots; saved arithmetic/PNG/journal checks, not independent simulator/encoder regeneration or derivative validation')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit()
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ('status', 'counts', 'P1_admission', 'P3_admission')}))
