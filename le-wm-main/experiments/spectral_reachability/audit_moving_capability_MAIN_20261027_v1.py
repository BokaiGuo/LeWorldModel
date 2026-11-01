"""Independent saved-state probe cross-check; no producer/model/physics imports."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path, expected):
    require(sha(path) == expected, 'external file SHA mismatch')
    def pairs(items):
        out = {}
        for k, v in items:
            require(k not in out, 'duplicate key')
            out[k] = v
        return out
    def constant(value):
        raise ValueError('nonfinite constant ' + value)
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs, parse_constant=constant)


def key(row):
    return row['root_slot'], row['condition'], row['arm']


def features(row):
    state = np.asarray(row['state'], dtype=np.float64)
    require(state.shape == (7,) and np.isfinite(state).all(), 'native state7')
    require(row['snapshot']['state'] == row['state'], 'state snapshot join')
    require(row['snapshot']['state_float64_bits'] == state.astype('<f8').tobytes().hex(), 'state bytes')
    block = row['snapshot']['bodies']['block']
    return dict(object_xy=state[2:4], agent_xy=state[:2], joint_xy=state[:4],
                object_angle_unwrapped=np.asarray([block['unwrapped_angle']], dtype=np.float64))


def compare_group(rows, analyzed, upstream):
    require(len(rows) == 5 and [r['probe'] for r in rows] == ['zero', '+x', '-x', '+y', '-y'], 'fixed probe bank')
    require(all(r['status'] == 'probe_completed' and r['start_verified'] is True and
                len(r['responses']) == 5 and len(r['trajectory']) == 22 and
                r['start_state'] == rows[0]['start_state'] for r in rows), 'complete exact-start probes')
    endpoints = []
    for row in rows:
        seq = [x for x in row['trajectory'] if x['phase'] == 'probe']
        require([x['step'] for x in seq] == list(range(1, 6)), 'real probe steps')
        for observed, response in zip(seq, row['responses'], strict=True):
            require(observed['state'] == response['state'] and
                    np.array_equal(np.asarray(response['delta_native7']),
                                   np.asarray(response['state']) - row['start_state']), 'returned response consistency')
        endpoints.append(features(seq[-1]))
    amplitude = float(np.float32(.1))
    checked = 0
    summary = {}
    for space in endpoints[0]:
        matrix = np.column_stack([(endpoints[1][space] - endpoints[2][space]) / (2 * amplitude),
                                  (endpoints[3][space] - endpoints[4][space]) / (2 * amplitude)])
        target = analyzed['spaces'][space]
        np.testing.assert_allclose(matrix, target['matrix'], rtol=1e-13, atol=1e-13)
        singular = np.linalg.svd(matrix, compute_uv=False)
        np.testing.assert_allclose(singular, target['spectrum']['singular_values'], rtol=1e-13, atol=1e-13)
        if space != 'object_angle_unwrapped':
            indices = {'object_xy': slice(2, 4), 'agent_xy': slice(0, 2), 'joint_xy': slice(0, 4)}[space]
            direction = np.asarray(upstream['goal'])[indices] - np.asarray(upstream['root_state'])[indices]
            length = np.linalg.norm(direction)
            expected = None if length == 0 else direction / length
            if expected is None:
                require(target['task_direction'] is None, 'zero direction stays undefined')
            else:
                np.testing.assert_allclose(expected, target['task_direction'], rtol=1e-13, atol=1e-13)
        for i, endpoint in enumerate(endpoints):
            delta = endpoint[space] - endpoints[0][space]
            np.testing.assert_allclose(delta, target['finite_responses'][i]['delta'], rtol=1e-13, atol=1e-13)
            if space != 'object_angle_unwrapped' and expected is not None:
                np.testing.assert_allclose(float(expected @ delta), target['finite_responses'][i]['task_projection'],
                                           rtol=1e-13, atol=1e-13)
        summary[space] = dict(singular_values=singular.tolist(), matrix=matrix.tolist())
        checked += 1
    return dict(identity=list(key(rows[0])), matrices_checked=checked, spaces=summary)


def audit(directory, report_sha, analysis_path, analysis_sha):
    directory = Path(directory).resolve()
    report = read(directory/'report.json', report_sha)
    analysis = read(analysis_path, analysis_sha)
    require(report['status'] == 'capability_probes_recorded' and report['errors'] == [] and
            report['inputs_final_verified'] is True and report['new_neural_calls'] == 0 and
            analysis['status'] == 'analyzed' and analysis['P1_admission'] is False and
            analysis['physical_probe_report_sha256'] == report_sha, 'successful engineering evidence only')
    require(sha(directory/'journal.jsonl') == report['journal_sha256'], 'journal seal')
    previous = None
    census = Counter()
    event_count = 0
    for i, line in enumerate((directory/'journal.jsonl').read_bytes().splitlines()):
        e = json.loads(line)
        body = {k: v for k, v in e.items() if k != 'event_sha256'}
        digest = hashlib.sha256(json.dumps(body, sort_keys=True, allow_nan=False).encode()).hexdigest()
        require(e['index'] == i and e['previous_sha256'] == previous and e['event_sha256'] == digest, 'event chain')
        previous = digest
        if e['kind'] in ('operation_attempted', 'operation_completed'):
            census[e['resource'], e['kind']] += 1
        require(e['kind'] != 'operation_failed', 'no hidden dispatch failure')
        event_count += 1
    require(previous == report['journal_chain_tail'] and e['kind'] == 'run_finished', 'terminal journal')
    for resource, count in report['counts'].items():
        require(count['attempted'] == census[resource, 'operation_attempted'] and
                count['completed'] == census[resource, 'operation_completed'] and
                count['attempted'] == count['completed'], 'independent meter census')
    require(len(report['rows']) == 40 and len(analysis['rows']) == 8, 'fixed denominator')
    prepared_path = next(path for path, h in analysis['input_bindings'].items() if h == report['prepare_sha256'])
    prepared = read(prepared_path, report['prepare_sha256'])
    upstream = read(Path(prepared['physical_output'])/'report.json', prepared['physical_report_sha256'])
    checks = []
    for i, analyzed in enumerate(analysis['rows']):
        group = report['rows'][5*i:5*i+5]
        require(key(analyzed) == key(upstream['rows'][i]) == key(group[0]) and
                all(key(r) == key(group[0]) for r in group), 'case identities')
        checks.append(compare_group(group, analyzed, upstream['rows'][i]))
    require(sha(directory/'report.json') == report_sha and sha(analysis_path) == analysis_sha, 'final input seals')
    return dict(status='independently_verified', physical_report_sha256=report_sha,
                analysis_sha256=analysis_sha, branches=40, roots=2, matrices_checked=32,
                events_checked=event_count, counts=report['counts'], rows=checks,
                new_neural_calls=0, new_physics_calls=0, P1_admission=False,
                limitations=['known calibration roots only', 'finite-scale secants, not global controllability',
                             'no hidden solver/cache serialization claim'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('physical-output', 'physical-report-sha', 'analysis', 'analysis-sha', 'output'):
        p.add_argument('--'+name, required=True)
    a = p.parse_args()
    result = audit(a.physical_output, a.physical_report_sha, a.analysis, a.analysis_sha)
    value = json.dumps(result, sort_keys=True, allow_nan=False).encode()
    with Path(a.output).open('xb') as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(status=result['status'], sha256=sha(a.output), matrices=32, branches=40)))


if __name__ == '__main__':
    main()
