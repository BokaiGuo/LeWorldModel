"""Independent saved-only native arithmetic; no producer, model or env imports."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def native(state, goal):
    delta = np.asarray(state, dtype=np.float64)-np.asarray(goal, dtype=np.float64)
    if delta.shape != (7,) or not np.isfinite(delta).all():
        raise ValueError('finite full native7')
    angle = abs(float(delta[4]))
    # Preserve the official formula, including its historical multi-turn behavior.
    angle_term = min(angle, 2*np.pi-angle)
    xy = float(np.linalg.norm(delta[:4]))
    return dict(reward=-float(np.linalg.norm(delta)), joint_xy=xy,
        native_success=bool(xy < 20 and angle_term < np.pi/9))


def audit(folder, expected):
    folder = Path(folder)
    if sha(folder/'report.json') != expected:
        raise ValueError('external actual report SHA')
    report = json.loads((folder/'report.json').read_bytes())
    if report['status'] != 'complete' or report['errors'] or len(report['rows']) != 40:
        raise ValueError('all40 actual complete required')
    events = [json.loads(line) for line in (folder/'journal.jsonl').read_bytes().splitlines()]
    counts = {}
    tail = None
    phase_counts = {}
    raw_count = 0
    rows = {r['job_digest']: r for r in report['rows']}
    for index, event in enumerate(events):
        body = {k: v for k, v in event.items() if k != 'event_sha256'}
        digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        if event['index'] != index or event['previous_sha256'] != tail or digest != event['event_sha256']:
            raise ValueError('independent full journal chain')
        tail = digest
        if event['kind'] in ('operation_attempted', 'operation_returned'):
            resource = counts.setdefault(event['resource'], dict(attempted=0, completed=0))
            resource['attempted' if event['kind'] == 'operation_attempted' else 'completed'] += 1
        if event['kind'] != 'raw_return':
            continue
        raw_count += 1
        phase = event['phase']
        phase_counts[phase] = phase_counts.get(phase, 0)+1
        row = rows[event['job_digest']]
        info = event['info']
        info_path = folder/info['file']
        if info_path.name != info['file'] or sha(info_path) != info['sha256']:
            raise ValueError('actual lossless info file')
        with np.load(info_path, allow_pickle=False) as arrays:
            def get(name):
                spec = info['tree'][name]
                value = arrays[spec['array']]
                if value.dtype.str != spec['dtype'] or list(value.shape) != spec['shape']:
                    raise ValueError('original dtype/shape')
                return value
            goal = get('goal_state')
            if phase != 'warmup' and not np.array_equal(goal, row['goal']):
                raise ValueError('actual post-binding goal')
            state = np.asarray(event['state'], dtype=np.float64)
            if (not np.array_equal(state[:2], get('pos_agent')) or
                    not np.array_equal(state[2:4], get('block_pose')[:2]) or
                    not np.array_equal(state[5:], get('vel_agent'))):
                raise ValueError('full native7 returned info join')
            metrics = native(state, goal)
        if (not np.isclose(metrics['reward'], event['reward'], rtol=0, atol=1e-10) or
                metrics['native_success'] != event['terminated'] or
                event['truncated'] is not False or event['raw_bits'] != event['actual_bits']):
            raise ValueError('independent reward/termination/raw mismatch')
    if counts != report['counts'] or phase_counts != dict(warmup=400, prefix=100, probe=200) or raw_count != 700:
        raise ValueError('complete physical resource and phase census')
    if tail != report['journal_tail'] or events[-1]['kind'] != 'run_finished':
        raise ValueError('independent terminal chain')
    if sha(folder/'report.json') != expected:
        raise ValueError('report changed during read')
    return dict(schema='MAIN.mixedmode_capability_native_audit.v1', report_sha256=expected,
        journal_events=len(events), journal_sha256=sha(folder/'journal.jsonl'),
        raw_returns=raw_count, phase_counts=phase_counts, independent_counts=counts,
        all_native_rewards_and_terminations_match=True, new_neural_calls=0,
        new_physics_calls=0, P1_admission=False, independent_confirmation=False)


def main():
    folder = ROOT/'results/agent_mixedmode_capability_physical_20261028_v1_MAIN01'
    destination = ROOT/'results/audit_mixedmode_capability_native_MAIN_20261028_v1_MAIN01.json'
    if destination.exists():
        raise FileExistsError(destination)
    result = audit(folder, '57ac436ff6cd7fcffcda62e0a165782b719c0383404647f2f87c1fa97ab009c8')
    payload = json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    with destination.open('xb') as stream:
        if stream.write(payload) != len(payload):
            raise OSError('short independent audit write')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(output=str(destination), sha256=sha(destination), result=result), sort_keys=True))


if __name__ == '__main__':
    main()
