"""Independent saved native-log recomputation; imports no producer/consumer."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'results/agent_p2_three_program_physical_20261029_v1_MAIN01/report.json'
REPORT_SHA = '49b682d0ca326590c88ef69d74eecfdd9d5c19f21667e790a7c1458aac98e064'


def require(value, message):
    if not value:
        raise ValueError(message)


def metrics(state, goal):
    d = np.asarray(state, dtype=np.float64) - np.asarray(goal, dtype=np.float64)
    require(d.shape == (7,) and np.isfinite(d).all(), 'finite native7')
    angle = float(min(abs(d[4]), 2*np.pi-abs(d[4])))
    return dict(joint_xy=float(np.linalg.norm(d[:4])), object_xy=float(np.linalg.norm(d[2:4])),
                wrapped_angle=float(abs(np.arctan2(np.sin(d[4]), np.cos(d[4])))),
                native_angle=angle, native_distance=float(np.linalg.norm(d)),
                success=bool(np.linalg.norm(d[:4]) < 20 and angle < np.pi/9))


def run(output):
    require(hashlib.sha256(REPORT.read_bytes()).hexdigest() == REPORT_SHA, 'sealed actual report')
    report = json.loads(REPORT.read_bytes())
    require(len(report['rows']) == 4 and report['errors'] == [], 'all four native branches')
    rows = []
    for i, row in enumerate(report['rows']):
        require(row['root_slot'] == i//2 and row['own_returned'] == 15, 'fixed order and full15')
        plan = row['plan']
        raw = np.asarray(plan['commands'], dtype='<f4')
        require(raw.shape == (15, 2) and raw.tobytes().hex() == plan['planner_plan']['raw_float32_bits'], 'literal full15 raw bits')
        require(np.array_equal(raw, np.asarray(plan['planner_plan']['commands'], dtype='<f4')), 'planner/executor raw equality')
        require(hashlib.sha256(raw.tobytes()).hexdigest() == plan['raw_float32_sha256'], 'raw checksum')
        require(len(row['responses']) == 15, 'all native responses')
        for step, response in enumerate(row['responses'], 1):
            m = metrics(response['state'], row['goal'])
            require(response['step'] == step and response['reward'] == -m['native_distance'], 'native reward')
            require(response['terminated'] is m['success'] and response['truncated'] is False, 'original success and termination')
            frames = [f for f in row['trajectory'] if f['phase'] == 'plan' and f['step'] == step and f['overlay_index'] == 0]
            require(len(frames) == 1 and frames[0]['state'] == response['state'], 'same-time native state/frame join')
        m = metrics(row['final_state'], row['goal'])
        require(row['final_state'] == row['responses'][-1]['state'] and row['native_success'] is m['success'], 'terminal state/label')
        require(m['object_xy'] == row['final_metrics']['object_position_error'], 'object metric exact')
        require(m['wrapped_angle'] == row['final_metrics']['wrapped_object_angle_error'], 'wrapped angle exact')
        rows.append(dict(root_slot=row['root_slot'], arm=row['arm'], metrics=m,
                         predicted_Q=plan['planner_plan']['predicted_Q']))
    result = dict(schema='main_p2_native_recompute_20261029_v1', report_sha256=REPORT_SHA,
                  response_denominator=60, branches=rows, new_neural_calls=0, new_physics_calls=0,
                  scientific_admission=False, equal_compute=False, fresh_roots=False)
    output = Path(output).resolve()
    require(output.parent == ROOT/'results' and output.name.startswith('main_p2_native_recompute_20261029_v1_'), 'exclusive result path')
    with output.open('x') as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.write('\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    run(parser.parse_args().output)
