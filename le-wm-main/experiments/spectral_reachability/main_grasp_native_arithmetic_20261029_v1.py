"""Independent arithmetic on the complete sealed GRASP native trajectory."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NAME = 'main_grasp_native_arithmetic_20261029_v1'
REPORT = ROOT/'results/agent_p2_history_grasp_physical_20261029_v2_MAIN01/report.json'
REPORT_SHA = 'd71588499aae674790ba95107650116b12aae07b5c17c718d432f003451534e6'
RECEIPT = ROOT/'results/agent_p2_history_grasp_physical_20261029_v2_consume_MAIN01.json'
RECEIPT_SHA = '17373c3d9878d2bfab9eebfa6ac16c11e8ebf2736d8b77c4d206b5a45cf4957e'


def bound(path, expected):
    b = path.read_bytes()
    if hashlib.sha256(b).hexdigest() != expected:
        raise ValueError('sealed source SHA')
    return json.loads(b)


def run():
    r, c = bound(REPORT, REPORT_SHA), bound(RECEIPT, RECEIPT_SHA)
    assert r['rows'] == c['rows'] and r['errors'] == c['errors'] == []
    assert c['physical_report_sha256'] == REPORT_SHA and len(r['rows']) == 2
    rows = []
    for slot, row in enumerate(r['rows']):
        assert row['root_slot'] == slot and row['own_returned'] == len(row['responses']) == 15
        goal = np.asarray(row['goal'], np.float64)
        values = []
        for step, response in enumerate(row['responses'], 1):
            assert response['step'] == step
            state = np.asarray(response['state'], np.float64)
            assert state.shape == goal.shape == (7,) and np.isfinite(state).all()
            d = state-goal
            xy, obj = float(np.linalg.norm(d[:4])), float(np.linalg.norm(d[2:4]))
            angle = min(abs(float(d[4])), 2*np.pi-abs(float(d[4])))
            wrapped = float(abs(np.arctan2(np.sin(d[4]), np.cos(d[4]))))
            distance = float(np.linalg.norm(d))
            success = bool(xy < 20 and angle < np.pi/9)
            assert response['reward'] == -distance
            assert response['native_success'] is response['terminated'] is success
            assert response['native_metrics']['native_angle_term'] == angle
            assert response['native_metrics']['object_position_error'] == obj
            assert response['native_metrics']['native_distance'] == distance
            assert not response['truncated'] and (step == 15 or not success)
            values.append(dict(step=step, joint_xy=xy, object_xy=obj, native_angle=angle,
                wrapped_angle=wrapped, velocity_error=float(np.linalg.norm(d[5:])),
                native7_distance=distance, native_success=success,
                wrapped_success=bool(xy < 20 and wrapped < np.pi/9), negative_native_angle=bool(angle < 0)))
        rows.append(dict(root_slot=slot, all_steps=values, endpoint=values[-1]))
    output = dict(schema=NAME, source_report_sha=REPORT_SHA, source_receipt_sha=RECEIPT_SHA,
        full_roots=2, full_responses=30, rows=rows, new_neural_calls=0, new_physics_calls=0,
        scientific_admission=False, equal_budget_comparison=False)
    path = ROOT/'results'/(NAME+'_MAIN01.json')
    with path.open('x') as f:
        json.dump(output, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         endpoints=[x['endpoint'] for x in rows]), sort_keys=True))


if __name__ == '__main__':
    run()
