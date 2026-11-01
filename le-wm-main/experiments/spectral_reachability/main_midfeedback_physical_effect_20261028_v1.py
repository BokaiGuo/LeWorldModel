"""Independent paired native-state arithmetic; depends on a pinned strict receipt."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NAME = 'main_midfeedback_physical_effect_20261028_v1'
OLD = ROOT/'results/agent_p2_contextual_composition_physical_handoff_20261028_v1_MAIN01/report.json'
NEW = ROOT/'results/agent_p2_contextual_midfeedback_recompile_20261028_v1_physical_MAIN01/report.json'
COMPILE = ROOT/'results/agent_p2_contextual_midfeedback_recompile_20261028_v1_cuda_MAIN01/report.json'
PINS = ((OLD, 'c9e8c37984926a68bf2e4d3a92fe47737c03e719ed51917da58b72c888eb038d'),
        (NEW, 'cd1accce54cbf6c48195490f8cd588dc7c4d701b1c6bc2f32844ab4d5a351603'),
        (COMPILE, '8146692ccc9a3c2c6c09ff3c99419274e2e8a8ff86bb3ac0465af01dc6401eea'))
AXES = ('joint_xy_error', 'agent_xy_error', 'object_xy_error',
        'agent_velocity_error', 'wrapped_angle_error', 'native_reward')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(state, goal):
    s, g = np.asarray(state, np.float64), np.asarray(goal, np.float64)
    require(s.shape == g.shape == (7,) and np.isfinite(s).all() and np.isfinite(g).all(), 'finite native7')
    d = s-g
    xy = float(np.linalg.norm(d[:4]))
    angle = abs(float(np.arctan2(np.sin(d[4]), np.cos(d[4]))))
    a = abs(float(d[4])); native_angle = min(a, 2*np.pi-a)
    return dict(joint_xy_error=xy, agent_xy_error=float(np.linalg.norm(d[:2])),
        object_xy_error=float(np.linalg.norm(d[2:4])), agent_velocity_error=float(np.linalg.norm(d[5:])),
        wrapped_angle_error=angle, native_reward=-float(np.linalg.norm(d)),
        native_success=bool(xy < 20 and native_angle < np.pi/9))


def compare(old, new, pair, compiled):
    require(old['index'] == new['index'] == pair['index'] == compiled['index'], 'same row')
    require(all(old[k] == new[k] for k in ('root_slot', 'condition', 'arm', 'task_id', 'goal', 'root_state')), 'same task/goal/root')
    require(old['status'] == new['status'] == 'horizon_completed' and
            old['own_returned'] == new['own_returned'] == 10, 'fixed complete10 cohort; no early endpoint substitution')
    left = [x for x in old['checkpoints'] if x['step'] == 5]
    right = [x for x in new['checkpoints'] if x['step'] == 5]
    require(len(left) == len(right) == 1 and left == right, 'exact shared prefix checkpoint/native/PNG')
    m0, m1 = metrics(old['final_state'], old['goal']), metrics(new['final_state'], new['goal'])
    for actual, strict in ((m0, pair['control']), (m1, pair['intervention'])):
        require(all(actual[k] == strict['final_metrics'][k] for k in (*AXES, 'native_success')), 'independent state arithmetic vs strict')
    delta = {k: m1[k]-m0[k] for k in AXES}
    require(all(delta[k] == pair['new_minus_control'][k] for k in AXES), 'independent native paired differences')
    old_q, new_q = [x['cost'] for x in compiled['evaluations'][:2]]
    return dict(index=new['index'], root_slot=new['root_slot'], condition=new['condition'], arm=new['arm'],
        same_prefix_exact=True, control=m0, intervention=m1, new_minus_control=delta,
        old_actual_mid_predicted_Q=old_q, new_actual_mid_predicted_Q=new_q,
        predicted_Q_delta=new_q-old_q, modeled_improvement=new_q < old_q)


def build(receipt, expected):
    for path, h in PINS:
        require(sha(path) == h, 'pinned actual report')
    require(sha(receipt) == expected, 'pinned strict receipt')
    strict = json.loads(Path(receipt).read_bytes())
    require(strict['status'] == 'paired_known_intervention_recorded_NOT_science' and
            strict['physical_report_sha256'] == PINS[1][1] and strict['feedback_compile_sha256'] == PINS[2][1] and
            strict['control_physical_report_sha256'] == PINS[0][1], 'actual full strict dependency')
    reports = [json.loads(p.read_bytes()) for p, _ in PINS]
    require(all(len(r['rows']) == 8 for r in (*reports, strict)), 'all8; never filter')
    rows = [compare(a, b, s, c) for a, b, s, c in zip(reports[0]['rows'], reports[1]['rows'], strict['rows'], reports[2]['rows'], strict=True)]
    return dict(schema=NAME, rows=rows, fixed_plans=8, known_roots=2, fresh_roots=0,
        axis_counts={k: dict(better=sum(r['new_minus_control'][k] < 0 for r in rows),
            worse=sum(r['new_minus_control'][k] > 0 for r in rows), tie=sum(r['new_minus_control'][k] == 0 for r in rows))
            for k in AXES if k != 'native_reward'},
        modeled_worse=sum(r['predicted_Q_delta'] > 0 for r in rows),
        control_success=sum(r['control']['native_success'] for r in rows),
        intervention_success=sum(r['intervention']['native_success'] for r in rows),
        authority=dict(reports={str(p): h for p, h in PINS}, strict_receipt=str(Path(receipt).absolute()), strict_sha256=expected,
            independent_state_arithmetic=True, full_upstream_rebuilt_here=False),
        independent_confirmation=False, P1_admission=False, P2_admission=False,
        interpretation='All known paired actions retained. Native reward has opposite benefit sign to error axes; no independent rates or equal-FLOPs claim.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('receipt', 'receipt-sha', 'output'):
        p.add_argument('--'+key, required=True)
    a = p.parse_args(); output = Path(a.output).absolute()
    require(output.resolve() == output and output.parent == ROOT/'results' and output.name.startswith(NAME+'_'), 'own output')
    result = build(a.receipt, a.receipt_sha)
    with output.open('xb') as f:
        f.write(json.dumps(result, sort_keys=True, allow_nan=False).encode()+b'\n')
    print(json.dumps(dict(output=str(output), sha256=sha(output), axis_counts=result['axis_counts'], modeled_worse=result['modeled_worse'])))


if __name__ == '__main__':
    main()
