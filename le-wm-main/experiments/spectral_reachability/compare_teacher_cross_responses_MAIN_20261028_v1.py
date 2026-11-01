"""Fixed-bank saved physical response differences; no winner selection or admission."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load(path, expected):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('actual saved artifact SHA')
    return json.loads(raw)


def main():
    report = load(ROOT / 'results/agent_p1_teacher_cross_library_physical_20261028_v1_MAIN02/report.json',
                  '464656755bb83cec14b7acdd457641592b11be6d335f285166687df15368bc01')
    audit = load(ROOT / 'results/audit_teacher_cross_physical_MAIN_20261028_v1_MAIN02.json',
                 '3dfc301c40471e0620caf8660baf4d8ad6534f25055e546aadc7466f61e79b78')
    if len(report['rows']) != 40 or not all(r['actual20_returns'] for r in audit['rows']):
        raise ValueError('all40 actual physical endpoints required, RGB remains partial')
    groups = []
    for task in range(4):
        zrows, srows = report['rows'][10*task:10*task+5], report['rows'][10*task+5:10*task+10]
        zero = {role: np.asarray(rows[0]['last_return']['state'], np.float64) for role, rows in [('Z', zrows), ('S', srows)]}
        comparisons = []
        for z, s in zip(zrows, srows, strict=True):
            if z['task'] != task or s['task'] != task or z['library_label'] != s['library_label']:
                raise ValueError('full ordered same five-way common bank')
            if np.asarray(z['suffix_commands'], '<f4').tobytes() != np.asarray(s['suffix_commands'], '<f4').tobytes():
                raise ValueError('same literal suffix including signed zero')
            effects = {}
            for role, row in [('Z', z), ('S', s)]:
                endpoint = np.asarray(row['last_return']['state'], np.float64)
                delta = endpoint - zero[role]
                delta[4] = np.arctan2(np.sin(delta[4]), np.cos(delta[4]))
                baseline = np.asarray(report['rows'][10*task + (0 if role == 'Z' else 5)]['last_return']['state'], np.float64)
                goal = np.asarray(row['goal'], np.float64)
                effects[role] = dict(terminal_minus_same_elapsed_zero_state=delta.tolist(),
                                     object_xy_effect_norm=float(np.linalg.norm(delta[2:4])),
                                     joint_xy_goal_error_change=float(np.linalg.norm(endpoint[:4]-goal[:4])-np.linalg.norm(baseline[:4]-goal[:4])),
                                     RGB_available=row['status']=='horizon_completed')
            zd = np.asarray(effects['Z']['terminal_minus_same_elapsed_zero_state'])
            sd = np.asarray(effects['S']['terminal_minus_same_elapsed_zero_state'])
            comparisons.append(dict(library_label=z['library_label'], effects=effects,
                                    object_xy_response_difference=(sd-zd)[2:4].tolist(),
                                    joint_xy_error_change_difference=effects['S']['joint_xy_goal_error_change']-effects['Z']['joint_xy_goal_error_change']))
        groups.append(dict(task=task, root_slot=zrows[0]['root_slot'], condition=zrows[0]['condition'], all_five=comparisons))
    result = dict(status='all40_fixed_bank_same_elapsed_response_differences', groups=groups,
                  fixed_denominator=40, known_roots=2, independent_confirmation=False,
                  full_RGB_release=False, C2_non_greedy_premise=False, scientific_admission=False,
                  state_velocity_contact_mediation_isolated=False,
                  interpretation='same suffix and elapsed-zero control; state-dependent response only, not isolated contact causality or task success')
    output = ROOT / 'results/compare_teacher_cross_responses_MAIN_20261028_v1_MAIN02.json'
    payload = json.dumps(result, sort_keys=True, allow_nan=False).encode()
    with output.open('xb') as stream:
        if stream.write(payload) != len(payload):
            raise ValueError('short write')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(status=result['status'], output=str(output), sha256=hashlib.sha256(payload).hexdigest(),
                          fixed_denominator=40, scientific_admission=False)))


if __name__ == '__main__':
    main()
