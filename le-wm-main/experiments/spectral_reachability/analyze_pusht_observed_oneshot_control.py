"""Known-root paired feedback intervention; no held-out or fairness inference."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.spectral_reachability.run_pusht_observed_oneshot_suffix_control import one_shot


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def keyed(rows, key):
    result = {r[key]: r for r in rows}
    if not result or len(result) != len(rows):
        raise ValueError('unique identities required')
    return result


def analyze(control, original, official):
    for report in (control, original):
        if report['status'] != 'completed' or not report['gates'] or not all(v is True for v in report['gates'].values()):
            raise ValueError('complete physical evidence required')
    if official['status'] != 'completed':
        raise ValueError('completed official task audit required')
    cm, om, tm = [keyed(r['rows'], 'root_sha256') for r in (control, original, official)]
    if len(cm) != 4 or not (set(cm) == set(om) == set(tm)):
        raise ValueError('four identical root sets required')
    rows = []
    for root in sorted(cm):
        new, old, task = [keyed(r[root]['branches'], 'candidate_id') for r in (cm, om, tm)]
        if len(new) != 16 or not (set(new) == set(old) == set(task)):
            raise ValueError('full identical candidate banks required')
        if cm[root]['selected_candidate_id'] != om[root]['selected_candidate_id']:
            raise ValueError('selector changed')
        values = []
        for name in sorted(new):
            n, o, t = new[name], old[name], task[name]
            if n['prefix_replay_exact'] is not True or o['status'] != 'executed' or len(o['stages']) != 2 or len(t['stages']) != 2:
                raise ValueError('exact prefix replay and full old execution required')
            values.append({'candidate_id': name,
                'pose_delta': n['pose']['pose_cost']-o['stages'][-1]['pose']['pose_cost'],
                'official_distance_delta': n['official']['official_state_distance']-t['stages'][-1]['official_state_distance'],
                'metric_delta': n['actual_metric_cost']-o['stages'][-1]['metric_cost'],
                'official_success': n['official']['official_success'], 'object_success': n['official']['object_only_success']})
        selected = next(v for v in values if v['candidate_id'] == cm[root]['selected_candidate_id'])
        rows.append({'root_sha256': root, 'root_index': cm[root]['root_index'], 'candidates': values,
            'mean_pose_delta': float(np.mean([v['pose_delta'] for v in values])),
            'mean_official_distance_delta': float(np.mean([v['official_distance_delta'] for v in values])),
            'mean_metric_delta': float(np.mean([v['metric_delta'] for v in values])), 'selected': selected,
            'pose_better': sum(v['pose_delta'] < 0 for v in values), 'pose_worse': sum(v['pose_delta'] > 0 for v in values),
            'pose_equal': sum(v['pose_delta'] == 0 for v in values)})
    return {'schema': 'pusht_observed_oneshot_control_analysis.v1', 'rows': rows,
            'scope': 'known calibration paired one-shot feedback intervention, not independent P1 or equal-total-budget comparison',
            'P1_admission': False, 'outcome_selection': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('control', 'original', 'official', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    control, original, official = [json.loads(path.read_text()) for path in (args.control, args.original, args.official)]
    if control['protocol']['bindings'].get(str(args.original.resolve())) != digest(args.original):
        raise ValueError('physical control baseline binding mismatch')
    if official['inputs'].get(str(args.original.resolve())) != digest(args.original):
        raise ValueError('official baseline binding mismatch')
    for row in control['rows']:
        for branch in row['branches']:
            plan_path = Path(branch['plan_path'])
            if digest(plan_path) != branch['plan_sha256']:
                raise ValueError('pre-execution plan changed')
            plan = json.loads(plan_path.read_text())
            solution = one_shot(plan['goal'], plan['nominal'], plan['jacobian'], plan['projection'])
            np.testing.assert_allclose(solution, plan['params_float64'], atol=1e-10, rtol=1e-10)
    result = analyze(control, original, official)
    result['inputs'] = {str(path.resolve()): digest(path) for path in (args.control, args.original, args.official)}
    result['one_shot_plans_recomputed'] = 64
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps([{k: r[k] for k in ('root_index', 'mean_pose_delta', 'mean_official_distance_delta', 'mean_metric_delta', 'pose_better', 'pose_worse', 'pose_equal')} for r in result['rows']], indent=2))
