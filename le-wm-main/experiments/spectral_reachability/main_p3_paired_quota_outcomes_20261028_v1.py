"""Fixed48 P3 outcomes; distinguish first-chunk evidence from closed-loop C3."""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECEIPT = ROOT/'results/agent_p3_paired_quota_physical_handoff_20261028_v1_consume_MAIN01.json'
RECEIPT_SHA = 'e138a33d175e1de850264c41f1df459354d77b1cedc7d3a3b0b50ed09c1e327c'
REPORT = ROOT/'results/agent_p3_paired_quota_physical_handoff_20261028_v1_MAIN01'
REPORT_SHA = '1cd5d5808b615fbf6519191a6862f966fb6b3a934c1b9c1a3a1091d6db1dac70'
METRICS = ('joint_xy_error', 'native_distance', 'object_position_error', 'wrapped_angle_error', 'velocity_error')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compact(row):
    result = row['physical_result']
    return dict(ordinal=row['ordinal'], child_id=row['child_id'], mode=row['mode'],
                prepared_route=row['route'], actual_physics_available=result is not None,
                missing_reason=row['missing_reason'], fresh_execution=row['fresh_execution'],
                start_sha256=row['start_sha256'], goal_sha256=row['goal_sha256'],
                command=deepcopy(row['command']), decision=deepcopy(row['decision']),
                execution_quota=deepcopy(row['execution_quota']),
                own_returned=None if result is None else result['own_returned'],
                status=None if result is None else result['status'],
                native_success=None if result is None else result['native_success'],
                final_metrics=None if result is None else deepcopy(result['final_metrics']))


def assemble(receipt):
    original = receipt['rows']
    require(receipt['complete'] is True and len(original) == 48 and
            receipt['fixed_denominator'] == 48 and receipt['child_denominator'] == 24,
            'full strict48 source, not successful-only subset')
    rows = [compact(row) for row in original]
    pairs = []
    for i in range(0, 48, 2):
        f, a = rows[i:i+2]
        require(f['ordinal'] == i and a['ordinal'] == i+1 and f['mode'] == 'fixed' and a['mode'] == 'adaptive'
                and f['child_id'] == a['child_id'], 'fixed24 paired order')
        for key in ('start_sha256', 'goal_sha256'):
            require(f[key] == a[key], 'same paired source identity '+key)
        available = f['actual_physics_available'] and a['actual_physics_available']
        equal5 = available and f['own_returned'] == a['own_returned'] == 5
        deltas = {}
        for metric in METRICS:
            x = f['final_metrics'][metric] if available else None
            y = a['final_metrics'][metric] if available else None
            require(x is None or (math.isfinite(x) and math.isfinite(y)), 'finite physical metric')
            deltas[metric] = None if x is None else y-x
        pairs.append(dict(child_id=f['child_id'], fixed_ordinal=i, adaptive_ordinal=i+1,
            both_physical_available=available, same_actual_five_step_endpoint=equal5,
            actual_steps=[f['own_returned'], a['own_returned']],
            native_success=[f['native_success'], a['native_success']],
            adaptive_minus_fixed_final_metric= deltas,
            same_raw_action=f['command'] == a['command'] if available else None,
            adaptive_imagined_horizon=None if a['decision'] is None else a['decision']['horizon'],
            actual_closed_loop_policy_compared=False))
    available = [p for p in pairs if p['both_physical_available']]
    equal5 = [p for p in pairs if p['same_actual_five_step_endpoint']]
    resources = receipt['runtime_actual_resources']
    completed = {mode: resources[mode]['completed'] for mode in ('fixed', 'adaptive')}
    return dict(schema='main_p3_paired_quota_outcomes_20261028_v1',
        status='fixed48_first_chunk_outcomes_not_closed_loop_C3', rows=rows, pairs=pairs,
        fixed_denominator=48, child_denominator=24, physical_pair_count=len(available),
        missing_pair_count=24-len(available), equal_actual5_pair_count=len(equal5),
        native_success_rows={mode:sum(r['native_success'] is True for r in rows if r['mode']==mode)
                             for mode in ('fixed', 'adaptive')},
        equal5_jointXY_descriptive=dict(
            adaptive_better=sum(p['adaptive_minus_fixed_final_metric']['joint_xy_error']<0 for p in equal5),
            adaptive_worse=sum(p['adaptive_minus_fixed_final_metric']['joint_xy_error']>0 for p in equal5),
            ties=sum(p['adaptive_minus_fixed_final_metric']['joint_xy_error']==0 for p in equal5)),
        runtime_completed_resources=completed,
        predict_calls_equal=completed['fixed']['predict_API']==completed['adaptive']['predict_API'],
        resource_vectors_equal=completed['fixed']==completed['adaptive'],
        prepared_route_census=receipt['census'], independent_roots=2, independent_pair_samples=None,
        confidence_interval=None, total_FLOPs=None, equal_FLOPs=False,
        new_neural_calls=0, new_physics_calls=0, P1_admission=False, P3_admission=False,
        C3_advantage_established=False,
        interpretation='All48 retained; first5 execution is not complete adaptive-H3 policy efficacy. Early success retained, not padded. Duplicate trajectories are not independent samples.')


def build():
    from experiments.spectral_reachability import agent_p3_paired_quota_physical_handoff_20261028_v1 as source
    require(sha(RECEIPT) == RECEIPT_SHA, 'pinned actual strict receipt')
    accepted = json.loads(RECEIPT.read_bytes())
    rebuilt = source.consume(REPORT, REPORT_SHA)
    require(canonical(accepted) == canonical(rebuilt), 'entire actual strict closure rebuilt')
    result = assemble(rebuilt)
    result['authority'] = dict(receipt=str(RECEIPT), receipt_sha256=RECEIPT_SHA,
                              report=str(REPORT/'report.json'), report_sha256=REPORT_SHA,
                              full_native_closure_rebuilt=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).absolute()
    require(output.resolve() == output and output.parent == ROOT/'results' and
            output.name.startswith('main_p3_paired_quota_outcomes_20261028_v1_'), 'exclusive own output')
    result = build()
    with output.open('xb') as stream:
        stream.write(canonical(result)+b'\n')
    print(json.dumps(dict(output=str(output), sha256=sha(output), status=result['status'],
                         physical_pairs=result['physical_pair_count'], successes=result['native_success_rows'])))


if __name__ == '__main__':
    main()
