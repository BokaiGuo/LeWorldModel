"""Saved native scalars: six fixed branches, three same-time checkpoints.

Imported full physical strict is authentication, not independent execution.
The arithmetic below never calls producer metrics, a model or an environment.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import shutil

import numpy as np

from experiments.spectral_reachability import agent_p2_common_resource_physical_20261029_v1 as physical

NAME = 'agent_p2_common_resource_physical_summary_20261029_v1'
ROOT = physical.ROOT
require, sha, raw = physical.require, physical.sha, physical.raw
CAP, FLOOR = 16*1024**2, 200*1024**2
CP = (5, 10, 15)
PAIRS = (('joint', 'composition'), ('joint', 'grasp'), ('composition', 'grasp'))
PINS = dict(zip(map(str, physical.own_files()), (
    'fb660748af42fa98bcb124fcb67b5ee21059f7e10b457496521afc0a1bbb19ff',
    '56cc4006779e45c441d52110666a0b511dd94e63d2305677c481abde1e9d531c',
    '32b6cb297e41cccd17f5de0d75c2f86567815770b98cd584b5ae607ac4eb895f'), strict=True))
SCALARS = ('native7_distance', 'native7_reward', 'native7_squared', 'joint_xy_error',
    'joint_xy_squared', 'agent_xy_error', 'agent_xy_squared', 'block_xy_error',
    'block_xy_squared', 'raw_angle_squared', 'agent_velocity_error',
    'agent_velocity_squared', 'native_angle_term', 'wrapped_angle_error')
ERRORS = tuple(k for k in SCALARS if k != 'native7_reward')
FLAGS = dict(P1_admission=False, P2_admission=False, P2_mechanism=False,
    scientific_admission=False, freshness=False, independent_confirmation=False,
    equal_FLOPs=False, equal_total_compute=False, selection=False)


def own_files():
    return [Path(__file__).resolve(), ROOT/'tests'/('test_'+NAME+'.py'), ROOT/'innerdocs'/(NAME+'.md')]


def digest(x):
    return hashlib.sha256(raw(x)).hexdigest()


def read_bound(path, expected, cap=physical.META):
    physical.native.actual_hash(expected)
    p = Path(physical.native.absolute(str(path)))
    require(p.is_file() and not p.is_symlink() and p.stat().st_size <= cap, 'bounded regular saved JSON')
    b = p.read_bytes()
    require(len(b) <= cap and hashlib.sha256(b).hexdigest() == expected, 'actual external saved SHA')
    return physical.native.decode(b)


def vec7(x):
    require(type(x) is list and len(x) == 7 and all(type(v) in (int, float) for v in x), 'numeric native7, no bool')
    a = np.asarray(x, dtype=np.float64)
    require(np.isfinite(a).all(), 'finite native7')
    return a


def scalars(state, goal):
    """Independent float64 arithmetic, literal installed native success formula.

    Native7 raw angle and native min-angle (possibly negative outside one turn)
    are deliberately NOT replaced by the wrapped-angle diagnostic.
    """
    s, g = vec7(state), vec7(goal)
    d = s-g
    sq = d*d
    xy = float(np.linalg.norm(d[:4]))
    angle = float(min(abs(d[4]), 2*np.pi-abs(d[4])))
    wrapped = float(abs(np.arctan2(np.sin(d[4]), np.cos(d[4]))))
    result = dict(state=s.tolist(), goal=g.tolist(), residual7=d.tolist(),
        native7_distance=float(np.linalg.norm(d)), native7_reward=-float(np.linalg.norm(d)),
        native7_squared=float(np.sum(sq, dtype=np.float64)), joint_xy_error=xy,
        joint_xy_squared=float(np.sum(sq[:4], dtype=np.float64)),
        agent_xy_error=float(np.linalg.norm(d[:2])), agent_xy_squared=float(np.sum(sq[:2], dtype=np.float64)),
        block_xy_error=float(np.linalg.norm(d[2:4])), block_xy_squared=float(np.sum(sq[2:4], dtype=np.float64)),
        raw_angle_squared=float(sq[4]), agent_velocity_error=float(np.linalg.norm(d[5:])),
        agent_velocity_squared=float(np.sum(sq[5:], dtype=np.float64)),
        native_angle_term=angle, negative_native_angle=bool(angle < 0), wrapped_angle_error=wrapped,
        native_success=bool(xy < 20 and angle < np.pi/9),
        wrapped_success_diagnostic=bool(xy < 20 and wrapped < np.pi/9),
        state7_order=['agent_x','agent_y','block_x','block_y','block_angle_mod2pi','agent_vx','agent_vy'])
    require(all(np.isfinite(result[k]) for k in SCALARS), 'finite float64 scalar result')
    return result


def identity(p, root):
    slot = p['model_slots'][root]
    native = p['source_roots'][root]
    require(slot['root_slot'] == root and slot['condition'] == 'off', 'original off model slot')
    return dict(root_slot=root, task_id=slot['task_id'], task_file=slot['task_file'], task_sha256=slot['task_sha256'],
        source_slot_sha256=digest(p['source_slots'][root]), source_report_sha256=p['source_report_sha256'],
        native_root_sha256=native['root_sha256'], native_goal_sha256=digest(native['goal']),
        native_goal_pose_sha256=digest(native['goal_pose']), model_tensor_hash=p['expected_model_hash'],
        causal_context_sha256=digest(slot['packet']['root_context']), model_goal_sha256=digest(slot['packet']['goal']),
        warm_raw10_sha256=digest(slot['warm_raw10']), source_root_metadata_sha256=digest(native))


def panel(p, strict):
    """Pure saved seam; production build always reconstructs full strict first."""
    physical.validate_execution(p)
    require(strict['schema'] == physical.NAME+'.consume' and strict['fixed_denominator'] == 6
        and strict['known_roots'] == 2 and strict['fresh_roots'] == strict['inherited'] == 0
        and strict['plans'] == p['plans'] and strict['planner_authority'] == p['planner_authority']
        and len(strict['rows']) == 6 and strict['new_neural_calls'] == strict['new_physics_calls'] == 0,
        'complete six-slot original physical authority')
    require(all(strict[k] is False for k in FLAGS if k != 'selection'), 'no inherited science/equal-compute admission')
    rows = []
    for i, (root, condition, arm) in enumerate(physical.ORDER):
        branch = strict['rows'][i]
        plan = p['plans'][i]
        source = p['source_roots'][root]
        ident = identity(p, root)
        require((branch['index'], branch['root_slot'], branch['condition'], branch['arm']) == (i, root, condition, arm)
            and branch['task_id'] == ident['task_id'] and branch['plan'] == plan, 'fixed physical identity/order/task/plan')
        baseline = scalars(source['root_state'], source['goal'])
        if branch['initial_exact']:
            require(branch['root_state'] == source['root_state'] and branch['goal'] == source['goal'], 'actual root/goal matches source baseline')
        cps = {f['step']: f for f in branch['checkpoints']}
        responses = {f['step']: f for f in branch['responses']}
        require(len(cps) == len(branch['checkpoints']) and len(responses) == len(branch['responses'])
            and branch['missing_checkpoints'] == [s for s in physical.CP if s not in cps], 'unique actual checkpoints and exact missing list')
        if plan is None:
            require(branch['status'] == 'export_missing' and not responses and not cps and branch['own_returned'] == 0, 'missing export never executed')
        else:
            require(plan['planner_plan']['model_hash'] == ident['model_tensor_hash']
                and plan['task_id'] == ident['task_id'] and plan['native_root_sha256'] == ident['native_root_sha256'], 'model/raw/native same branch')
        for step in CP:
            f, response = cps.get(step), responses.get(step)
            measured = None
            if response is not None:
                require(branch['initial_exact'] and response['step'] <= branch['own_returned'] and plan is not None, 'actual own return before measurement')
                measured = scalars(response['state'], source['goal'])
                if f is not None:
                    require(f['state'] == response['state'] == f['snapshot']['state'], 'same-time checkpoint and actual returned state')
            else:
                require(f is None, 'no checkpoint without same-time labelled return')
            missing = []
            if measured is None:
                missing.append('export_missing' if plan is None else 'no_labelled_return_at_requested_step')
                if branch['status'] == 'native_stopped': missing.append('native_stop_no_padding')
                if branch['first_error'] is not None: missing.append('engineering_incident')
            if f is None: missing.append('checkpoint_RGB_missing')
            commands = None if plan is None else physical.raw15(plan['commands'])[:step]
            rows.append(dict(branch_index=i, root_slot=root, arm=arm, step=step, identity=deepcopy(ident),
                branch_status=branch['status'], selection_digest=branch['selection_digest'],
                requested_prefix_raw_float32_sha256=None if commands is None else hashlib.sha256(commands.tobytes()).hexdigest(),
                actual_own_attempted=branch['own_attempted'], actual_own_returned=branch['own_returned'],
                labelled_response_steps=sorted(responses), first_error=deepcopy(branch['first_error']),
                secondary_errors=deepcopy(branch['secondary_errors']), missing=missing,
                checkpoint=None if f is None else {k:deepcopy(f[k]) for k in ('step','png','file_sha256','pixel_sha256')},
                state_pointer=None if response is None else f'/rows/{i}/responses/{branch["responses"].index(response)}/state',
                baseline_reference=deepcopy(baseline), baseline_physically_verified=branch['initial_exact'], metrics=measured,
                error_reduction_from_original_root=None if measured is None else {k:baseline[k]-measured[k] for k in ERRORS},
                native7_reward_gain=None if measured is None else measured['native7_reward']-baseline['native7_reward'],
                terminated=None if response is None else response['terminated'],
                truncated=None if response is None else response['truncated']))
    pairs = []
    for root in (0, 1):
        for step in CP:
            for a, b in PAIRS:
                left, right = [next(r for r in rows if (r['root_slot'],r['step'],r['arm']) == (root,step,arm)) for arm in (a,b)]
                require(left['identity'] == right['identity'], 'same task/source/context/model/goal at same physical step')
                known = left['metrics'] is not None and right['metrics'] is not None
                pairs.append(dict(root_slot=root, step=step, a=a, b=b, known=known,
                    a_branch_index=left['branch_index'], b_branch_index=right['branch_index'],
                    missing_arms=[arm for arm,r in ((a,left),(b,right)) if r['metrics'] is None],
                    delta_b_minus_a=None if not known else {k:right['metrics'][k]-left['metrics'][k] for k in SCALARS},
                    error_reduction_b_minus_a=None if not known else {k:right['error_reduction_from_original_root'][k]-left['error_reduction_from_original_root'][k] for k in ERRORS},
                    native_success_a=None if left['metrics'] is None else left['metrics']['native_success'],
                    native_success_b=None if right['metrics'] is None else right['metrics']['native_success']))
    # Aggregate only matching method-pair/time. Never mix cp5 and cp15 or
    # substitute early-stop endpoints; two known-root clusters, not n=18 IID.
    aggregates = []
    for step in CP:
        for a,b in PAIRS:
            group = [r for r in pairs if (r['step'],r['a'],r['b']) == (step,a,b)]
            known = [r for r in group if r['known']]
            aggregates.append(dict(step=step,a=a,b=b,fixed_root_denominator=2,known_pairs=len(known),missing_pairs=2-len(known),
                delta_b_minus_a={k:dict(known_sum=None if not known else sum(r['delta_b_minus_a'][k] for r in known),
                    known_mean=None if not known else sum(r['delta_b_minus_a'][k] for r in known)/len(known)) for k in SCALARS}))
    return dict(rows=rows,pairs=pairs,same_time_aggregates=aggregates,fixed_branch_denominator=6,
        fixed_checkpoint_denominator=18,fixed_pair_denominator=18,known_roots=2,fresh_roots=0,
        known_checkpoint_scalars=sum(r['metrics'] is not None for r in rows),
        observed_success_by_time=[dict(step=s,fixed_denominator=6,known=sum(r['metrics'] is not None for r in rows if r['step']==s),
            success=sum(r['metrics']['native_success'] for r in rows if r['step']==s and r['metrics'] is not None)) for s in CP],
        physical_resources=deepcopy(strict['counts']),planning_resources=deepcopy(p['planner_authority']['resources']),
        errors=deepcopy(strict['errors']),**FLAGS)


def build_summary(output, report_sha, receipt, receipt_sha):
    for f,h in PINS.items(): require(sha(f) == h, 'frozen physical triple SHA')
    directory = Path(physical.native.absolute(str(output)))
    report = read_bound(directory/'report.json', report_sha)
    persisted = read_bound(receipt, receipt_sha)
    strict = physical.consume(directory, report_sha)
    require(raw(strict) == raw(persisted), 'whole external physical strict return rebuilt, not status or envelope')
    binding = report['execution_binding']
    p = read_bound(binding['prepared'], binding['prepare_sha256'])
    result = panel(p, strict)
    bindings = deepcopy(strict['bindings'])
    physical.native.merge(bindings, {str(directory/'report.json'):report_sha,
        str(Path(receipt).absolute()):receipt_sha, binding['prepared']:binding['prepare_sha256'], **PINS})
    for f in own_files(): physical.native.merge(bindings, {str(f):sha(f)})
    for f,h in bindings.items(): require(sha(f) == h, 'full original strict source/array/own closure stable')
    return dict(schema=NAME+'.summary',arguments=dict(output=str(directory),report_sha=report_sha,
        receipt=str(Path(receipt).absolute()),receipt_sha=receipt_sha),
        authentication='imported original physical.consume full return, not independent dynamics/neural validation',
        arithmetic='independent float64 saved native7; never producer metric_breakdown/final_metrics/reward',
        interpretation='two known roots; same-time comparisons only; no outcome selection or equal-FLOPs claim',
        metric_units=dict(XY='native pixel coordinates',velocity='native agent velocity',angle='radians',
            native7='original mixed-unit Euclidean norm, not a new combined score',
            squared='native7 squared decomposed only; no wrapped-angle replacement'),
        original_physical_status=report['status'],original_wall_seconds=report['wall_seconds'],
        bindings=bindings,summary=result,new_neural_calls=0,new_physics_calls=0)


def write_new(output, value):
    p = Path(physical.native.absolute(str(output)))
    b = raw(value)
    require(p.name.startswith(NAME+'_') and not p.exists() and not p.is_symlink()
        and len(b) <= CAP and shutil.disk_usage(p.parent).free >= FLOOR+len(b), 'exclusive own summary cap/floor')
    with p.open('xb') as stream:
        require(stream.write(b) == len(b), 'complete summary write')
        stream.flush(); os.fsync(stream.fileno())
    physical.paired.fsync_dir(p.parent)
    return dict(path=str(p),sha256=sha(p),bytes=len(b))


def verify(path, expected):
    saved = read_bound(path, expected, CAP)
    require(saved['schema'] == NAME+'.summary', 'own summary schema')
    rebuilt = build_summary(**saved['arguments'])
    require(raw(saved) == raw(rebuilt) and sha(path) == expected, 'entire summary independently rebuilt from authenticated saved inputs')
    return dict(verified=True,summary_sha256=expected,fixed_branch_denominator=6,fixed_checkpoint_denominator=18,new_neural_calls=0,new_physics_calls=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    build = sub.add_parser('build')
    for key in ('physical-output','report-sha','strict-receipt','strict-sha','output'): build.add_argument('--'+key,required=True)
    check = sub.add_parser('verify')
    check.add_argument('--summary',required=True); check.add_argument('--summary-sha',required=True)
    args = parser.parse_args(argv)
    if args.command == 'build':
        result = build_summary(args.physical_output,args.report_sha,args.strict_receipt,args.strict_sha)
        result = write_new(args.output,result)
    else: result = verify(args.summary,args.summary_sha)
    print(raw(result).decode())


if __name__ == '__main__': main()
