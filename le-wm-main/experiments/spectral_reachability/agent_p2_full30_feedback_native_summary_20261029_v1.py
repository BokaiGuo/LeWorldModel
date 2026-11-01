"""Independent O/I/F native7 saved arithmetic. No producer or numerical-library imports."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import struct

NAME = 'agent_p2_full30_feedback_native_summary_20261029_v1'
OLD = 'agent_p2_two_program_full30_physical_20261029_v1'
NEW = 'agent_p2_full30_control15_feedback_physical_20261029_v1'
ROOT = Path(__file__).resolve().parents[2]
CP = (0, 5, 10, 15, 20, 25, 30)
POLICIES = ('O', 'I', 'F')
ARMS = ('two_program_transfer', 'joint_two_round')
AXES = ('agent_x', 'agent_y', 'object_x', 'object_y', 'object_angle_raw', 'agent_vx', 'agent_vy')
FLAGS = ('P1_admission', 'P2_admission', 'P2_mechanism', 'scientific_admission',
         'freshness', 'independent_confirmation', 'physical_release', 'equal_FLOPs', 'equal_total_compute')
OLD_SHA = 'aabc5cce3855021eba0fe5751c0ae87809cfefb4c2ed54b0d2899048fd6b3cd8'
JOB_SHAS = ('565c432e08a7965079a10c709d90773edab4a967004ec07e43202b88a6c96d6a',
            'fc08ef91da4f158218d78a17da640a2664a8d1e63ebdf6e3b6fe6b6df5c2ee5e',
            '94157f79fd3ca2ea28e825914e56a9f364ba4f87e7871b6d09d9c9eeaa7b9c38',
            '8a66e6bb41c716c4d0b01b405745a7d356fcd4cfc7b6e17178e4084c27199f2d')
ENV_REL = '.venv/lib/python3.12/site-packages/stable_worldmodel/envs/pusht/env.py'
ENV_SHA = 'c6d2104adc368ff1c352edc0e35ac74da9f07fda3a748ddbef0ba6b7a2d1daa2'
LIMIT = 16 * 1024**2


def require(ok, label):
    if not ok:
        raise ValueError(label)


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode() + b'\n'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def absolute(path):
    p = Path(path)
    require(p.is_absolute() and p.resolve() == p and not p.is_symlink(), 'canonical absolute path')
    return p


def read(path, expected):
    p = absolute(path)
    require(p.is_file() and p.stat().st_size <= LIMIT, 'bounded regular JSON')
    b = p.read_bytes()
    require(len(b) <= LIMIT and hashlib.sha256(b).hexdigest() == expected, 'external SHA256')
    def unique(items):
        out = {}
        for k, v in items:
            require(k not in out, 'duplicate JSON key')
            out[k] = v
        return out
    value = json.loads(b, object_pairs_hook=unique)
    encode(value)
    return value


def exact(a, b):
    return encode(a) == encode(b)


def vector(value):
    require(type(value) is list and len(value) == 7 and
            all(type(v) in (int, float) and math.isfinite(v) for v in value), 'finite native state7/goal7')
    return [float(v) for v in value]


def metrics(state, goal):
    s, g = vector(state), vector(goal)
    d = [x-y for x, y in zip(s, g)]
    q = [x*x for x in d]
    require(all(math.isfinite(x) for x in q), 'finite squared axes')
    xy = math.fsum(q[:4]); total = math.fsum(q)
    native_angle = min(abs(d[4]), 2*math.pi-abs(d[4]))
    wrapped = abs(math.atan2(math.sin(d[4]), math.cos(d[4])))
    return dict(axis_residual=d, axis_squared=q,
                agent_xy_squared=math.fsum(q[:2]), object_xy_squared=math.fsum(q[2:4]),
                joint_xy_squared=xy, agent_xy_error=math.sqrt(math.fsum(q[:2])),
                object_xy_error=math.sqrt(math.fsum(q[2:4])),
                raw_angle_squared=q[4], velocity_squared=math.fsum(q[5:]),
                velocity_error=math.sqrt(math.fsum(q[5:])), object_pose=s[2:5],
                native7_squared=total, joint_xy_error=math.sqrt(xy), native7_distance=math.sqrt(total),
                native7_reward=-math.sqrt(total), native_angle_term=native_angle,
                negative_native_angle=native_angle < 0, wrapped_angle_error=wrapped,
                native_success=math.sqrt(xy) < 20 and native_angle < math.pi/9,
                wrapped_success_diagnostic=math.sqrt(xy) < 20 and wrapped < math.pi/9)


def authority(receipt, old, job=None):
    require(receipt['schema'] == (OLD if old else NEW)+'.consume', 'complete receipt schema')
    require(receipt['status'] == ('full30_four_native_recorded_NOT_admitted' if old else
                                'feedback_two_native_recorded_NOT_admitted'), 'native recorded status')
    n = 4 if old else 2
    require(receipt['fixed_denominator'] == n and len(receipt['rows']) == len(receipt['plans']) == n,
            'fixed receipt denominator')
    require(receipt['errors'] == [] and receipt['own_native_saved_reconstructed_here'] is True
            and receipt['known_roots'] == 2 and receipt['fresh_roots'] == 0, 'complete own native authority')
    require(all(receipt[k] is False for k in FLAGS), 'no scientific admission')
    if not old:
        require(type(receipt['job_index']) is int and receipt['job_index'] == job
                and receipt['global_policy_slots'] == 12, 'fixed job index/all12')
    for row, plan in zip(receipt['rows'], receipt['plans'], strict=True):
        require(exact(row['plan'], plan), 'row/plan exact join')
    for name, count in receipt['counts'].items():
        require(type(count) is dict and set(count) == {'attempted', 'completed'} and
                all(type(v) is int and v >= 0 for v in count.values()) and
                count['completed'] <= count['attempted'], 'resource axes, not zero-filled')


def row_panel(row, base, policy):
    require((row['root_slot'], row['arm'], row['condition']) == (base//2, ARMS[base%2], 'off'), 'fixed base identity')
    require(row['first_error'] is None and row['secondary_errors'] == [] and
            row['status'] in ('horizon_completed', 'native_stopped'), 'settled row, no engineering failure')
    goal = vector(row['goal']); root = vector(row['root_state'])
    require(exact(row['goal'], row['plan']['native_goal']), 'literal native goal')
    n = row['own_returned']
    require(type(n) is int and 0 <= n <= 30 and row['own_attempted'] == n and
            len(row['responses']) == n, 'contiguous actual returns')
    require(row['status'] != 'horizon_completed' or n == 30, 'full30 is thirty')
    for step, response in enumerate(row['responses'], 1):
        require(response['step'] == step and (step == n or not (response['terminated'] or response['truncated'])),
                'no response after native stop')
        vector(response['state'])
    frames = row['checkpoints']
    steps = [f['step'] for f in frames]
    require(steps == [x for x in CP if x <= n] and row['missing_checkpoints'] == [x for x in CP if x > n],
            'complete reached checkpoint roster, missing stays null')
    by_step = {f['step']: f for f in frames}
    result = []
    for cp in CP:
        frame = by_step.get(cp)
        state = None if frame is None else frame['state']
        recorded = None if cp == 0 or frame is None else row['responses'][cp-1]['native_success']
        if frame is not None:
            vector(state)
            require(exact(state, frame['snapshot']['state']) and exact(state, root if cp == 0 else
                    row['responses'][cp-1]['state']), 'checkpoint/snapshot/same-time response exact')
            require(frame['snapshot']['state_float64_bits'] == struct.pack('<7d', *state).hex(), 'native state float64 bits')
        m = None if state is None else metrics(state, goal)
        result.append(dict(base=base, root=base//2, arm=ARMS[base%2], policy=policy,
            control_step=cp, absolute_step=10+cp, state7=deepcopy(state), goal7=goal, metrics=m,
            recorded_transition_native_success=recorded,
            success_agreement=None if recorded is None else recorded is m['native_success'],
            missing_reason='checkpoint_not_reached' if frame is None else None))
    return result


def pair(left, right):
    a, b = left['metrics'], right['metrics']
    ok = a is not None and b is not None
    scalar_keys = ('agent_xy_squared', 'object_xy_squared', 'agent_xy_error', 'object_xy_error',
                   'joint_xy_squared', 'raw_angle_squared', 'velocity_error',
                   'velocity_squared', 'native7_squared', 'joint_xy_error', 'native7_distance',
                   'native7_reward', 'native_angle_term', 'wrapped_angle_error')
    delta = None if not ok else {k:b[k]-a[k] for k in scalar_keys}
    axis = None if not ok else [y-x for x, y in zip(a['axis_squared'], b['axis_squared'])]
    sign = lambda v: 'negative' if v < 0 else ('positive' if v > 0 else 'zero')
    return dict(base=left['base'], root=left['root'], arm=left['arm'], control_step=left['control_step'],
        absolute_step=left['absolute_step'], pair=left['policy']+'-'+right['policy'],
        delta_direction=right['policy']+' minus '+left['policy'], comparable=ok, delta=delta,
        axis_squared_delta=axis, axis_delta_sum=None if axis is None else math.fsum(axis),
        state_axis_delta=None if not ok else [y-x for x, y in zip(left['state7'], right['state7'])],
        object_pose_exact=None if not ok else exact(a['object_pose'], b['object_pose']),
        total_minus_axis_roundoff=None if axis is None else delta['native7_squared']-math.fsum(axis),
        native_success_delta=None if not ok else int(b['native_success'])-int(a['native_success']),
        joint_xy_sign=None if not ok else sign(delta['joint_xy_squared']),
        native7_sign=None if not ok else sign(delta['native7_squared']))


def summarize(original, jobs, original_path, original_sha):
    authority(original, True)
    require(len(jobs) == 4, 'all four jobs, not effect-selected subset')
    panel = []; rows = []
    for base, job in enumerate(jobs):
        authority(job, False, base)
        require(job['bindings'].get(original_path) == original_sha, 'explicit original receipt SHA ancestry')
        old = original['rows'][base]
        require(old['index'] == base, 'original ordered bases')
        trio = [old] + job['rows']
        for j, (policy, row) in enumerate(zip(POLICIES, trio, strict=True)):
            if j:
                require((row['base_index'], row['policy'], row['index']) == (base, policy, j-1), 'fixed I/F ordering')
                for key in ('task_id', 'root_sha256', 'selection_digest', 'goal_pixel_sha256', 'goal', 'root_state'):
                    require(exact(old[key], row[key]), 'same O/I/F '+key)
                require(exact(old['plan']['planner_plan'], row['plan']['planner_plan']), 'same original full model history/goal')
                require(exact(old['plan']['commands'][:15], row['plan']['commands'][:15]), 'fixed pre-feedback raw15')
            points = row_panel(row, base, policy)
            if j:
                original_points = panel[base*21:base*21+7]
                require(all(exact(x['state7'], y['state7']) for x, y in zip(original_points, points, strict=True) if
                            x['control_step'] <= 15), 'same physical prefix checkpoint states')
            panel.extend(points)
            rows.append(dict(base=base, policy=policy, status=row['status'], actual_controls=row['own_returned'],
                             task_id=row['task_id'], source_row_index=row['index']))
    pairs = []
    for base in range(4):
        for a, b in ((0, 1), (0, 2), (1, 2)):
            for k in range(7):
                pairs.append(pair(panel[base*21+a*7+k], panel[base*21+b*7+k]))
    require(len(panel) == len(pairs) == 84 and len(rows) == 12, 'complete fixed denominators')
    signs = {}
    for pair_name in ('O-I', 'O-F', 'I-F'):
        subset = [p for p in pairs if p['pair'] == pair_name]
        signs[pair_name] = {metric:{key:sum(p[metric] == key for p in subset)
                                   for key in ('negative', 'positive', 'zero', None)}
                            for metric in ('joint_xy_sign', 'native7_sign')}
    # JSON object keys cannot preserve None as None: use explicit named unknown bucket.
    for group in signs.values():
        for bucket in group.values():
            bucket['unknown'] = bucket.pop(None)
    axes = set(original['counts'])
    require(all(set(j['counts']) == axes for j in jobs), 'same resource names, no missing-axis zero fill')
    new_counts = {k:{v:sum(j['counts'][k][v] for j in jobs) for v in ('attempted', 'completed')} for k in sorted(axes)}
    return dict(denominator=dict(bases=4, slots=12, positions=84, pairs=84, known_roots=2, fresh_roots=0),
        axes=list(AXES), rows=rows, panel=panel, pairs=pairs, all_pair_sign_counts=signs,
        resources=dict(original_O=deepcopy(original['counts']), new_I_F=new_counts,
                       inherited_receipt_PNG=original['saved_PNG_verified'],
                       new_receipt_PNG=sum(j['saved_PNG_verified'] for j in jobs),
                       job_wall_seconds=[j['full_stage_wall_seconds'] for j in jobs],
                       wall_scope='four job inclusive walls; not orchestration wall or equal compute'),
        trust_boundary='five SHA-bound complete MAIN own receipts; no producer metrics used; no fullstrict rerun',
        success_semantics='env.eval_state: jointXY<20 and min(abs(dtheta),2pi-abs(dtheta))<pi/9; fsum norms; wrapped diagnostic separate',
        interpretation='exploratory fixed full panel; axis arithmetic is not causal attribution or scientific release',
        **{k:False for k in FLAGS})


def build(original, original_sha, jobs, job_shas):
    require(len(jobs) == len(job_shas) == 4, 'four explicit receipt paths and hashes')
    paths = [str(absolute(p)) for p in [original, *jobs]]
    require(len(set(paths)) == 5, 'distinct five receipt files')
    receipts = [read(p, h) for p, h in zip(paths, [original_sha, *job_shas], strict=True)]
    env = str(ROOT/ENV_REL)
    require(all(r['bindings'].get(env) == ENV_SHA for r in receipts) and sha(env) == ENV_SHA,
            'original eval_state source exact binding, read not import')
    result = summarize(receipts[0], receipts[1:], paths[0], original_sha)
    require(all(sha(p) == h for p, h in zip(paths, [original_sha, *job_shas], strict=True)),
            'five receipt bytes stable after independent arithmetic')
    own = [Path(__file__).resolve(), ROOT/'tests'/('test_'+NAME+'.py'), ROOT/'innerdocs'/(NAME+'.md')]
    return dict(schema=NAME+'.summary', arguments=dict(original=paths[0], original_sha=original_sha,
        jobs=paths[1:], job_shas=list(job_shas)), summary=result,
        own_sha256={str(p):sha(p) for p in own}, native_semantics_source=dict(path=env, sha256=ENV_SHA,
        function='PushT.eval_state', lines=[347, 355]),
        new_neural_calls=0, new_physics_calls=0, new_fullstrict_calls=0)


def write(output, value):
    p = absolute(output); b = encode(value)
    require(p.name.startswith(NAME+'_') and len(b) <= LIMIT, 'exclusive bounded output')
    with p.open('xb') as f:
        require(f.write(b) == len(b), 'complete write'); f.flush(); os.fsync(f.fileno())
    fd = os.open(p.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return dict(path=str(p), sha256=sha(p))


def verify(summary, summary_sha):
    saved = read(summary, summary_sha)
    require(exact(saved, build(**saved['arguments'])), 'whole summary independently rebuilt')
    return dict(verified=True, sha256=summary_sha, new_fullstrict_calls=0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    a = sub.add_parser('analyze')
    a.add_argument('--original', default=str(ROOT/'results'/(OLD+'_consume_MAIN01.json')))
    a.add_argument('--original-sha', default=OLD_SHA)
    a.add_argument('--jobs', nargs=4, default=[str(ROOT/'results'/(NEW+f'_consume_J{i}_MAIN01.json')) for i in range(4)])
    a.add_argument('--job-shas', nargs=4, default=JOB_SHAS)
    a.add_argument('--output', required=True)
    v = sub.add_parser('verify'); v.add_argument('--summary', required=True); v.add_argument('--summary-sha', required=True)
    args = ap.parse_args(argv)
    if args.command == 'verify':
        result = verify(args.summary, args.summary_sha)
    else:
        result = write(args.output, build(args.original, args.original_sha, args.jobs, args.job_shas))
    print(encode(result).decode(), end='')


if __name__ == '__main__':
    main()
