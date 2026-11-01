"""Independent saved full30 native arithmetic; no producer/model/physics import."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path

NAME = 'main_p2_full30_same_time_summary_20261029_v1'
SOURCE = 'agent_p2_two_program_full30_physical_20261029_v1'
CP = (0, 5, 10, 15, 20, 25, 30)
ARMS = ('two_program_transfer', 'joint_two_round')
FLAGS = ('P1_admission', 'P2_admission', 'P2_mechanism', 'scientific_admission',
         'freshness', 'equal_FLOPs', 'equal_total_compute', 'physical_release')
LIMIT = 16 * 1024**2


def require(ok, label):
    if not ok:
        raise ValueError(label)


def encode(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode() + b'\n'


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
    def pairs(items):
        out = {}
        for k, v in items:
            require(k not in out, 'duplicate key')
            out[k] = v
        return out
    obj = json.loads(b, object_pairs_hook=pairs)
    encode(obj)
    return obj


def scalars(state, goal):
    require(type(state) is list and type(goal) is list and len(state) == len(goal) == 7,
            'native seven-dimensional state/goal')
    require(all(type(v) in (float, int) and math.isfinite(v) for v in state + goal), 'finite native coordinates')
    d = [float(s) - float(g) for s, g in zip(state, goal)]
    q = [v*v for v in d]
    xy = math.fsum(q[:4]); angle = min(abs(d[4]), 2*math.pi-abs(d[4]))
    return dict(agent_xy_squared=math.fsum(q[:2]), object_xy_squared=math.fsum(q[2:4]),
                joint_xy_squared=xy, raw_angle_squared=q[4], velocity_squared=math.fsum(q[5:]),
                native7_squared=math.fsum(q), joint_xy_error=math.sqrt(xy),
                object_xy_error=math.sqrt(math.fsum(q[2:4])), native7_distance=math.sqrt(math.fsum(q)),
                native_angle_term=angle, wrapped_angle_error=abs(math.atan2(math.sin(d[4]), math.cos(d[4]))),
                native_success=bool(math.sqrt(xy) < 20 and angle < math.pi/9))


def summarize(p, r, c, paths, hashes):
    require(p['schema'] == SOURCE+'.prepare' and r['schema'] == SOURCE+'.report'
            and c['schema'] == SOURCE+'.consume', 'own full30 schemas')
    require(r['status'] == c['status'] == 'full30_four_native_recorded_NOT_admitted'
            and r['errors'] == c['errors'] == [] and r['fixed_denominator'] == c['fixed_denominator'] == 4
            and r['known_roots'] == c['known_roots'] == 2 and r['fresh_roots'] == c['fresh_roots'] == 0,
            'complete four branches, two known roots')
    require(all(r[k] is False and c[k] is False for k in FLAGS), 'scientific gates unchanged')
    require(r['execution_binding']['prepared'] == paths[0] and r['execution_binding']['prepare_sha256'] == hashes[0]
            and c['physical_report_sha256'] == hashes[1] and c['prepare_sha256'] == hashes[0]
            and c['bindings'][paths[0]] == hashes[0] and c['bindings'][paths[1]] == hashes[1], 'external authority joins')
    require(len(p['plans']) == len(r['rows']) == len(c['rows']) == 4
            and encode(c['rows']) == encode(r['rows']) and encode(c['plans']) == encode(p['plans'])
            and encode(r['counts']) == encode(c['counts']), 'all raw plans, rows and resources inherited exactly')
    panel = []; terminals = []
    for i, (plan, row) in enumerate(zip(p['plans'], r['rows'], strict=True)):
        require((row['index'], row['root_slot'], row['condition'], row['arm']) == (i, i//2, 'off', ARMS[i%2])
                and row['plan'] == plan and row['status'] in ('horizon_completed', 'native_stopped')
                and row['first_error'] is None and row['secondary_errors'] == [], 'fixed four identities')
        require(row['goal'] == plan['native_goal'] == p['source_roots'][i//2]['goal'], 'original physical goal')
        n = row['own_returned']
        require(type(n) is int and 0 <= n <= 30 and row['own_attempted'] == n
                and len(row['responses']) == n, 'actual contiguous unpadded length')
        require(row['status'] != 'horizon_completed' or n == 30, 'full horizon is thirty')
        for j, response in enumerate(row['responses'], 1):
            require(response['step'] == j and (j == n or not (response['terminated'] or response['truncated'])), 'no post-stop response')
        checkpoints = {f['step']: f for f in row['checkpoints']}
        require(len(checkpoints) == len(row['checkpoints']) and list(checkpoints) == [b for b in CP if b <= n]
                and row['missing_checkpoints'] == [b for b in CP if b > n], 'fixed checkpoint incidence')
        for b in CP:
            f = checkpoints.get(b)
            state = None if f is None else f['state']
            if f is not None:
                require(state == f['snapshot']['state'] and state == (row['root_state'] if b == 0 else row['responses'][b-1]['state']), 'same-time state/snapshot/response')
            values = None if state is None else scalars(state, row['goal'])
            panel.append(dict(slot=i, root=i//2, arm=ARMS[i%2], control_step=b, absolute_step=10+b,
                              checkpoint=deepcopy(f), metrics=values,
                              missing_reason='checkpoint_not_reached' if f is None else None))
        terminals.append(dict(slot=i, actual_control_step=n, metrics=None if row['final_state'] is None else scalars(row['final_state'], row['goal'])))
    pairs = []
    for root in (0, 1):
        left, right = r['rows'][2*root:2*root+2]
        require(left['goal'] == right['goal'] and left['root_state'] == right['root_state'], 'same root and goal')
        require(left['plan']['planner_plan']['root_context'] == right['plan']['planner_plan']['root_context'], 'same full model starting context')
        for j, b in enumerate(CP):
            a, z = panel[(2*root)*7+j]['metrics'], panel[(2*root+1)*7+j]['metrics']
            delta = None if a is None or z is None else {k: z[k]-a[k] for k in a if k != 'native_success'}
            parts = None if delta is None else {k: delta[k] for k in ('joint_xy_squared', 'raw_angle_squared', 'velocity_squared')}
            pairs.append(dict(root=root, control_step=b, direction='joint_two_round minus two_program_transfer',
                              comparable=delta is not None, delta=delta, squared_components=parts,
                              decomposition_roundoff=None if parts is None else delta['native7_squared']-math.fsum(parts.values()),
                              native_success_delta=None if delta is None else int(z['native_success'])-int(a['native_success'])))
    return dict(denominator=dict(branches=4,known_roots=2,fresh_roots=0,checkpoint_positions=28,pairs=14),
                panel=panel,pairs=pairs,terminals=terminals,counts=deepcopy(r['counts']),
                interpretation='independent math.fsum saved arithmetic; imported complete native receipt, no fullstrict rerun; unequal generation support/compute',
                all_four_full30=all(x['own_returned'] == 30 for x in r['rows']), **{k:False for k in FLAGS})


def analyze(prepared, prepare_sha, report, report_sha, receipt, receipt_sha):
    paths = [str(absolute(x)) for x in (prepared, report, receipt)]
    hashes = [prepare_sha, report_sha, receipt_sha]
    p, r, c = [read(x, h) for x, h in zip(paths, hashes)]
    root = Path(__file__).resolve().parents[2]
    own = [Path(__file__).resolve(), root/'tests'/('test_'+NAME+'.py')]
    return dict(schema=NAME+'.summary', arguments=dict(prepared=paths[0],prepare_sha=hashes[0],report=paths[1],report_sha=hashes[1],receipt=paths[2],receipt_sha=hashes[2]),
                summary=summarize(p,r,c,paths,hashes), own_sha256={str(x):sha(x) for x in own},
                new_neural_calls=0,new_physics_calls=0,new_fullstrict_calls=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=('analyze','verify'))
    for k in ('prepared','prepare-sha','report','report-sha','receipt','receipt-sha','output','summary','summary-sha'):
        ap.add_argument('--'+k)
    a = ap.parse_args()
    if a.command == 'verify':
        saved = read(a.summary, a.summary_sha)
        require(encode(saved) == encode(analyze(**saved['arguments'])), 'whole deterministic recomputation')
        print(json.dumps(dict(verified=True,sha256=a.summary_sha))); return
    result = analyze(a.prepared,a.prepare_sha,a.report,a.report_sha,a.receipt,a.receipt_sha)
    path = absolute(a.output); payload = encode(result)
    require(path.name.startswith(NAME+'_') and not path.exists() and len(payload) <= LIMIT, 'exclusive bounded new summary')
    with path.open('xb') as f:
        require(f.write(payload) == len(payload), 'complete write'); f.flush(); os.fsync(f.fileno())
    fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
    print(json.dumps(dict(path=str(path),sha256=sha(path))))


if __name__ == '__main__':
    main()
