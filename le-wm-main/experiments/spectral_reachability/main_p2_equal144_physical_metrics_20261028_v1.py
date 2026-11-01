"""MAIN saved-only strict receipt and independent native-task arithmetic.

No environment/model imports or executions; four fixed branches, two KNOWN roots.
Warmup returns precede goal binding and are NOT checked against the new goal.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import agent_p2_equal144_physical_handoff_20261028_v1 as native

NAME = 'main_p2_equal144_physical_metrics_20261028_v1'
REPORT_SHA = '50240e6f5e9bca90a5d8734cb5c41a5e06d61337b20dcd3d7fb9693fea45ebcb'
DIRECTORY = native.ROOT/'results/agent_p2_equal144_physical_handoff_20261028_v1_physics_MAIN01'


def metrics(state, goal):
    a, g = np.asarray(state, dtype=np.float64), np.asarray(goal, dtype=np.float64)
    if a.shape != (7,) or g.shape != (7,) or not np.isfinite([a, g]).all():
        raise ValueError('finite native seven-state vectors required')
    d = a-g
    angle = min(abs(float(d[4])), 2*math.pi-abs(float(d[4])))
    joint = float(np.linalg.norm(d[:4]))
    return dict(native7_distance=float(np.linalg.norm(d)),
                native7_reward=-float(np.linalg.norm(d)),
                joint_xy_error=joint, agent_xy_error=float(np.linalg.norm(d[:2])),
                object_xy_error=float(np.linalg.norm(d[2:4])),
                velocity_error=float(np.linalg.norm(d[5:])),
                raw_angle_difference=float(d[4]), native_angle_term=angle,
                native_success=bool(joint < 20 and angle < math.pi/9),
                native7_squared_components=dict(joint_xy=float(d[:4]@d[:4]),
                    raw_angle=float(d[4]**2), velocity=float(d[5:]@d[5:])))


def transitions(events, rows):
    checked = []
    for branch, row in enumerate(rows):
        es = [e for e in events if e['kind']=='returned_transition' and
              e.get('branch')==branch and e.get('phase')=='plan']
        if [e['step'] for e in es] != list(range(1, row['own_returned']+1)):
            raise ValueError('all actual plan returns, contiguous with no padding')
        for e in es:
            m = metrics(e['state'], e['returned_goal_state'])
            if (e['returned_goal_state'] != row['goal'] or
                e['command'] != row['selected_plan']['commands'][e['step']-1] or
                e['reward'] != m['native7_reward'] or e['terminated'] != m['native_success'] or
                type(e['truncated']) is not bool):
                raise ValueError('actual goal/raw command/reward/termination mismatch')
            checked.append(dict(branch=branch, step=e['step'], metrics=m,
                                truncated=e['truncated']))
        if es and es[-1]['state'] != row['final_state']:
            raise ValueError('final state must be actual final returned state')
    return checked


def analyze():
    receipt = native.consume(str(DIRECTORY), REPORT_SHA)
    report = receipt['report']
    events = [json.loads(s) for s in (DIRECTORY/'journal.jsonl').read_bytes().splitlines()]
    checked = transitions(events, report['rows'])
    rows = [dict(branch=i, root_slot=r['root_slot'], arm=r['arm'], status=r['status'],
                 metrics=None if r['final_state'] is None else metrics(r['final_state'], r['goal']))
            for i,r in enumerate(report['rows'])]
    pairs = []
    for root in (0,1):
        beam, spectral = rows[2*root:2*root+2]
        if (beam['root_slot'],spectral['root_slot']) != (root,root):
            raise ValueError('original root-pair order')
        delta = None if beam['metrics'] is None or spectral['metrics'] is None else {
            k: spectral['metrics'][k]-beam['metrics'][k]
            for k in ('native7_distance','joint_xy_error','agent_xy_error','object_xy_error','velocity_error')}
        pairs.append(dict(root_slot=root, spectral_minus_beam=delta))
    return dict(schema=NAME+'.analysis', native_strict_receipt=receipt,
                independent_plan_return_checks=checked, rows=rows, pairs=pairs,
                known_roots=2, new_physical_executions=4, full_branch_denominator=4,
                warmup_scope='not recomputed against post-warmup task goal',
                resources='144 predictor samples/API per method/root; AD, VJP, SVD and solve differ',
                neural_calls=0, new_physics=0, P1_admission=False, P2_admission=False,
                equal_total_compute=False, independent_confirmation=False, scientific_admission=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('analyze','verify'))
    p.add_argument('--output', required=True)
    p.add_argument('--sha256')
    a = p.parse_args(); path = Path(a.output).absolute()
    result = analyze(); payload = native.raw(result)
    if a.command == 'analyze':
        with path.open('xb') as f: f.write(payload)
    elif hashlib.sha256(path.read_bytes()).hexdigest()!=a.sha256 or path.read_bytes()!=payload:
        raise ValueError('entire saved receipt and independently recomputed analysis mismatch')
    print(json.dumps(dict(status='saved_physical_metrics_verified_not_admitted',
        output=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        checked_plan_returns=len(result['independent_plan_return_checks']),
        pairs=result['pairs'], scientific_admission=False),sort_keys=True))


if __name__=='__main__': main()
