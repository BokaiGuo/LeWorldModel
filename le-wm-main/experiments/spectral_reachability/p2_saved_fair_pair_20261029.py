"""Pinned saved-only P2 physical/resource joins; no model or simulator imports."""
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAME = 'p2_saved_fair_pair_20261029'
SOURCES = {
    'physical': ('agent_p2_intersection_physical_20261029_v2_consume_MAIN01.json', 'e96462847db675dc0b0c5376067df3654edd40805d6e374ffa3f8f1d5224e6c4'),
    'resources': ('agent_p2_intersection_v2_saved_analysis_20261029_v1_MAIN01.json', '5cc0c8e0bf370ba24a366555b2cbc7c29ef275f46206b4906f904cc64549a1b9'),
}
ORDER = ((0, 'joint'), (0, 'composition'), (0, 'grasp'),
         (1, 'grasp'), (1, 'composition'), (1, 'joint'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def read(path, expected):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size < 128*1024**2, 'bounded regular input')
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == expected, 'source SHA mismatch')
    def pairs(items):
        out = {}
        for k, v in items:
            require(k not in out, 'duplicate JSON key')
            out[k] = v
        return out
    def bad(value):
        raise ValueError('nonfinite JSON: ' + value)
    return json.loads(data, object_pairs_hook=pairs, parse_constant=bad)


def analyze(physical, resources):
    require(physical['fixed_denominator'] == resources['fixed_denominator'] == 6, 'six slots required')
    require(physical['status'] == 'intersection6_native_recorded_not_admitted' and not physical['errors'], 'physical receipt status')
    require(len(physical['rows']) == len(resources['rows']) == 6, 'six rows required')
    by = {}
    for i, (p, r) in enumerate(zip(physical['rows'], resources['rows'])):
        root, arm = ORDER[i]
        require((p['index'], p['root_slot'], p['arm']) == (i, root, arm), 'physical identity')
        require((r['slot'], r['root'], r['arm']) == (i, root, arm), 'resource identity')
        plan = p['plan']
        require(p['task_id'] == plan['task_id'] == r['task_id'], 'task binding')
        require(plan['raw_steps'] == len(plan['raw15']) == 15, 'planned raw action horizon')
        require(p['initial_exact'] is True and p['own_returned'] <= 15, 'physical root and observed horizon')
        require(r['resource_axes27']['FLOPs'] is None, 'unknown FLOPs')
        cps = {x['step']: x for x in p['checkpoints']}
        require(len(cps) == len(p['checkpoints']) and set(cps) == {s for s in (0, 5, 10, 15) if s <= p['own_returned']}, 'checkpoint coverage')
        require(cps[0]['state'] == p['root_state'], 'root checkpoint')
        by[root, arm] = (p, r, cps)
    results = []
    for root in (0, 1):
        for left, right in itertools.combinations(('joint', 'composition', 'grasp'), 2):
            a, ar, ac = by[root, left]
            b, br, bc = by[root, right]
            identity = all(a[k] == b[k] for k in ('task_id', 'root_state', 'goal', 'goal_pixel_sha256'))
            identity &= all(a['plan'][k] == b['plan'][k] for k in ('task_sha256', 'warm_raw10'))
            identity &= ar['goal_sha256'] == br['goal_sha256']
            for step in (5, 10, 15):
                reasons = []
                if not identity:
                    reasons.append('root_goal_or_history_mismatch')
                if step not in ac or step not in bc:
                    reasons.append('checkpoint_not_reached')
                delta = None
                if not reasons:
                    goal = a['goal']
                    require(len(goal) == len(ac[step]['state']) == len(bc[step]['state']) == 7, 'native7 shape')
                    def distance(state):
                        require(all(type(x) in (int, float) and math.isfinite(x) for x in state + goal), 'finite native7')
                        return math.sqrt(math.fsum((x-y)**2 for x, y in zip(state, goal)))
                    delta = distance(bc[step]['state']) - distance(ac[step]['state'])
                results.append(dict(root=root, left=left, right=right, step=step,
                    comparable=not reasons, missing_reasons=reasons, native7_right_minus_left=delta,
                    left_returned=a['own_returned'], right_returned=b['own_returned'],
                    resources_left=ar['resource_axes27'], resources_right=br['resource_axes27'],
                    side_left=ar['side_axes'], side_right=br['side_axes'],
                    wall_left=ar['elapsed_seconds'], wall_right=br['elapsed_seconds']))
    return dict(schema=NAME+'.analysis', source_sha256={k:v[1] for k,v in SOURCES.items()},
        inherited_strict_receipts=True, upstream_fullstrict_rerun=False, known_roots=2,
        planned_horizon_raw_steps=15, same_action_values_required=False, pairs=results,
        comparable=sum(x['comparable'] for x in results), total_pairs=len(results),
        no_composition_sra=dict(comparable=False, reason='inspected historical H2 records have different saved native roots and goals and 10-step endpoints versus H15; no authenticated same-task join'),
        fair_total_compute=False, equal_flops=None, winner=None,
        interpretation='same task and time descriptive native7 differences only; resource axes are not interchangeable; no gate decision')


def build():
    loaded = {k: read(ROOT/'results'/p, h) for k, (p,h) in SOURCES.items()}
    result = analyze(loaded['physical'], loaded['resources'])
    for p,h in SOURCES.values():
        read(ROOT/'results'/p,h)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('build', 'verify'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--sha256')
    args = parser.parse_args()
    require(args.output.name.startswith(NAME+'_'), 'exclusive output name')
    data = encode(build())
    if args.command == 'build':
        with args.output.open('xb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        digest = hashlib.sha256(data).hexdigest()
    else:
        require(args.sha256 and hashlib.sha256(args.output.read_bytes()).hexdigest() == args.sha256, 'output SHA')
        require(args.output.read_bytes() == data, 'whole rebuild mismatch')
        digest = args.sha256
    print(json.dumps(dict(sha256=digest, verified=args.command == 'verify', output=str(args.output))))


if __name__ == '__main__':
    main()
