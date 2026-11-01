"""CPU-only, anchored descriptive analysis of the six saved HOST branches."""
import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import struct

from agent_p3_c3_k17_physical_transport_20261029_v1 import read_journal

COMPLETION = '41ed9feb5fdfaaa1edbb9edf6b2dc1f28df130826b5129dba82c3173bceb2818'
RECEIPT = '20caa3ba066ca86299c59192505aba5a13dce7a6037af207df87c8281e01a155'
ORDER = [(3, 0, 'fixedH1'), (4, 0, 'staticH3_progress'), (5, 0, 'adaptive'),
         (9, 1, 'fixedH1'), (10, 1, 'staticH3_progress'), (11, 1, 'adaptive')]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_bytes(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def analyze(bundle, receipt_path, completion_sha=COMPLETION, receipt_sha=RECEIPT):
    bundle, receipt_path = Path(bundle), Path(receipt_path)
    require(sha(bundle / 'completion.json') == completion_sha, 'completion anchor')
    require(sha(receipt_path) == receipt_sha, 'receipt anchor')
    completion, receipt = load(bundle / 'completion.json'), load(receipt_path)
    require(sha(bundle / 'manifest.json') == completion['manifest_sha256'], 'manifest anchor')
    manifest = load(bundle / 'manifest.json')
    for name, expected in manifest['files'].items():
        path = Path(name)
        require(not path.is_absolute() and '..' not in path.parts and (bundle / path).is_file(), 'manifest path')
        require(sha(bundle / path) == expected, 'manifest file: ' + name)
    for name in ('native/report.json', 'native/journal.jsonl'):
        require(name in manifest['files'], 'missing manifest file')
    journal_path = bundle / 'native/journal.jsonl'
    journal_sha = sha(journal_path)
    require(receipt['audit']['journal_sha256'] == journal_sha == receipt['guard_frames_audit']['journal_sha256'], 'receipt journal binding')
    require(receipt['status'] == 'saved_engineering_only' and not receipt['scientific_pass'], 'receipt status')
    events = read_journal(journal_path)
    report = load(bundle / 'native/report.json')
    rows = report['outer']['rows']
    require(len(rows) == 6 and [(r['slot'], r['root_slot'], r['mode']) for r in rows] == ORDER, 'slot order')
    goals, transitions = {}, {}
    for event in events:
        p = event['payload']
        if event['kind'] == 'goal_attached':
            b = p['branch']; require(b not in goals, 'duplicate goal'); goals[b] = p
        if event['kind'] == 'returned_transition' and p['phase'] == 'plan':
            key = (p['branch'], p['step']); require(key not in transitions, 'duplicate transition'); transitions[key] = p
    require(set(goals) == set(range(6)) and set(transitions) == set(itertools.product(range(6), range(1, 6))), 'missing goal or step')
    counts = [Counter() for _ in rows]
    for receipt_item in report['outer']['receipts']:
        b = receipt_item['branch']; require(type(b) is int and 0 <= b < 6 and receipt_item['status'] == 'returned', 'bad resource receipt')
        counts[b][receipt_item['resource']] += 1
    total = sum(counts, Counter())
    require(all(total[k] == v['completed'] == v['attempted'] and v['failed'] == 0 for k, v in receipt['audit']['counts'].items()) and set(total) == set(receipt['audit']['counts']), 'resource totals')
    branches = []
    for b, row in enumerate(rows):
        require(row['index'] == b and row['status'] == 'horizon_completed', 'branch status')
        goal = goals[b]
        require((goal['slot'], goal['root_slot'], goal['mode']) == ORDER[b], 'goal identity')
        plan_frames = [x for x in row['trajectory'] if x['phase'] == 'plan']
        frames = {x['step']: x for x in plan_frames}
        require(len(plan_frames) == 6 and len(frames) == 6 and set(frames) == set(range(6)), 'plan frame coverage')
        steps = []
        for t in range(1, 6):
            p = transitions[b, t]
            require((p['slot'], p['root_slot'], p['mode']) == ORDER[b] and p['goal_state'] == goal['goal'], 'transition identity/goal')
            require(p['requested_bits'] == p['actual_bits'] and p['actual_dtype'] == '<f4', 'action mismatch')
            action = list(struct.unpack('<2f', bytes.fromhex(p['actual_bits'])))
            require(all(math.isfinite(x) for x in action + p['state'] + [p['reward']]), 'nonfinite')
            require(p['state'] == frames[t]['state'] and p['snapshot'] == frames[t]['snapshot'], 'transition/frame mismatch')
            steps.append(dict(step=t, action_bits=p['actual_bits'], action=action, state=p['state'], reward=p['reward'], terminated=p['terminated'], truncated=p['truncated']))
        require(goal['snapshot'] == frames[0]['snapshot'], 'initial snapshot mismatch')
        branches.append(dict(index=b, slot=row['slot'], root=row['root_slot'], mode=row['mode'], goal=goal['goal'], goal_pose=goal['goal_pose'], goal_rgb_sha256=goal['goal_rgb_sha256'], initial_snapshot=frames[0]['snapshot'], resources=dict(sorted(counts[b].items())), steps=steps))
    pairs = []
    for a, b in itertools.combinations(branches, 2):
        if a['root'] != b['root']: continue
        require(a['goal'] == b['goal'] and a['goal_pose'] == b['goal_pose'] and a['goal_rgb_sha256'] == b['goal_rgb_sha256'], 'same-root goal mismatch')
        require(a['initial_snapshot'] == b['initial_snapshot'], 'same-root initial snapshot mismatch')
        same_prefix = 0
        for x, y in zip(a['steps'], b['steps']):
            if x['action_bits'] != y['action_bits']: break
            same_prefix += 1
        diffs = []
        for x, y in zip(a['steps'], b['steps']):
            ad = [v-u for u, v in zip(x['action'], y['action'])]
            diffs.append(dict(step=x['step'], action_bit_equal=x['action_bits']==y['action_bits'], action_b_minus_a=ad, action_l2=math.hypot(*ad), state_b_minus_a=[v-u for u, v in zip(x['state'], y['state'])], reward_b_minus_a=y['reward']-x['reward']))
        pairs.append(dict(root=a['root'], a=a['mode'], b=b['mode'], slots=[a['slot'],b['slot']], initial_snapshot_equal=True, goal_equal=True, action_bit_equal_prefix=same_prefix, steps=diffs))
    return dict(schema='p3_k17_host_pairs_codex.v1', anchors=dict(completion_sha256=completion_sha, receipt_sha256=receipt_sha, manifest_sha256=completion['manifest_sha256'], journal_sha256=journal_sha), scope='descriptive, within-root, same plan step; differences are b minus a', branches=branches, pairs=pairs, resource_totals=dict(sorted(total.items())), outer_wall_seconds=report['outer']['wall_seconds'], environment_evidence=report['environment_evidence'], limitations=['not equal total budgets', 'no independent generalization', 'no trusted native provenance certification', 'no scientific pass', 'planning/neural compute and per-method wall time unavailable'])


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bundle'); parser.add_argument('receipt'); parser.add_argument('output')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    data = encoded(analyze(args.bundle, args.receipt))
    path = Path(args.output)
    if args.verify:
        require(path.read_bytes() == data, 'saved result differs from recomputation')
    else:
        with path.open('xb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
    print(hashlib.sha256(data).hexdigest())


if __name__ == '__main__':
    main()
