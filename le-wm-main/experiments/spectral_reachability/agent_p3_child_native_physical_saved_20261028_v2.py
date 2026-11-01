"""Saved-only exact native census repair. Frozen execution and gates unchanged.

Copies the complete frozen v1 validator with ONLY two census literal changes.
No monkeypatch, native factory, neural loader, or physical execution entry.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import agent_p3_child_native_physical_20261028_v1 as base

NAME = 'agent_p3_child_native_physical_saved_20261028_v2'
ROOT = base.ROOT
require, sha, canonical = base.require, base.sha, base.canonical
digest, READ, FLAGS = base.digest, base.READ, dict(base.FLAGS)
ledger, new_row, png = base.ledger, base.new_row, base.png
same_frame, raw, metric, old, compiler = base.same_frame, base.raw, base.metric, base.old, base.compiler
exact = base.exact
PINS = dict(zip(map(str, base.own_files()), (
    '7f1d0565d961d9b1e33dc6c601ca4f4bd62ea1969dc2411126beed7b99704955',
    '6bf6375895de8656cb04ba771b74dce63f2babf6ea508f0d04da18af1c37e2d6',
    'd10a3374407d83481c7085814b5409e7286c48e3d2bf166e2a067f13ecaaa068'), strict=True))
ACTUAL_OUTPUT = ROOT/'results/agent_p3_child_native_physical_20261028_v1_MAIN01'
ACTUAL_REPORT_SHA = '56c3e97208597bdc8424dcfb263c8840683c6b3413abeb79d01fa46762dc5df6'
ACTUAL_PREPARE_SHA = 'e2a8752f351d47d780a49774c07b9bead1a33c75690ca286ef9bff4727681f95'


def own_files():
    return [Path(__file__).resolve(), ROOT/f'tests/{NAME}_test.py', ROOT/f'innerdocs/{NAME}.md']


def pins():
    compiler.contract.check_sources(PINS)
    return dict(PINS)


def validate_saved(p, report, events, output):
    require(report['profile'] == p['profile'] and report['fixed_denominator'] == 48 and report['child_denominator'] == 24 and
            len(report['rows']) == len(p['rows']) == 48, 'whole physical denominator/profile')
    require(all(report[k] is False for k in FLAGS), 'scientific gates remain false')
    counts, pending = ledger(events, p['profile']['caps'])
    require(counts == report['resources'] and pending == report['pending'], 'independent full ledger reconciliation')
    complete = report['status'] == 'recorded' and not report['errors'] and not pending and not report['secondary_errors']
    require(type(report['wall_seconds']) in (int, float) and np.isfinite(report['wall_seconds']) and
            report['wall_seconds'] >= 0 and (not complete or report['wall_seconds'] < 600), 'actual execution wall scope')
    allframes = [e for e in events if e['kind'] == 'fresh_observation']
    require(len({e['receipt']['path'] for e in allframes}) == len(allframes), 'no duplicate/borrowed physical PNG')
    for e in allframes:
        require(e['receipt']['observed_event_index'] == e['index'], 'actual observation event ordinal')
        png(e['receipt'], output)
    for i, (plan, row) in enumerate(zip(p['rows'], report['rows'], strict=True)):
        expected = new_row(i, plan)
        for k in ('ordinal', 'episode_id', 'plan_sha256', 'route'):
            require(row[k] == expected[k], 'fixed physical plan/order identity')
        es = [e for e in events if e.get('episode_id') == row['episode_id']]
        if plan['route'] != 'execute':
            require(row == expected and not es, 'refused/not reached retained unknown, never executed'); continue
        fs = [e['receipt'] for e in es if e['kind'] == 'fresh_observation']
        require(len(fs) == row['PNG'], 'every physical RGB accounted')
        if not complete and row['status'] not in ('returned5', 'native_stopped'): continue
        starts = [e for e in es if e['kind'] == 'branch_started']; finishes = [e for e in es if e['kind'] == 'branch_finished']
        require(len(starts) == len(finishes) == 1 and starts[0]['ordinal'] == i and
                starts[0]['plan_sha256'] == digest(plan) and finishes[0]['row'] == row, 'durable exact branch start/terminal report row')
        require(row['error'] is None and not row['secondary_errors'] and row['initial_exact'] is True and
                row['warmup_returned'] == 10 and row['prefix_returned'] == plan['child']['boundary'], 'complete original recovery')
        b = plan['child']['boundary']; n = row['own_returned']
        require(type(n) is int and 1 <= n <= 5 and len(fs) == 12+b+n, 'native raw5 earlystop no padding')
        phases = [('warmup', j) for j in range(11)] + [('plan', 0)] + [('prefix', j) for j in range(1, b+1)] + [('own', j) for j in range(1, n+1)]
        require([(f['phase'], f['step']) for f in fs] == phases, 'full physical observation time order')
        for f, source in zip(fs[:12+b], plan['source_frames'], strict=True): same_frame(f, source)
        seals = [e for e in es if e['kind'] == 'child_boundary_sealed']
        require(len(seals) == 1 and seals[0]['boundary'] == row['boundary'] == fs[11+b] and
                seals[0]['context_sha256'] == plan['child']['context_sha256'] and
                seals[0]['raw_history10'] == plan['child']['raw_history10'] and
                seals[0]['selected_sha256'] == digest(plan['selected']), 'same actual native child and raw/context seal')
        ds = [e for e in es if e['kind'] == 'authorized_dispatch']; rs = [e for e in es if e['kind'] == 'step_return']
        commands = np.concatenate([raw(plan['child']['executed_prefix'], b), raw(plan['selected']['raw5'], 5)[:n]])
        require(len(ds) == len(rs) == b+n and row['executed_raw'] == commands.tolist(), 'exact prefix+selected raw5 actual denominator')
        for j, (d, r, command) in enumerate(zip(ds, rs, commands, strict=True)):
            phase, stepnum = ('prefix', j+1) if j < b else ('own', j-b+1)
            require((d['phase'], d['step']) == (r['phase'], r['step']) == (phase, stepnum) and
                    d['index'] < r['index'] < fs[12+j]['observed_event_index'], 'dispatch -> return -> current observed state')
            calls = [e for e in es if e['kind'] == 'operation_attempted' and e['resource'] == 'public_step' and
                     d['index'] < e['index'] < r['index']]
            require(len(calls) == 1, 'one literal authorized public API call per selected/prefix ordinal')
            returned = next(e for e in es if e['kind'] == 'operation_completed' and e['operation_id'] == calls[0]['operation_id'])
            require(calls[0]['region'] == ('prefix' if phase == 'prefix' else 'suffix') and returned['index'] < r['index'] and
                    sum(e['kind'] == 'operation_attempted' and e.get('resource') == 'physics_step' and
                        calls[0]['index'] < e['index'] < returned['index'] for e in es) == 10, 'ten actual nested native integrations per public return')
            if phase == 'own': require(fs[11+b]['observed_event_index'] < seals[0]['index'] < d['index'], 'whole native boundary sealed BEFORE selected raw5 dispatch')
            exact(raw([d['command']], 1), command[None], 'literal dispatched selected/raw prefix')
            exact(raw([r['command']], 1), command[None], 'literal returned selected/raw prefix')
            f = fs[12+j]; require(r['state'] == f['state'], 'actual returned native state')
            m = metric(f['goal'], f['state'])
            require(r['reward'] == m['native7_reward'] and r['terminated'] == m['native_success'] and
                    type(r['truncated']) is bool, 'native jointXY+angle termination, native7 reward separately')
            if j < b+n-1: require(not r['terminated'] and not r['truncated'], 'no dispatch after native stop')
        goal = plan['source_frames'][-1]
        for f in fs[12+b:]:
            require(all(f[k] == goal[k] for k in ('goal', 'goal_rgb_sha256', 'goal_pose')), 'own task goal/context not replaced')
        last = rs[-1]
        require(row['last_observation'] == fs[-1] and row['final_metrics'] == metric(fs[-1]['goal'], fs[-1]['state']) and
                row['terminated'] == last['terminated'] and row['truncated'] == last['truncated'] and
                (row['status'] == 'native_stopped') == bool(last['terminated'] or last['truncated']) and
                (n == 5 or row['status'] == 'native_stopped'), 'native earlystop terminal receipt')
        # Every public call includes setup warmup; all actual returns stay on the branch ledger.
        pub = [e for e in es if e['kind'] == 'operation_completed' and e['resource'] == 'public_step']
        require(len(pub) == 10+b+n, 'warmup+executed prefix+own raw5 full actual public cost')
        want = dict(constructor=1, reset=2, set_state=4, goal_render=2, goal_setter=3, goal_bind=1,
                    public_step=10+b+n, physics_step=4+10*(10+b+n), explicit_render=12+b+n)
        for resource, count in want.items():
            for kind in ('operation_attempted', 'operation_completed'):
                require(sum(e['kind'] == kind and e['resource'] == resource for e in es if 'resource' in e) == count,
                        'exact real native branch/reset/nested resource census: '+resource)
        require(sum(e['kind'] == 'cleanup_attempted' for e in es) == sum(e['kind'] == 'cleanup_completed' for e in es) == 1,
                'actual branch cleanup')
        warm = [e for e in es if e['kind'] == 'warmup_return']
        require([e['step'] for e in warm] == list(range(1, 11)) and all(not e['terminated'] and not e['truncated'] for e in warm),
                'all nonzero warmup returns, no reset/padding after stop')
        for e, command in zip(warm, old.kernel.moving.warmup_commands(), strict=True):
            exact(raw([e['command']], 1), command[None], 'original literal nonzero warmup history')
    require(not complete or all(r['status'] in ('returned5', 'native_stopped', compiler.BLOCKED, 'not_reached') for r in report['rows']),
            'complete cannot promote any engineering failure/unstarted branch')
    return dict(schema=NAME+'.consume', status='native_children_verified_not_admitted' if complete else 'partial_preserved_not_verified',
                fixed_denominator=48, child_denominator=24, routes=dict(execute=41, refused=5, not_reached=2),
                independent_roots=2, rows=report['rows'], resources=counts, complete=complete,
                restoration_cost_included=True, new_neural_calls=0, **FLAGS)

def consume(output, expected):
    pins()
    output = base.absolute(output)
    require(output == ACTUAL_OUTPUT and expected == ACTUAL_REPORT_SHA and sha(output/'report.json') == expected,
            'external actual frozen v1 physical report SHA/path')
    r = READ(output/'report.json', base.FINAL)
    require(r['schema'] == base.NAME+'.report' and r['prepare_sha256'] == ACTUAL_PREPARE_SHA,
            'unchanged v1 execution schema/actual prepare SHA')
    # Keep the full original material -> compiler.consume/full strict -> P4.consume
    # closure. No calls to base.consume, whose exact census is the diagnosed bug.
    p = base.verify(r['prepared'], r['prepare_sha256'])
    require(READ(output/'binding.json') == dict(prepared=r['prepared'], prepare_sha256=r['prepare_sha256']),
            'actual execution/prepare binding')
    require(sha(output/'journal.jsonl') == r['journal_sha256'], 'durable actual journal SHA')
    require(sum(f.stat().st_size for f in output.iterdir() if f.is_file()) <= base.CAP and
            all(f.is_file() and not f.is_symlink() for f in output.iterdir()), 'bounded flat saved output')
    result = validate_saved(p, r, base.journal(output/'journal.jsonl', base.JOURNAL_CAP), output)
    compiler.contract.check_sources(p['sources'])
    pins()
    require(sha(output/'report.json') == expected, 'actual report unchanged')
    result.update(report_sha256=expected, prepare_sha256=r['prepare_sha256'],
        repaired_consumer_sources=dict(PINS, **{str(f):sha(f) for f in own_files()}),
        repair=dict(scope='exact per-branch native census ONLY; old cap upper bounds unchanged',
                    reset=2, set_state=4, goal_render=2, goal_setter=3, setup_physics=4,
                    old_logs_edited=False, physics_replayed=False, neural_calls=0))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('consume',))
    parser.add_argument('--output', required=True)
    parser.add_argument('--report-sha256', required=True)
    parser.add_argument('--receipt', required=True)
    args = parser.parse_args()
    path = base.absolute(args.receipt)
    require(path.parent == ROOT/'results' and path.name.startswith(NAME+'_'), 'exclusive own v2 saved receipt')
    result = consume(args.output, args.report_sha256)
    print(json.dumps(base.write_new(path, result, base.FINAL), sort_keys=True))


if __name__ == '__main__': main()

