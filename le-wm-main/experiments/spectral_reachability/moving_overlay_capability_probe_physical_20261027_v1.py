"""Fixed moving-overlay suffix probes. MAIN-only physics; saved-only preparation."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.spectral_reachability import p1_moving_overlay_recompile_physical_20261027_v1 as base

NAME = 'moving_overlay_capability_probe_physical_20261027_v1'
AUTH = 'MAIN_MOVING_OVERLAY_CAPABILITY_PROBE_PHYSICS_ONLY'
ROOT, require, sha, raw = base.ROOT, base.require, base.sha, base.raw
CAP, FLOOR, FINAL = 192*1024**2, 200*1024**2, 8*1024**2
PROBE_NAMES = ('zero', '+x', '-x', '+y', '-y')


def probes():
    # Raw world-coordinate action, same .1 amplitude as frozen public warmup.
    directions = ((0, 0), (.1, 0), (-.1, 0), (0, .1), (0, -.1))
    return [dict(name=n, commands=np.tile(np.asarray(v, dtype=np.float32), (5, 1)).tolist())
            for n, v in zip(PROBE_NAMES, directions, strict=True)]


def contract():
    return dict(anchors=8, probes_per_anchor=5, fixed_denominator=40, independent_roots=2,
                order='slot/condition(off,on)/arm(GN,D2P)/probe(zero,+x,-x,+y,-y)',
                warmup=10, prefix=5, suffix=5, probes=probes(), raw_dtype='float32',
                action_scale=100, relative=True, native_position_clip=False,
                raw_action_bounds=[-1, 1], state_dimension=7, wall_seconds=900,
                output_cap_bytes=CAP, disk_floor_bytes=FLOOR, final_reserve_bytes=FINAL,
                restoration='fresh native reset recipe + identical warmup/prefix/render sequence; no suffix setter',
                cache_identity='deterministic reconstruction, NOT serialized/cache-bit certification',
                caps=dict(constructor=40, close=40, reset=80, public_step=800,
                          physics_step=8160, set_state=160, goal_setter=120,
                          goal_render=80, goal_bind=40, explicit_render=2640),
                no_neural=True, no_goal_pose_edit=True, P1_admission=False,
                interpretation='mechanism calibration only; no task-winning or physical benefit claim')


def read(path, limit=8*1024**2):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= limit,
            'bounded regular metadata')
    def pairs(items):
        result = {}
        for k, v in items:
            require(k not in result, 'duplicate JSON key'); result[k] = v
        return result
    result = json.loads(path.read_bytes(), object_pairs_hook=pairs)
    raw(result)  # Reject NaN/Inf, including parseable Python JSON extensions.
    return result


def own_files():
    return [Path(__file__).resolve(), ROOT/'tests'/('test_'+NAME+'.py'),
            ROOT/'innerdocs'/(NAME+'.md'), *base.own_files()]


def bind(bindings, path, expected=None):
    p = Path(path).resolve()
    require(p.is_file() and not p.is_symlink(), 'regular saved input')
    h = sha(p)
    require(expected is None or h == expected, 'saved input SHA mismatch')
    require(str(p) not in bindings or bindings[str(p)] == h, 'conflicting input seal')
    bindings[str(p)] = h
    return p


def validate_rows(report, plans):
    require(report['schema'] == base.NAME+'.report' and
            report['status'] == 'moving_overlay_recompile_physical_recorded' and
            report['errors'] == [] and report['inputs_final_verified'] is True and
            report['fixed_denominator'] == 8 and report['new_neural_calls'] == 0,
            'complete sealed physical report required')
    require(len(report['rows']) == len(plans) == 8, 'eight physical anchors required')
    for i, (row, plan) in enumerate(zip(report['rows'], plans, strict=True)):
        require(row['index'] == i and all(row[k] == plan[k] for k in
                ('root_slot', 'condition', 'arm', 'selection_digest', 'task_id')),
                'physical/planner exact identity join')
        require(row['initial_exact'] is True and row['warmup_returned'] == 10 and
                row['own_returned'] >= 5 and row['first_error'] is None and
                not row['secondary_errors'], 'unbroken true prefix5 required')
        require(len([c for c in row['checkpoints'] if c['step'] == 5]) == 1,
                'unique prefix5 checkpoint')
        for phase, steps in (('warmup', range(11)), ('plan', range(6))):
            for step in steps:
                require(len([r for r in row['trajectory'] if
                             r['phase'] == phase and r['step'] == step]) == 1,
                        'complete warmup/history/prefix receipts')


def validate_transitions(events, branch, plan):
    for phase, commands in (('warmup', base.moving.warmup_commands()),
                            ('plan', base.previous.commands(plan['commands'])[:5])):
        for step, command in enumerate(commands, 1):
            found = [e for e in events if e['kind'] == 'returned_transition' and
                     e.get('branch') == branch and e['phase'] == phase and e['step'] == step]
            require(len(found) == 1, 'unique original public transition')
            e = found[0]
            require(e['requested_bits'] == e['actual_bits'] == command.tobytes().hex() and
                    e['actual_dtype'] == np.dtype(np.float32).str and
                    e['terminated'] is False and e['truncated'] is False,
                    'original prefix/history action bits or native stop')


def material(physical_output, physical_report_sha):
    """Read-only metadata/array/hash closure. No factory, step, model or physics."""
    base.hash_value(physical_report_sha)
    directory = Path(physical_output).resolve()
    rp = directory/'report.json'
    require(sha(rp) == physical_report_sha, 'external physical report SHA')
    report = read(rp)
    ebp = directory/'execution_binding.json'
    require(sha(ebp) == report['execution_binding_sha256'], 'physical execution binding SHA')
    eb = read(ebp)
    require(eb == report['execution_binding'] and
            eb['prepare_sha256'] == report['prepare_sha256'], 'physical binding identity')
    original, data = base.verify(eb['prepared'], eb['prepare_sha256'])
    require(report['planner_report_sha256'] == original['planner_report_sha256'] and
            report['source_report_sha256'] == original['source_report_sha256'],
            'physical/planner/source closure')
    validate_rows(report, original['plans'])
    bindings = deepcopy(original['bindings'])
    for p, h in ((rp, physical_report_sha), (ebp, report['execution_binding_sha256']),
                 (eb['prepared'], eb['prepare_sha256']),
                 (directory/'journal.jsonl', report['journal_sha256'])):
        bind(bindings, p, h)
    events = base.previous.journal(directory/'journal.jsonl', report['journal_chain_tail'])
    anchors = []
    for i, row in enumerate(report['rows']):
        validate_transitions(events, i, original['plans'][i])
        receipts = []
        for phase, steps in (('warmup', range(11)), ('plan', range(6))):
            for step in steps:
                selected = next(r for r in row['trajectory'] if r['phase'] == phase and r['step'] == step)
                triplet = []
                for index in range(3):
                    found = [e for e in events if e['kind'] == 'fresh_observation' and
                             e.get('branch') == i and e['receipt']['phase'] == phase and
                             e['receipt']['step'] == step and e['receipt']['overlay_index'] == index]
                    require(len(found) == 1, 'unique durable physical render trace')
                    event = found[0]; r = event['receipt']
                    require(event['pre'] == event['post'] and
                            r['snapshot'] == event['post']['snapshot'] and
                            r['state'] == r['snapshot']['state'], 'immutable physical observation')
                    image = base.paired.flat(directory, r['png'])
                    path = bind(bindings, image, r['file_sha256'])
                    require(base.moving.image_sha(base.old_image(dict(path=str(path),
                            file_sha256=r['file_sha256'], pixel_sha256=r['pixel_sha256']))) == r['pixel_sha256'],
                            'physical pixel seal')
                    triplet.append(deepcopy(event))
                require(selected == triplet[1 if row['condition'] == 'on' else 0]['receipt'],
                        'report trajectory/journal join')
                receipts.extend(triplet)
        anchor = next(c for c in row['checkpoints'] if c['step'] == 5)
        selected = next(r for r in row['trajectory'] if r['phase'] == 'plan' and r['step'] == 5)
        require(all(anchor[k] == selected[k] for k in anchor), 'checkpoint/trajectory identity')
        anchors.append(dict(index=i, checkpoint=deepcopy(anchor), trace=receipts))
    for f in own_files(): bind(bindings, f)
    p = dict(schema=NAME+'.prepare', contract=contract(), physical_output=str(directory),
             physical_report_sha256=physical_report_sha, physical_prepare_sha256=eb['prepare_sha256'],
             planner_report_sha256=original['planner_report_sha256'],
             plans=deepcopy(original['plans']), anchors=anchors, bindings=bindings)
    return p, data


def prepare(output, physical_output, physical_report_sha):
    p, _ = material(physical_output, physical_report_sha)
    value = raw(p); path = Path(output).absolute()
    require(len(value) <= 8*1024**2 and not path.exists() and
            shutil.disk_usage(path.parent).free >= FLOOR+len(value), 'exclusive bounded prepare')
    with path.open('xb') as f:
        require(f.write(value) == len(value), 'short prepare'); f.flush(); os.fsync(f.fileno())
    base.paired.fsync_dir(path.parent)
    return dict(prepared=str(path), prepare_sha256=sha(path))


def verify(prepared, expected):
    base.hash_value(expected)
    require(sha(prepared) == expected, 'external prepare SHA')
    # Trace closure can exceed base.read's 2MiB metadata cap; bounded separately.
    require(Path(prepared).stat().st_size <= 8*1024**2, 'bounded probe prepare')
    p = read(prepared)
    now, data = material(p['physical_output'], p['physical_report_sha256'])
    require(p == now and sha(prepared) == expected, 'complete probe prepare closure')
    return p, data


class Store(base.Store):
    def __init__(self, path):
        self.path = Path(path).absolute()
        require(not self.path.exists() and not self.path.is_symlink() and
                shutil.disk_usage(self.path.parent).free >= FLOOR+CAP, 'exclusive output + disk floor')
        self.path.mkdir(); base.paired.fsync_dir(self.path.parent)
        self.previous = None; self.index = 0

    def check(self, size=4096, final=False):
        used = sum(f.stat().st_size for f in self.path.iterdir() if f.is_file())
        require(used+size <= CAP-(0 if final else FINAL) and
                shutil.disk_usage(self.path).free >= FLOOR+size, 'probe output cap/disk floor')


class Meter(base.Meter):
    def __init__(self, store, started=None):
        super().__init__(store, started)
        self.counts = {k: dict(attempted=0, completed=0) for k in contract()['caps']}
        self.active = {k: 0 for k in self.counts}

    def call(self, kind, fn, *, cleanup=False, **fields):
        require(not cleanup or kind == 'close', 'cleanup only close')
        if not cleanup:
            require(time.monotonic()-self.started < 900, '900s predispatch wall cap')
            self.store.check()
        c = self.counts[kind]
        require(c['attempted'] < contract()['caps'][kind], 'predispatch '+kind+' cap')
        c['attempted'] += 1; fields = {**self.where, **fields}
        self.store.emit('operation_attempted', resource=kind, **fields)
        self.active[kind] += 1
        try:
            result = fn(); c['completed'] += 1
        except BaseException as e:
            self.store.emit('operation_failed', resource=kind, error=base.error(kind, e), **fields)
            raise
        finally: self.active[kind] -= 1
        self.store.emit('operation_completed', resource=kind, **fields)
        return result


class TraceStore:
    """Observe already-journaled receipts without changing sealed helper globals."""
    def __init__(self, store):
        self.store = store; self.observations = []

    def __getattr__(self, name): return getattr(self.store, name)

    def emit(self, kind, **fields):
        self.store.emit(kind, **fields)
        if kind == 'fresh_observation': self.observations.append(deepcopy(fields))


def compare_trace(row, anchor, store, meter):
    # Every replay image is already durable before this exact comparison.
    require(len(row['trajectory']) == 17, 'all original warmup and prefix observations')
    require(len(store.observations) == len(anchor['trace']) == 51, 'full original off/on/off trace')
    for actual, expected in zip(store.observations, anchor['trace'], strict=True):
        require(actual['pre'] == actual['post'] == expected['pre'] == expected['post'],
                'full dynamics/goal/renderer invariant differs; no suffix dispatch')
        require(all(actual['receipt'][k] == expected['receipt'][k] for k in
                    ('phase', 'step', 'condition', 'overlay_index', 'file_sha256', 'pixel_sha256', 'state', 'snapshot')),
                'full triplet receipt differs; no suffix dispatch')
    for actual in row['trajectory']:
        index = 1 if row['condition'] == 'on' else 0
        expected = next(e['receipt'] for e in anchor['trace'] if
                        e['receipt']['phase'] == actual['phase'] and e['receipt']['step'] == actual['step'] and
                        e['receipt']['overlay_index'] == index)
        require(all(actual[k] == expected[k] for k in
                    ('phase', 'step', 'condition', 'overlay_index', 'file_sha256', 'pixel_sha256', 'state', 'snapshot')),
                'exact replay trace mismatch; no suffix dispatch')
    actual = row['checkpoints'][-1]
    require(actual['step'] == 5 and all(actual[k] == anchor['checkpoint'][k] for k in
                ('state', 'snapshot', 'file_sha256', 'pixel_sha256')), 'exact original prefix5 anchor')
    store.emit('suffix_start_verified', **meter.where, checkpoint=actual,
               restoration=contract()['restoration'], cache_identity=contract()['cache_identity'])


def episode(env, root, source_plan, plan, anchor, probe, row, meter, store):
    store = TraceStore(store)
    body = env.unwrapped; packet = root['packets'][plan['condition']]
    obs, native_goal = base.place(env, source_plan, root['slot'], meter, store)
    frames = [base.capture(env, obs, 'warmup', 0, row, meter, store, root['warm'][0])]
    for step, command in enumerate(base.moving.warmup_commands(), 1):
        require(not body.eval_state(native_goal, base.source.vector(obs['state']))[0], 'warmup already terminal')
        obs, info, t, tr, reward, actual = base.dispatch(env, command, 'warmup', step, row, meter, store, native_goal)
        image = base.capture(env, obs, 'warmup', step, row, meter, store, root['warm'][step])
        base.paired.check_return(env, command, actual, info, native_goal, reward, t)
        require(not t and not tr, 'warmup stop; no padding')
        if step in (5, 10): frames.append(image)
    require(base.moving.exact(np.asarray(frames), packet['rgb_history']) and
            base.moving.exact(packet['raw_action_history'], base.moving.warmup_commands()), 'exact history3/raw10')
    goal = base.source.vector(root['goal'])
    require(base.source.root_hash(base.source.vector(obs['state']), goal) == root['root_sha256'], 'exact physical root')
    base.bind_goal(body, goal, packet['goal_rgb'], meter, store)
    bound = base.invariant(body)
    base.capture(env, obs, 'plan', 0, row, meter, store)
    require(not body.eval_state(goal, base.source.vector(obs['state']))[0], 'root already terminal')
    for step, command in enumerate(base.previous.commands(plan['commands'])[:5], 1):
        obs, info, t, tr, reward, actual = base.dispatch(env, command, 'plan', step, row, meter, store, goal)
        base.capture(env, obs, 'plan', step, row, meter, store)
        base.paired.check_return(env, command, actual, info, goal, reward, t)
        require(not t and not tr, 'prefix stop; no suffix approximation')
    compare_trace(row, anchor, store, meter)
    # Compare ALL triplet invariants, not only the selected on/off label.
    require({k: v for k, v in base.invariant(body).items() if k != 'snapshot'} ==
            {k: v for k, v in bound.items() if k != 'snapshot'}, 'goal/renderer drift before suffix')
    start = base.source.vector(obs['state']).copy(); row['start_state'] = start.tolist()
    row['start_verified'] = True
    for step, command in enumerate(np.asarray(probe['commands'], dtype=np.float32), 1):
        obs, info, t, tr, reward, actual = base.dispatch(env, command, 'probe', step, row, meter, store, goal)
        base.capture(env, obs, 'probe', step, row, meter, store)
        base.paired.check_return(env, command, actual, info, goal, reward, t)
        now = base.invariant(body)
        require({k: v for k, v in now.items() if k != 'snapshot'} ==
                {k: v for k, v in bound.items() if k != 'snapshot'} and
                base.moving.exact(np.asarray(info['goal']), packet['goal_rgb']), 'goal/renderer drift')
        state = base.source.vector(obs['state']); delta = state-start
        row['responses'].append(dict(step=step, state=state.tolist(), delta_native7=delta.tolist(),
                                    delta_angle_wrapped=float(np.arctan2(np.sin(delta[4]), np.cos(delta[4]))),
                                    terminated=t, truncated=tr))
        if t or tr:
            row['status'] = 'native_stopped'; return
    row['status'] = 'probe_completed'


def paired_responses(rows):
    """Descriptive D2P-minus-GN vectors, never ranking or weighted scoring."""
    result = []
    for slot in (0, 1):
        for condition in base.CONDITIONS:
            for probe in PROBE_NAMES:
                pair = [next(r for r in rows if (r['root_slot'], r['condition'], r['arm'], r['probe']) ==
                             (slot, condition, arm, probe)) for arm in base.ARMS]
                starts = [r['start_state'] if r['start_verified'] else None for r in pair]
                delta = None if any(s is None for s in starts) else (np.asarray(starts[1])-starts[0]).tolist()
                steps = []
                for step in range(1, 6):
                    responses = [next((x for x in r['responses'] if x['step'] == step), None) for r in pair]
                    difference = None if any(s is None for s in starts) or any(x is None for x in responses) else (
                        np.asarray(responses[1]['delta_native7'])-responses[0]['delta_native7']).tolist()
                    steps.append(dict(step=step, D2P_minus_GN_response_delta_native7=difference))
                result.append(dict(root_slot=slot, condition=condition, probe=probe,
                                   D2P_minus_GN_prefix_state_native7=delta, steps=steps))
    return result


def execute(p, data, store, factory, *, authorization, prepare_sha256, started=None):
    require(authorization == AUTH and p['schema'] == NAME+'.prepare' and p['contract'] == contract(), 'MAIN authorization / frozen contract')
    base.hash_value(prepare_sha256)
    rows = [dict(index=5*i+j, anchor_index=i, root_slot=plan['root_slot'], condition=plan['condition'],
                 arm=plan['arm'], selection_digest=plan['selection_digest'], task_id=plan['task_id'],
                 probe=probe['name'], status='not_started', start_verified=False, start_state=None,
                 responses=[], trajectory=[], checkpoints=[], missing_checkpoints=[0, 5, 10],
                 PNG=0, warmup_returned=0, own_attempted=0, own_returned=0, secondary_errors=[])
            for i, plan in enumerate(p['plans']) for j, probe in enumerate(probes())]
    require(len(rows) == 40 and len(p['anchors']) == 8, 'fixed forty branches')
    meter = Meter(store, started); errors = []; seen = []; final_ok = False
    try:
        store.put('execution_binding.json', dict(prepare_sha256=prepare_sha256,
                  physical_report_sha256=p['physical_report_sha256'], planner_report_sha256=p['planner_report_sha256']))
        store.emit('run_started', contract=contract(), prepare_sha256=prepare_sha256)
        for row in rows:
            meter.where = {k: row[k] for k in ('root_slot', 'condition', 'arm', 'probe')}
            meter.where['branch'] = row['index']
            env = None
            try:
                env = meter.call('constructor', factory)
                require(all(env is not e and env.unwrapped is not e.unwrapped for e in seen), 'fresh environment per intervention')
                seen.append(env); base.validate_env(env.unwrapped)
                require(env.unwrapped.render_action is False and env.unwrapped.render_mode == 'rgb_array', 'native renderer')
                row['status'] = 'running'; i = row['anchor_index']
                with base.native_hooks(env, meter, row['secondary_errors']):
                    episode(env, data['roots'][row['root_slot']], data['source_plan'], p['plans'][i],
                            p['anchors'][i], probes()[row['index'] % 5], row, meter, store)
                require(not row['secondary_errors'], 'hook restoration failed')
            except BaseException as e:
                row['status'] = 'engineering_failed'; row['error'] = base.error('branch', e)
                errors.append(dict(index=row['index'], **row['error']))
            finally:
                if env is not None:
                    try: meter.call('close', env.close, cleanup=True)
                    except BaseException as e:
                        row['status'] = 'engineering_failed'; errors.append(dict(index=row['index'], **base.error('close', e)))
            store.emit('branch_closed', **meter.where, status=row['status'], responses=len(row['responses']))
    except BaseException as e: errors.append(base.error('execution', e))
    finally:
        try:
            for path, h in p['bindings'].items(): require(sha(path) == h, 'final input/source seal')
            final_ok = True
        except BaseException as e: errors.append(base.error('final_inputs', e))
        if time.monotonic()-meter.started >= 900: errors.append(dict(stage='wall', message='900s exhausted'))
        if not errors:
            try:
                c = {k: v['completed'] for k, v in meter.counts.items()}
                require(all(v['attempted'] == v['completed'] for v in meter.counts.values()), 'unreturned operation')
                require(c['constructor'] == c['close'] == 40 and c['reset'] == 80 and
                        c['physics_step'] == 2*c['reset']+10*c['public_step'] and
                        c['set_state'] == 2*c['reset'] and c['goal_bind'] == 40 and
                        c['goal_render'] == c['reset'] and c['goal_setter'] == c['reset']+c['goal_bind'] and
                        c['public_step'] == sum(r['warmup_returned']+r['own_returned'] for r in rows) and
                        c['explicit_render'] == sum(r['PNG'] for r in rows), 'native resource census')
            except BaseException as e: errors.append(base.error('census', e))
        status = 'capability_probes_recorded' if not errors and final_ok and all(
            r['status'] in ('probe_completed', 'native_stopped') for r in rows) else 'partial_or_failed'
        report = dict(schema=NAME+'.report', status=status, rows=rows, errors=errors,
                      fixed_denominator=40, independent_roots=2, prepare_sha256=prepare_sha256,
                      physical_report_sha256=p['physical_report_sha256'], planner_report_sha256=p['planner_report_sha256'],
                      contract=contract(), counts=meter.counts, inputs_final_verified=final_ok,
                      new_neural_calls=0, P1_admission=False, wall_seconds=time.monotonic()-meter.started)
        report['paired_descriptive_responses'] = paired_responses(rows)
        try:
            store.emit('run_finished', status=status, counts=meter.counts, errors=errors)
            report.update(journal_chain_tail=store.previous, journal_sha256=sha(store.path/'journal.jsonl'))
            store.put('report.json', report, final=True)
        except BaseException as e:
            report['status'] = 'partial_or_failed'; errors.append(base.error('final_report', e))
    return report


def run(prepared, expected, output, factory, *, authorization):
    require(authorization == AUTH, 'MAIN explicit physics authorization')
    started = time.monotonic(); p, data = verify(prepared, expected)
    return execute(p, data, Store(output), factory, authorization=authorization,
                   prepare_sha256=expected, started=started)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'verify', 'run'))
    for name in ('output', 'physical-output', 'physical-report-sha', 'prepared', 'prepare-sha'):
        parser.add_argument('--'+name)
    parser.add_argument('--authorize-physics', action='store_true'); a = parser.parse_args(argv)
    if a.command == 'prepare': result = prepare(a.output, a.physical_output, a.physical_report_sha)
    elif a.command == 'verify':
        p, _ = verify(a.prepared, a.prepare_sha); result = dict(status='verified_no_execution', contract=p['contract'])
    else:
        require(a.authorize_physics, 'MAIN --authorize-physics required')
        def factory():
            import gymnasium as gym
            import stable_worldmodel  # noqa: F401
            return gym.make('swm/PushT-v1', render_mode='rgb_array', resolution=224, with_target=False)
        result = run(a.prepared, a.prepare_sha, a.output, factory, authorization=AUTH)
    print(base.source.canonical({k: v for k, v in result.items() if k != 'rows'}))
    return int(result.get('status') == 'partial_or_failed')


if __name__ == '__main__': raise SystemExit(main())
