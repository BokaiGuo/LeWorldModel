"""MAIN-only physical reconstruction of sealed observed children, zero neural calls.

The protocol-only document is frozen; see the separate implementation document.
prepare/verify/consume are saved arithmetic. Only explicitly authorized run imports physics.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from experiments.spectral_reachability import agent_p3_child_program_cuda_20261028_v1 as compiler
from experiments.spectral_reachability import agent_p4_moving_multiboundary_20261028_v1 as p4

NAME = 'agent_p3_child_native_physical_20261028_v1'
ROOT, old = compiler.ROOT, p4.old
require, sha, digest, canonical = compiler.require, compiler.sha, compiler.digest, compiler.canonical
READ = compiler.base.read
AUTH = 'MAIN_P3_CHILD_NATIVE_PHYSICS_ONLY'
CAP, FINAL, FLOOR = 512*1024**2, 128*1024**2, 200*1024**2
PNG_CAP, JOURNAL_CAP = 256*1024, 80*1024**2
COMPILER_OUTPUT = ROOT/'results/agent_p3_child_program_cuda_20261028_v1_cuda_MAIN01'
COMPILER_SHA = '885f69f6de5ce9b78b8f3acdc84f14ef8891ba3ecd79912fc9af4e1920fc7ffb'
STRICT = ROOT/'results/agent_p3_child_program_cuda_20261028_v1_consume_MAIN01.json'
STRICT_SHA = 'd6322717231d72bb18e26eab1acbf6afe2dbc65eaa601af1fface21083a7e91c'
P4_OUTPUT = ROOT/'results/agent_p4_moving_multiboundary_20261028_v1_MAIN02'
P4_SHA = '60489b7db3d4eff70d1be46d47d92209e6adcf333ae1d0a0e389b259ec1f6084'
P4_PREPARE = ROOT/'results/agent_p4_moving_multiboundary_20261028_v1_prepare_MAIN02.json'
P4_PREPARE_SHA = '8528e8d7f9b688b1c93be154305ccc35b8ed6218de63e736b1ae567691c432a2'
FLAGS = dict(P1_admission=False, P3_admission=False, full_C3_demonstrated=False,
             compute_advantage=False, independent_confirmation=False, scientific_admission=False)
PINS = {str(Path(compiler.__file__).resolve()): 'f3bc4a58fb4dfc1ef640abd2ced50eafe5848a7dd9fc9110901cb98db5293e17',
        str(ROOT/'tests/agent_p3_child_program_cuda_20261028_v1_test.py'): '802fa926b75e1b758406e614681d470b6695d6befab7cc590d33b397a2546413',
        str(ROOT/'innerdocs/agent_p3_child_program_cuda_20261028_v1.md'): 'f8cd3c7579db41d1f65b3f9d41de9a47aacc2d4ac99bcf94b284e83abb6c57e0',
        str(Path(p4.__file__).resolve()): '2776029cc98f8e7d80bb566183ecdd5b44d8e4a0cd25e3f85fff23056b4bb68a',
        str(Path(old.__file__).resolve()): '6742a1b35102e433d7772e4a208b4f97b37b76f815b991cfc0e472763754ace9'}


def own_files():
    return [Path(__file__).resolve(), ROOT/f'tests/{NAME}_test.py',
            ROOT/f'innerdocs/{NAME}_implementation.md']


def exact(a, b, why):
    a, b = np.asarray(a), np.asarray(b)
    require(a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes(), why)


def raw(value, count):
    a = np.asarray(value)
    require(a.shape == (count, 2) and a.dtype.kind in 'fi' and np.isfinite(a).all() and
            np.array_equal(a, a.astype(np.float32)) and (abs(a) <= 1).all(), 'literal finite float32 raw actions')
    return a.astype(np.float32)


def native_frame(f):
    """Authenticate full native snapshot, not merely rolling visual seam.

    Native observation wraps angle, while body angle and its raw bits do not.
    """
    s = np.asarray(f['state'], dtype=np.float64)
    require(s.shape == (7,) and np.isfinite(s).all(), 'native seven finite values')
    snap = f['snapshot']; bodies = snap['bodies']
    for b in bodies.values():
        values = [*b['position'], *b['velocity'], b['unwrapped_angle'], b['angular_velocity'], *b['force'], b['torque']]
        require(np.isfinite(values).all() and type(b['is_sleeping']) is bool and
                np.asarray(values, dtype=np.float64).tobytes().hex() == b['float64_bits'], 'whole body float64 bits')
    agent, block = bodies['agent'], bodies['block']
    expected = np.asarray([*agent['position'], *block['position'], block['unwrapped_angle'] % (2*np.pi),
                           *agent['velocity']], dtype=np.float64)
    exact(s, expected, 'native observation angle modulo2pi; other six exact')
    exact(s, np.asarray(snap['state'], dtype=np.float64), 'snapshot observed state')
    require(s.tobytes().hex() == snap['state_float64_bits'], 'native state float64 bits')
    return f


def png(f, directory):
    from PIL import Image
    path = Path(f['path'])
    require(path.is_absolute() and path.parent == Path(directory) and not path.is_symlink() and
            path.stat().st_size <= PNG_CAP and sha(path) == f['file_sha256'], 'bound flat PNG bytes')
    with Image.open(path) as image:
        require(image.format == 'PNG' and image.size == (224, 224) and image.mode == 'RGB', 'native RGB224 PNG')
        a = np.asarray(image).copy()
    require(old.array_sha(a) == f['pixel_sha256'] and sha(path) == f['file_sha256'], 'actual PNG pixels')
    old.kernel.moving.guard(a, f['state']); native_frame(f)
    return a


def same_frame(a, b):
    for k in ('file_sha256', 'pixel_sha256', 'state', 'snapshot', 'goal', 'goal_rgb_sha256', 'goal_pose'):
        require(canonical(a[k]) == canonical(b[k]), 'whole original native/PNG/goal identity: '+k)


def journal(path, cap):
    require(path.stat().st_size <= cap, 'bounded journal')
    events, previous = [], None
    with path.open('rb') as stream:
        for line in stream:
            require(line.endswith(b'\n'), 'terminal journal newline')
            e = json.loads(line); check = dict(e); h = check.pop('event_sha256')
            require(e['index'] == len(events) and e['previous_sha256'] == previous and
                    old.source.digest(check) == h, 'complete journal hash chain')
            events.append(e); previous = h
    return events


def metric(goal, state):
    d = old.source.vector(state)-old.source.vector(goal)
    result = old.kernel.metric(goal, state)
    result.update(native7_reward=-float(np.linalg.norm(d)), joint_xy_error=float(np.linalg.norm(d[:4])),
                  object_position_error=float(np.linalg.norm(d[2:4])), velocity_error=float(np.linalg.norm(d[5:])),
                  wrapped_angle_error=float(abs(np.arctan2(np.sin(d[4]), np.cos(d[4])))))
    return result


def project(cp, receipt, pp, report, events, refs):
    require(len(cp['children']) == 24 and len(cp['branches']) == len(receipt['rows']) == 48, 'full24/48 denominator')
    frames = {}; returns = {}
    for e in events:
        if e['kind'] == 'fresh_observation':
            f = e['receipt']; key = (e['episode_id'], f['phase'], f['step'])
            require(key not in frames and f['observed_event_index'] == e['index'], 'unique original native frame')
            frames[key] = f
        if e['kind'] == 'control_return':
            key = (e['episode_id'], e['step']); require(key not in returns, 'unique original action ordinal')
            returns[key] = e
    rows = []
    for i, (branch, selected) in enumerate(zip(cp['branches'], receipt['rows'], strict=True)):
        c = compiler.contract.validate_child(cp['children'][i//2], cp['tasks'])
        require((branch['child_id'], branch['mode'], branch['root_slot']) ==
                (selected['child_id'], selected['mode'], selected['root_slot']) ==
                (c['child_id'], compiler.contract.MODES[i%2], c['root_slot']) and
                selected['branch_sha256'] == digest(branch), 'exact method identity and branch digest')
        route = 'execute' if branch['status'] == compiler.READY else branch['status']
        item = dict(child=deepcopy(c), mode=branch['mode'], route=route, selected=deepcopy(selected['selected']),
                    source_frames=None, source_returns=None)
        if c['status'] != 'not_reached':
            ref = refs[c['root_slot']]; eid, b = c['episode_id'], c['boundary']
            source_row = next(r for r in report['rows'] if r['episode_id'] == eid)
            layer = pp['layers'][compiler.contract.EPISODES.index(eid)//2]
            require((source_row['source_slot'], source_row['prefix_arm'], source_row['mode']) ==
                    (c['root_slot'], c['prefix_arm'], c['source_mode']) and layer['arm'] == c['prefix_arm'] and
                    layer['source_slot'] == c['root_slot'] and
                    c['ancestry']['source_report_sha256'] == P4_SHA, 'original source/task/layer/observed ancestry')
            require(cp['tasks'][c['root_slot']]['binding']['root_sha256'] == ref['root_sha256'] and
                    cp['tasks'][c['root_slot']]['binding']['task_sha256'] == layer['task_sha256'], 'same native root and task export')
            exact(raw(source_row['executed_raw'][:b], b), raw(c['executed_prefix'], b), 'actual original prefix raw bits')
            expected_history = np.concatenate([old.kernel.raw_history(ref['raw_history']), raw(c['executed_prefix'], b)])[-10:]
            exact(raw(c['raw_history10'], 10), expected_history, 'observed child raw10 including nonzero warmup')
            fs = [deepcopy(frames[(eid, 'warmup', j)]) for j in range(11)] + [deepcopy(frames[(eid, 'plan', j)]) for j in range(b+1)]
            rs = [deepcopy(returns[(eid, j)]) for j in range(1, b+1)]
            for j, e in enumerate(rs):
                exact(raw([e['command']], 1)[0], raw(c['executed_prefix'], b)[j], 'source return ordinal/raw')
                require(e['terminated'] is e['truncated'] is False, 'observed child must not restore after native earlystop')
            for f in fs:
                png(f, P4_OUTPUT)
                if f['phase'] == 'plan':
                    exact(np.asarray(f['goal']), np.asarray(ref['goal']), 'literal original native task goal')
                    require(f['goal_rgb_sha256'] == old.array_sha(ref['goal_rgb']), 'actual target RGB, not goal_pose')
            for j, n in enumerate((0, 5, 10)):
                exact(np.asarray(fs[n]['state']), np.asarray(ref['states'][j]), 'original warmup native state')
                require(fs[n]['snapshot'] == ref['dynamics'][j] and fs[n]['pixel_sha256'] == old.array_sha(ref['history'][j]),
                        'original warmup snapshot/RGB ancestry')
            last = fs[-1]; obs = c['observation']
            require((last['file_sha256'], last['pixel_sha256'], last['observed_event_index']) ==
                    (obs['png_sha256'], obs['pixel_sha256'], obs['observed_index']), 'actual boundary native PNG -> encoded context')
            item.update(source_frames=fs, source_returns=rs)
        if route == 'execute':
            require(selected['status'] == compiler.FINAL and selected['error'] is None and item['selected'] is not None and
                    item['selected']['parent_context_sha256'] == c['context_sha256'], 'full strict selected raw5 at same child')
            raw(item['selected']['raw5'], 5)
        else:
            require(item['selected'] is None and selected['status'] == branch['status'], 'refused/not reached remains unknown')
        rows.append(item)
    require([sum(r['route'] == k for r in rows) for k in ('execute', compiler.BLOCKED, 'not_reached')] == [41, 5, 2],
            'sealed actual41/5/2 roster, never outcome subset')
    return rows


def profile(rows):
    active = [r for r in rows if r['route'] == 'execute']; n = len(active)
    prefix = sum(r['child']['boundary'] for r in active); public = n*15+prefix
    return dict(caps=dict(constructor=n, close=n, reset=3*n, set_state=6*n, goal_render=3*n,
                         goal_setter=4*n, goal_bind=n, public_step=public, physics_step=6*n+10*public,
                         explicit_render=17*n+prefix), recovery_public_cap=10*n+prefix, own_public_cap=5*n,
                prefix_public_cap=prefix, output_bytes=CAP, final_reserve_bytes=FINAL, journal_bytes=JOURNAL_CAP,
                png_bytes=PNG_CAP, disk_floor_bytes=FLOOR, wall_seconds=600,
                wall_scope='execute entry after verify; through cleanup, before final report write',
                neural_calls=0, no_padding=True, no_retry=True, independent_roots=2,
                nested_resources_not_added=True, reset_physics_included=True)


def material():
    compiler.contract.check_sources(PINS)
    # All source boundaries checked BEFORE calling either potentially costly saved consumer.
    for path, h in ((STRICT, STRICT_SHA), (COMPILER_OUTPUT/'report.json', COMPILER_SHA),
                    (P4_OUTPUT/'report.json', P4_SHA), (P4_PREPARE, P4_PREPARE_SHA)):
        require(sha(path) == h, 'external real saved authority SHA: '+str(path))
    actual = compiler.consume(str(COMPILER_OUTPUT), COMPILER_SHA)
    require(canonical(actual) == canonical(READ(STRICT, 128*1024**2)), 'complete actual compiler full strict receipt equality')
    original = p4.consume(str(P4_OUTPUT), P4_SHA, str(P4_PREPARE), P4_PREPARE_SHA)
    require(original['status'] == 'saved_multiboundary_verified_NOT_scientific_pass', 'complete native/budget/fullarray P4 consumer')
    cr = READ(COMPILER_OUTPUT/'report.json', 256*1024**2)
    require(sha(cr['prepared']) == cr['prepare_sha256'], 'compiler prepare edge')
    cp = READ(cr['prepared'], 64*1024**2); pp = READ(P4_PREPARE, 64*1024**2)
    sources = deepcopy(cp['sources']); compiler.contract.merge(sources, PINS)
    require(sources.get(str(P4_PREPARE)) == P4_PREPARE_SHA and sources.get(str(P4_OUTPUT/'report.json')) == P4_SHA,
            'P4 authority is original compiler bound ancestry, not directory inference')
    manifest = old.read(pp['original_manifest'])
    data = old.kernel.load_source(manifest['sourcepack'], old.SOURCE_SHA, manifest['source_prepared'], manifest['source_prepare_sha256'])
    events = journal(P4_OUTPUT/'journal.jsonl', old.CAP)
    rows = project(cp, actual, pp, old.read(P4_OUTPUT/'report.json'), events, data['references'])
    files = [*own_files(), STRICT, COMPILER_OUTPUT/'report.json', Path(cr['prepared']), P4_PREPARE,
             P4_OUTPUT/'report.json', P4_OUTPUT/'journal.jsonl', Path(pp['original_manifest'])]
    for r in rows:
        for f in r['source_frames'] or []:
            compiler.contract.merge(sources, {f['path']: f['file_sha256']})
    compiler.contract.merge(sources, {str(p): sha(p) for p in files})
    compiler.contract.merge(sources, pp['bindings']); compiler.contract.merge(sources, data['bindings'])
    compiler.contract.check_sources(sources)
    return dict(schema=NAME+'.prepare', compiler_report_sha256=COMPILER_SHA, strict_sha256=STRICT_SHA,
                p4_report_sha256=P4_SHA, rows=rows, profile=profile(rows), sources=sources,
                source_manifest=pp['original_manifest'], fixed_denominator=48, child_denominator=24,
                historical_neural_resources=actual['new_resources'], **FLAGS)


def absolute(path):
    p = Path(path); require(p.is_absolute(), 'absolute CLI path'); return p


def write_new(path, value, cap=64*1024**2):
    path = absolute(path); payload = canonical(value)
    require(not path.exists() and not path.is_symlink() and len(payload) <= cap and
            shutil.disk_usage(path.parent).free >= FLOOR+len(payload), 'NEW bounded exclusive metadata/floor')
    with path.open('xb') as f:
        require(f.write(payload) == len(payload), 'complete metadata'); f.flush(); os.fsync(f.fileno())
    compiler.base.fsync_dir(path.parent)
    return dict(path=str(path), sha256=sha(path), bytes=len(payload))


def verify(path, expected):
    path = absolute(path); require(sha(path) == expected, 'external prepare SHA')
    p = READ(path, 64*1024**2); compiler.contract.check_sources(p['sources'])
    require(canonical(p) == canonical(material()) and sha(path) == expected, 'complete saved authority/native reconstruction')
    return p


class Store:
    def __init__(self, path):
        self.path = absolute(path)
        require(not self.path.exists() and not self.path.is_symlink() and shutil.disk_usage(self.path.parent).free >= FLOOR+CAP,
                'NEW512MiB directory +200MiB floor')
        self.path.mkdir(); compiler.base.fsync_dir(self.path.parent)
        self.index, self.previous, self.broken = 0, None, False
        self.recovery, self.recovered = None, 0

    def check(self, size=4096, final=False):
        used = sum(p.stat().st_size for p in self.path.iterdir() if p.is_file())
        require(used+size <= CAP-(0 if final else FINAL) and shutil.disk_usage(self.path).free >= FLOOR+size,
                'actual whole output cap/final reserve/floor')

    def write(self, name, value, append=False, final=False):
        require(Path(name).name == name and name not in ('', '.', '..') and not (self.path/name).is_symlink(), 'flat own output')
        self.check(len(value), final)
        if append:
            require(name == 'journal.jsonl' and (self.path/name).stat().st_size+len(value) <= JOURNAL_CAP
                    if (self.path/name).exists() else name == 'journal.jsonl' and len(value) <= JOURNAL_CAP, 'bounded journal append')
        with (self.path/name).open('ab' if append else 'xb') as f:
            require(f.write(value) == len(value), 'complete artifact'); f.flush(); os.fsync(f.fileno())
        compiler.base.fsync_dir(self.path)
        return sha(self.path/name) if not append else None

    def put(self, name, value): return self.write(name, canonical(value))

    def emit(self, kind, **fields):
        require(not self.broken, 'no retry after journal failure')
        event = dict(index=self.index, previous_sha256=self.previous, kind=kind, **deepcopy(fields))
        event['event_sha256'] = old.source.digest(event)
        try: self.write('journal.jsonl', (old.source.canonical(event)+'\n').encode(), append=True)
        except BaseException:
            self.broken = True; raise
        self.previous = event['event_sha256']; self.index += 1
        if kind == 'fresh_observation' and self.recovery is not None and fields['receipt']['phase'] != 'own':
            require(self.recovered < len(self.recovery), 'no extra recovery frame')
            native_frame(fields['receipt']); same_frame(fields['receipt'], self.recovery[self.recovered])
            self.recovered += 1

    def frame(self, name, image):
        from PIL import Image
        stream = io.BytesIO(); Image.fromarray(image).save(stream, format='PNG'); value = stream.getvalue()
        require(len(value) <= PNG_CAP, 'bounded native PNG'); return self.write(name, value)

    def arrays(self, name, **values):
        stream = io.BytesIO(); np.savez(stream, **values)
        require(stream.tell() <= 1024**2, 'bounded invalid render diagnostic'); return self.write(name, stream.getvalue())


class Meter(old.Meter):
    def __init__(self, store, caps):
        super().__init__(store)
        self.caps = deepcopy(caps); self.counts = {k: dict(attempted=0, completed=0) for k in caps}


def new_row(i, plan):
    return dict(ordinal=i, episode_id=plan['child']['child_id']+'_'+plan['mode'], plan_sha256=digest(plan),
                route=plan['route'], status='not_started' if plan['route'] == 'execute' else plan['route'],
                PNG=0, checkpoints=[], warmup_returned=0, prefix_returned=0, own_returned=0,
                executed_raw=[], initial_exact=False, last_observation=None, final_metrics=None,
                terminated=None, truncated=None, error=None, secondary_errors=[])


def step(env, command, row, meter, store, phase, ordinal, goal, goal_rgb):
    store.emit('authorized_dispatch', episode_id=row['episode_id'], phase=phase, step=ordinal, command=command.tolist())
    obs, reward, terminated, truncated, info = env.step(command.copy())
    row[phase+'_returned'] += 1
    actual = np.asarray(env.unwrapped.latest_action).copy()
    store.emit('step_return', episode_id=row['episode_id'], phase=phase, step=ordinal, command=actual.tolist(),
               reward=float(reward), terminated=bool(terminated), truncated=bool(truncated), state=np.asarray(obs['state']).tolist())
    exact(actual, command, 'actual literal raw step'); row['executed_raw'].append(actual.tolist())
    exact(np.asarray(info['goal_state']), np.asarray(goal), 'unchanged goal_state')
    exact(np.asarray(info['goal']), np.asarray(goal_rgb), 'unchanged actual goal RGB')
    old.native_transition(env.unwrapped, goal, obs, reward, terminated, truncated, row)
    _, f = old.capture(env, obs, row, store, phase, ordinal)
    return f, bool(terminated), bool(truncated)


def execute(env, plan, ref, source_plan, row, meter, store):
    store.recovery, store.recovered = plan['source_frames'], 0
    require(p4.setup(env, ref, source_plan, row, meter, store), 'original warmup cannot stop before observed child')
    fs = plan['source_frames']; same_frame(row['last_observation'], fs[11])
    # setup already checks warmup0/5/10 against source blind snapshots. Check ALL intervening frames too in saved consumer.
    meter.region = 'prefix'
    for j, command in enumerate(raw(plan['child']['executed_prefix'], plan['child']['boundary']), 1):
        f, terminated, truncated = step(env, command, row, meter, store, 'prefix', j, ref['goal'], ref['goal_rgb'])
        same_frame(f, fs[11+j]); require(not terminated and not truncated, 'no resetting/padding a stopped original prefix')
    boundary = row['last_observation']; same_frame(boundary, fs[-1]); native_frame(boundary)
    require(store.recovered == len(fs), 'all original native warmup/prefix frames checked BEFORE own dispatch')
    row['boundary'] = deepcopy(boundary)
    store.emit('child_boundary_sealed', episode_id=row['episode_id'], context_sha256=plan['child']['context_sha256'],
               raw_history10=plan['child']['raw_history10'], selected_sha256=digest(plan['selected']), boundary=boundary)
    meter.region = 'suffix'
    for j, command in enumerate(raw(plan['selected']['raw5'], 5), 1):
        f, terminated, truncated = step(env, command, row, meter, store, 'own', j, ref['goal'], ref['goal_rgb'])
        row.update(final_metrics=metric(f['goal'], f['state']), terminated=terminated, truncated=truncated)
        if terminated or truncated: break
    row['status'] = 'native_stopped' if row['terminated'] or row['truncated'] else 'returned5'


def run(prepared, expected, output, authorization):
    require(authorization == AUTH, 'MAIN explicit physical execution authorization required')
    p = verify(prepared, expected)  # outside execution wall, no neural or physics
    return run_verified(p, prepared, expected, output)


def run_verified(p, prepared, expected, output):
    """Internal implementation; CLI must pass run authorization. RAM tests inject no native backend."""
    started = time.monotonic()
    m = old.read(p['source_manifest'])
    data = old.kernel.load_source(m['sourcepack'], old.SOURCE_SHA, m['source_prepared'], m['source_prepare_sha256'])
    store = Store(output); meter = Meter(store, p['profile']['caps'])
    meter.job_check = lambda: require(time.monotonic()-started < 600, 'execute entry600s deadline')
    rows = [new_row(i, r) for i, r in enumerate(p['rows'])]; errors = []
    store.put('binding.json', dict(prepared=str(prepared), prepare_sha256=expected))
    store.emit('run_started', profile=p['profile'])
    for plan, row in zip(p['rows'], rows, strict=True):
        if plan['route'] != 'execute' or errors: continue
        env = None; meter.episode = row['episode_id']; meter.region = 'setup'
        store.emit('branch_started', episode_id=row['episode_id'], ordinal=row['ordinal'], plan_sha256=row['plan_sha256'])
        try:
            env = meter.call('constructor', lambda: old.native_factory(meter))
            execute(env, plan, data['references'][plan['child']['root_slot']], data['plan'], row, meter, store)
        except Exception as exc:
            row['status'] = 'engineering_failed'; row['error'] = dict(type=type(exc).__name__, message=str(exc))
            errors.append(dict(ordinal=row['ordinal'], **row['error']))
        finally:
            if env is not None:
                # Cleanup must still happen after exhausted wall/output. Count honestly without re-dispatching physics.
                meter.counts['close']['attempted'] += 1
                if not store.broken:
                    try: store.emit('cleanup_attempted', episode_id=row['episode_id'])
                    except Exception as exc: errors.append(dict(stage='cleanup_journal', message=str(exc)))
                try:
                    env.close(); meter.counts['close']['completed'] += 1
                    if not store.broken:
                        try: store.emit('cleanup_completed', episode_id=row['episode_id'])
                        except Exception as exc: errors.append(dict(stage='cleanup_journal', message=str(exc)))
                except Exception as exc:
                    row['secondary_errors'].append(dict(stage='close', message=str(exc)))
                    errors.append(dict(ordinal=row['ordinal'], stage='close', message=str(exc)))
            if not store.broken:
                try: store.emit('branch_finished', episode_id=row['episode_id'], row=row, cleanup_counts=meter.counts['close'])
                except Exception as exc: errors.append(dict(stage='journal', message=str(exc)))
    if time.monotonic()-started >= 600: errors.append(dict(stage='wall', message='execution600s exceeded'))
    try: compiler.contract.check_sources(p['sources'])
    except Exception as exc: errors.append(dict(stage='final_sources', message=str(exc)))
    report = dict(schema=NAME+'.report', prepared=str(prepared), prepare_sha256=expected, rows=rows,
                  resources=meter.counts, pending=list(meter.pending.values()), secondary_errors=meter.secondary,
                  errors=errors, fixed_denominator=48, child_denominator=24, profile=p['profile'],
                  wall_seconds=time.monotonic()-started, journal_sha256=sha(store.path/'journal.jsonl'),
                  status='recorded' if not errors else 'engineering_failed', **FLAGS)
    store.write('report.json', canonical(report), final=True)
    return dict(output=str(store.path), report_sha256=sha(store.path/'report.json'), status=report['status'], errors=errors)


def ledger(events, caps):
    counts = {k: dict(attempted=0, completed=0) for k in caps}; pending = {}; serial = 0; closing = set(); closed = set(); stack = []
    regions = dict(constructor=('setup',), reset=('setup',), set_state=('setup',), goal_render=('setup',),
                   goal_setter=('setup', 'goal_bind'), goal_bind=('goal_bind',),
                   public_step=('warmup', 'prefix', 'suffix'), physics_step=('setup', 'warmup', 'prefix', 'suffix'),
                   explicit_render=('setup', 'warmup', 'goal_bind', 'prefix', 'suffix'))
    for e in events:
        if e['kind'] == 'operation_attempted':
            require(e['operation_id'] == serial and e['resource'] in caps and e['units'] == 1, 'known unique attempted operation')
            require(e['region'] in regions[e['resource']], 'original native resource scope, no setters/reset after recovery')
            serial += 1; pending[e['operation_id']] = e; stack.append(e['operation_id']); counts[e['resource']]['attempted'] += 1
        elif e['kind'] == 'operation_completed':
            require(stack and stack.pop() == e['operation_id'], 'actual nested native entry/return order')
            a = pending.pop(e['operation_id'])
            require(all(e[k] == a[k] for k in ('resource', 'units', 'episode_id', 'region')), 'same actual operation return')
            counts[e['resource']]['completed'] += 1
        elif e['kind'] == 'cleanup_attempted':
            require(e['episode_id'] not in closing and e['episode_id'] not in closed, 'one actual cleanup per branch')
            closing.add(e['episode_id']); counts['close']['attempted'] += 1
        elif e['kind'] == 'cleanup_completed':
            require(e['episode_id'] in closing, 'cleanup actual return')
            closing.remove(e['episode_id']); closed.add(e['episode_id']); counts['close']['completed'] += 1
        elif e['kind'] == 'branch_finished':
            require(e['cleanup_counts'] == counts['close'], 'cumulative cleanup return receipt')
    require(all(0 <= v['completed'] <= v['attempted'] <= caps[k] for k, v in counts.items()), 'attempt/return caps, nested not added')
    return counts, [{k: e[k] for k in ('operation_id', 'episode_id', 'resource', 'units', 'region')} for e in pending.values()]


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
        want = dict(constructor=1, reset=3, set_state=6, goal_render=3, goal_setter=4, goal_bind=1,
                    public_step=10+b+n, physics_step=6+10*(10+b+n), explicit_render=12+b+n)
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
    output = absolute(output); require(sha(output/'report.json') == expected, 'external actual physical report SHA')
    r = READ(output/'report.json', FINAL); require(r['schema'] == NAME+'.report', 'new own physical schema')
    p = verify(r['prepared'], r['prepare_sha256'])
    require(READ(output/'binding.json') == dict(prepared=r['prepared'], prepare_sha256=r['prepare_sha256']), 'actual execution/prepare binding')
    require(sha(output/'journal.jsonl') == r['journal_sha256'], 'durable actual journal SHA')
    require(sum(f.stat().st_size for f in output.iterdir() if f.is_file()) <= CAP and
            all(f.is_file() and not f.is_symlink() for f in output.iterdir()), 'bounded flat saved output')
    result = validate_saved(p, r, journal(output/'journal.jsonl', JOURNAL_CAP), output)
    compiler.contract.check_sources(p['sources'])
    require(sha(output/'report.json') == expected, 'actual report unchanged'); result['report_sha256'] = expected
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('prepare'); a.add_argument('--output', required=True)
    for name in ('verify', 'run'):
        a = sub.add_parser(name); a.add_argument('--prepared', required=True); a.add_argument('--prepare-sha256', required=True)
        if name == 'run': a.add_argument('--output', required=True); a.add_argument('--authorization', default='')
    a = sub.add_parser('consume'); a.add_argument('--output', required=True); a.add_argument('--report-sha256', required=True)
    a.add_argument('--receipt', required=True)
    args = parser.parse_args()
    if args.cmd == 'prepare': result = write_new(args.output, material())
    elif args.cmd == 'verify':
        p = verify(args.prepared, args.prepare_sha256); result = dict(verified=True, profile=p['profile'])
    elif args.cmd == 'run': result = run(args.prepared, args.prepare_sha256, absolute(args.output), args.authorization)
    else: result = write_new(args.receipt, consume(args.output, args.report_sha256), FINAL)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    if args.cmd == 'run' and result['status'] != 'recorded': raise SystemExit(2)


if __name__ == '__main__': main()
