"""Exact native replay before target-color rendering. MAIN-only physics, no neural calls."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

import numpy as np

NAME = 'agent_p5_replay_target_renderer_20261028_v1'
ROOT = Path(__file__).resolve().parents[2]
V3 = 'agent_p5_renderer_factorial_20261028_v3'
V3_SHA = 'b2b41d1aee81d37a1569fe4943fa1b31cb8c7dbee30790fa4c07ee7133a97fcd'
V3_PREP = ROOT/'results/agent_p5_renderer_factorial_20261028_v3_prepare_MAIN01.json'
V3_PREP_SHA = 'fdf76d7f0d404065bdee5abd404d4f39e0f68877263cc4862ce2dc84fe20109a'
SCENE = 'p5_native_goal_scene_20261027_v1'
SCENE_SHA = 'ea2e77e43a6e46f9f9c8aafd6c728eeb999cbfefd0284656f5492178c8ece16d'
STATES = ('xy', 'yx', 'x_minus_x', 'zero_zero', 'goal')
COLORS = ('visible', 'white')
RGBA = {'visible': [144, 238, 144, 255], 'white': [255, 255, 255, 255]}
NAMES = ('rgb.npy', 'state.npy', 'proprio.npy', 'transition.json', 'full_public_state.json')
CAP, RESERVE, FLOOR, WALL = 32*1024**2, 2*1024**2, 200*1024**2, 900
EXPECTED = dict(env_construct=5, seed=10, reset=5, control_step=80, space_step=805,
                native_get_obs=85, native_render=95, setup=5, set_state=5,
                space_construct=5, native_goal_body=176, body_construct=191,
                debug_draw=95, close=5)
DELTAS = dict(construct=dict(env_construct=1,seed=1), seed=dict(seed=1),
    reset=dict(reset=1,setup=1,set_state=1,space_construct=1,space_step=1,
        native_get_obs=1,native_render=1,native_goal_body=1,body_construct=4,debug_draw=1),
    step=dict(control_step=1,space_step=10,native_get_obs=1,native_render=1,
        native_goal_body=2,body_construct=2,debug_draw=1),
    geometry=dict(native_goal_body=1,body_construct=1),
    color=dict(native_render=1,native_goal_body=1,body_construct=1,debug_draw=1),close=dict(close=1))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(x):
    return hashlib.sha256(canonical(x)).hexdigest()


def sha(path):
    path = Path(path)
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'regular nonsymlink file')
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024**2), b''):
            h.update(b)
    return h.hexdigest()


def read(path, expected=None):
    require(expected is None or sha(path) == expected, 'external file SHA')
    sha(path)
    require(Path(path).stat().st_size <= 16*1024**2, 'JSON read bound')
    def pairs(items):
        out = {}
        for k, v in items:
            require(k not in out, 'duplicate key')
            out[k] = v
        return out
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs,
                      parse_constant=lambda x: require(False, 'nonfinite JSON'))


def sync(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def new(path, data):
    path = Path(path)
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'nonsymlink destination')
    with path.open('xb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    sync(path.parent)


def npy(a):
    f = io.BytesIO(); np.save(f, a, allow_pickle=False)
    return f.getvalue()


def load_bound(name, expected):
    path = ROOT/'experiments/spectral_reachability'/f'{name}.py'
    require(sha(path) == expected, 'sealed helper source '+name)
    return importlib.import_module('experiments.spectral_reachability.'+name)


def protocol():
    return dict(states=list(STATES), colors=list(COLORS), denominator=10, baseline_denominator=5,
        control_order='four prefix10 with original full-prefix gate, four suffix10, independent goal reset; all five baselines then ten colors',
        cache='original live native spaces retained; no hydration/reindex/cache fabrication',
        target='only native goal_color changes, original object restored in finally; agent stays visible',
        expected_native_entries=EXPECTED, public_operations=dict(construct=5, seed=5, reset=5,
        step=80, geometry=1, color=10, close=5), output_cap_bytes=CAP,
        final_reserve_bytes=RESERVE, free_floor_bytes=FLOOR, wall_seconds=WALL,
        neural=0, physics_steps=805, new_renders=10, native_renders=95,
        scientific_admission=False, task_success=None, fresh=False, selection=False,
        counter_semantics='Python entries/return-events; durable public intent/returned/failed separate; exceptional return-events are not successful calls')


def validate_slots(states, plans):
    require([s['state_id'] for s in states] == list(STATES), 'exact five states/order')
    require([p['arm'] for p in plans] == list(STATES[:4]), 'exact four plans/order')
    for p in plans:
        for key in ('prefix_raw_u', 'reference_suffix_raw_u'):
            a = np.asarray(p[key], np.float64)
            require(a.shape == (10, 2) and np.isfinite(a).all(), 'exact finite raw10')


def build():
    v3 = load_bound(V3, V3_SHA)
    v = v3.verify(V3_PREP, V3_PREP_SHA)
    bindings = {**v['bindings'], str(V3_PREP): V3_PREP_SHA}
    def bind(path, expected=None):
        path = str(Path(path).absolute()); h = sha(path)
        require(expected is None or h == expected, 'authority edge SHA')
        bindings[path] = h
        return read(path, h)
    authority = v['native_manifest']['authority']
    attribution = bind(authority['manifest'], bindings[authority['manifest']])
    paths = attribution['authority']['paths']
    original = bind(paths['phys_report'], attribution['bindings'][paths['phys_report']])
    physical = bind(paths['phys_manifest'], original['manifest_sha256'])
    require(original['manifest_digest'] == digest(physical) and original['global_error'] is None,
            'original physical report -> exact original prepare')
    plans = physical['source_contract']['generator_private']['plans']
    states = v['native_manifest']['states']
    validate_slots(states, plans)
    require([r['arm'] for r in original['rows']] == list(STATES[:4]), 'original rows/order')
    frames = {}
    for s, row in zip(states[:4], original['rows']):
        require([r['frame'] for r in row['records']] == list(range(21)), 'all original frames')
        require(Path(s['native_path']) == Path(paths['phys_report']).parent/s['state_id']/'frame_20/full_public_state.json',
                'state provenance -> exact original physical row')
        frames[s['state_id']] = []
        for r in row['records']:
            rec = {}
            for name in NAMES:
                path = Path(paths['phys_report']).parent/s['state_id']/f"frame_{r['frame']:02d}"/name
                bind(path, r['files'][name]['sha256']) if path.suffix == '.json' else bindings.update({str(path): sha(path)})
                require(sha(path) == r['files'][name]['sha256'] and path.stat().st_size == r['files'][name]['bytes'], 'original frame receipt')
                rec[name] = dict(path=str(path), **r['files'][name])
            frames[s['state_id']].append(rec)
    # Goal source and protocol come from the already-bound original scene report, not guessed target setters.
    goal_report_path = Path(states[-1]['native_path']).parent.parent/'report.json'
    require(str(goal_report_path) in bindings, 'goal report must already be in v3 closure')
    goal_report = bind(goal_report_path, bindings[str(goal_report_path)])
    require(goal_report['protocol']['schema'] == SCENE and goal_report['errors'] == [], 'original goal scene protocol')
    scene = ROOT/'experiments/spectral_reachability'/f'{SCENE}.py'
    require(sha(scene) == SCENE_SHA, 'original goal scene source')
    bindings[str(scene)] = SCENE_SHA
    goal_frame = {}
    for name in NAMES:
        path = Path(states[-1]['native_path']).parent/name
        bindings[str(path)] = sha(path)
        goal_frame[name] = dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)
    frames['goal'] = [goal_frame]
    geometry_path = Path(states[-1]['native_path']).parent.parent/'geometry.json'
    require(sha(geometry_path) == bindings[str(geometry_path)], 'original extra goal geometry receipt')
    for path in (Path(__file__).resolve(), ROOT/f'tests/test_{NAME}.py', ROOT/f'innerdocs/{NAME}.md'):
        bindings[str(path)] = sha(path)
    require(all(sha(f) == h for f, h in bindings.items()), 'complete stable input closure')
    return dict(schema=NAME+'.prepare', protocol=protocol(), v3_prepare=str(V3_PREP), v3_sha256=V3_PREP_SHA,
        original_physical_report=paths['phys_report'], original_physical_prepare=paths['phys_manifest'],
        states=states, plans=plans, frames=frames, goal_protocol=goal_report['protocol'],
        goal_geometry=dict(path=str(geometry_path),sha256=bindings[str(geometry_path)]),
        bindings=bindings, scientific_admission=False, physics_executed=False)


def verify(path, expected):
    p = read(path, expected)
    require(p == build(), 'complete deterministic replay prepare')
    return p


class Store:
    def __init__(self, out):
        self.out = Path(out); self.started = time.monotonic(); self.used = 0
        require(not self.out.exists() and self.out.parent.is_dir(), 'exclusive output')
        require(not any(p.is_symlink() for p in (self.out, *self.out.parents)), 'output ancestry')
        require(shutil.disk_usage(self.out.parent).free >= FLOOR+CAP, 'output cap plus 200MiB floor')
        self.out.mkdir(); sync(self.out.parent)
        self.previous = None; self.sequence = 0; self.files = {}
        self.journal = (self.out/'journal.jsonl').open('xb')
        self.journal.flush(); os.fsync(self.journal.fileno()); sync(self.out)

    def check(self, size=0, final=False):
        require(self.used+size <= CAP-(0 if final else RESERVE), 'output byte cap/reserve')
        require(shutil.disk_usage(self.out).free >= FLOOR+size, '200MiB floor')
        if not final:
            require(time.monotonic()-self.started < WALL, '900s work wall')

    def save(self, name, data, final=False):
        self.check(len(data), final)
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'local store path')
        path = self.out/name
        path.parent.mkdir(parents=True, exist_ok=True)
        cur = path.parent
        while cur != self.out.parent:
            sync(cur); cur = cur.parent
        new(path, data); self.used += len(data)
        rec = dict(path=name, sha256=sha(path), bytes=len(data)); self.files[name] = rec
        return rec

    def json(self, name, x, final=False):
        return self.save(name, canonical(x)+b'\n', final)

    def event(self, event, final=False):
        row = dict(sequence=self.sequence, previous_sha256=self.previous, event=event)
        row['entry_sha256'] = digest(row)
        b = canonical(row)+b'\n'; self.check(len(b), final)
        self.journal.write(b); self.journal.flush(); os.fsync(self.journal.fileno())
        self.used += len(b); self.sequence += 1; self.previous = row['entry_sha256']

    def image(self, stem, a):
        from PIL import Image
        a = np.asarray(a)
        require(a.shape == (224, 224, 3) and a.dtype == np.uint8, 'native uint8 RGB224')
        arr = self.save(stem+'.npy', npy(a))
        f = io.BytesIO(); Image.fromarray(a).save(f, format='PNG')
        png = self.save(stem+'.png', f.getvalue())
        return dict(npy=arr, png=png, pixel_sha256=hashlib.sha256(a.tobytes()).hexdigest())


class Meter:
    """Observe unmodified functions, no callback or method replacement."""
    def __init__(self, cls, pymunk, store, public_state=None):
        targets = dict(env_construct=cls.__init__, seed=cls.seed, reset=cls.reset, control_step=cls.step,
            native_get_obs=cls._get_obs, native_render=cls._render_frame, setup=cls._setup,
            set_state=cls._set_state, native_goal_body=cls._get_goal_pose_body, close=cls.close,
            space_construct=pymunk.Space.__init__, space_step=pymunk.Space.step,
            body_construct=pymunk.Body.__init__, debug_draw=pymunk.Space.debug_draw)
        require(all(hasattr(f, '__code__') for f in targets.values()), 'observable original native methods')
        self.codes = {f.__code__: k for k, f in targets.items()}
        self.entries = Counter(); self.returns = Counter(); self.errors = []
        self.store = store; self.context = None
        self.public_state = public_state; self.render_views = {}

    def profile(self, frame, event, arg):
        if event not in ('call', 'return') or frame.f_code not in self.codes:
            return
        key = self.codes[frame.f_code]
        counts = self.entries if event == 'call' else self.returns
        counts[key] += 1
        try:
            if key == 'native_render' and self.public_state is not None:
                env = frame.f_locals['self']
                if event == 'call':
                    self.render_views[id(env)] = dict(before=self.public_state(env))
                else:
                    self.render_views[id(env)]['after'] = self.public_state(env)
            self.store.event(dict(phase='native_'+event, name=key, context=self.context,
                dt=float(frame.f_locals['dt']) if key == 'space_step' else None))
            if self.entries[key] > EXPECTED[key]:
                raise ValueError('native entry cap '+key)
        except BaseException as exc:
            if len(self.errors) < 32:
                self.errors.append(type(exc).__name__+': '+str(exc))

    def receipt(self):
        return dict(entries={k:self.entries[k] for k in EXPECTED},
                    return_events={k:self.returns[k] for k in EXPECTED}, errors=list(self.errors))


def operation(store, meter, role, op, details, fn, cleanup=False):
    meter.context = dict(state_id=role, operation=op, frame=details.get('frame'))
    if not cleanup:
        require(not meter.errors, 'native observer error')
    store.event(dict(phase='intent', state_id=role, operation=op, details=details), final=cleanup)
    try:
        result = fn()
    except BaseException as exc:
        store.event(dict(phase='failed', state_id=role, operation=op,
                         error=type(exc).__name__+': '+str(exc)), final=True)
        raise
    store.event(dict(phase='returned', state_id=role, operation=op), final=cleanup)
    return result


def byte_gate(actual, expected):
    require(Path(actual).read_bytes() == Path(expected['path']).read_bytes()
            and sha(actual) == expected['sha256'], 'exact original baseline/frame bytes')


def topology(env):
    return (id(env.space), id(env.agent), id(env.block),
            tuple(sorted(id(x) for x in env.space.bodies)), tuple(sorted(id(x) for x in env.space.shapes)))


def render_color(env, label, pygame, base, store, meter, role, cell):
    old = env.goal_color; before = base.public_state(env); ids = topology(env)
    stem = role+'/'+label
    cell['before'] = store.json(stem+'_before.json', before)
    cell['original_color'] = list(old)
    try:
        env.goal_color = pygame.Color('LightGreen' if label == 'visible' else 'White')
        cell['requested_rgba'] = list(env.goal_color)
        rgb = operation(store, meter, role, 'color', dict(color=label), lambda: env._render_frame('rgb_array'))
        # Durable failing frame and native state BEFORE invariance guards.
        cell['image'] = store.image(stem, rgb)
        after = base.public_state(env)
        cell['after'] = store.json(stem+'_after.json', after)
        require(before == after and topology(env) == ids, 'render changed native state/RNG/topology')
        require(cell['requested_rgba'] == RGBA[label], 'actual native color')
    finally:
        env.goal_color = old
        cell['restored'] = store.json(stem+'_restored.json', dict(color=list(env.goal_color),
            same_color_object=env.goal_color is old, public_state=base.public_state(env),
            topology_unchanged=topology(env) == ids), final=True)
    cell['status'] = 'rendered'


def execute(p, prepared_sha, out, loader, stable=lambda: None):
    store = Store(out)
    rows = [dict(state_id=k, status='not_run', records=[], baseline=None,
                 shots=[dict(shot=s, status='not_run', files={}) for s in ('baseline', *COLORS)],
                 cells=[dict(color=c, status='not_run') for c in COLORS]) for k in STATES]
    errors = []; envs = {}; meter = None; profiled = False; gate = False
    store.json('execution_binding.json', dict(schema=NAME+'.execution', prepare_sha256=prepared_sha,
        prepare_digest=digest(p), protocol=protocol()))
    def capture(role, frame, obs, state, transition):
        row = rows[STATES.index(role)]; stem = role+f'/frame_{frame:02d}'
        rec = dict(frame=frame, image=store.image(stem+'/rgb', obs['visual']), files={})
        row['records'].append(rec)
        rec['files']['rgb.npy'] = rec['image']['npy']
        for name, a in (('state', state), ('proprio', obs['proprio'])):
            rec['files'][name+'.npy'] = store.save(stem+'/'+name+'.npy', npy(np.asarray(a)))
        rec['files']['transition.json'] = store.json(stem+'/transition.json', base.numeric_json(transition))
        rec['files']['full_public_state.json'] = store.json(stem+'/full_public_state.json', base.public_state(envs[role]))
        if frame == (0 if role == 'goal' else 20):
            view = meter.render_views[id(envs[role])]
            shot = row['shots'][0]
            shot['files'] = dict(rgb=rec['image']['npy'], png=rec['image']['png'],
                before=store.json(role+'/baseline_before.json', view['before']),
                after=store.json(role+'/baseline_after.json', view['after']))
            shot['status'] = 'captured'
            require(view['before'] == view['after'] == base.public_state(envs[role]), 'baseline actual native render invariance')
        for name in NAMES:
            byte_gate(store.out/rec['files'][name]['path'], p['frames'][role][frame][name])
        store.event(dict(phase='frame_byte_gated', state_id=role, frame=frame))
        return rec
    def step(role, frame, raw):
        env = envs[role]; row = rows[STATES.index(role)]
        previous = row['records'][-1]['files']['full_public_state.json']['sha256']
        pos = np.asarray(env.agent.position, np.float64)
        u = np.asarray(raw, np.float64)
        command = dict(frame=frame, raw_env_argument=u.tolist(), dtype=u.dtype.str,
            agent_xy_before=pos.tolist(), absolute_target=(pos+100*u).tolist(),
            pre_public_sha256=previous, expected_substeps=10, dt=.01)
        store.json(role+f'/command_{frame:02d}.json', command)
        obs, reward, done, info = operation(store, meter, role, 'step', command, lambda: env.step(u.copy()))
        capture(role, frame, obs, info['state'], dict(kind='step', reward=reward, done=done, info=info))
        require(done is False, 'native terminal honored; no padding')
    def construct(role, ctor, seed):
        def create():
            envs[role] = cls(**ctor)
            return envs[role]
        env = operation(store, meter, role, 'construct', dict(constructor=ctor), create)
        store.json(role+'/constructor_rng.json', base.rng_state(env))
        operation(store, meter, role, 'seed', dict(seed=seed), lambda: env.seed(seed))
        store.json(role+'/pre_reset_rng.json', base.rng_state(env))
        obs, state = operation(store, meter, role, 'reset', {}, env.reset)
        capture(role, 0, obs, state, dict(kind='independent_goal_scene_reset' if role == 'goal' else 'reset'))
    try:
        cls, pymunk, pygame, base, probe = loader()
        store.json('runtime_probe.json', probe)
        meter = Meter(cls, pymunk, store, base.public_state)
        require(sys.getprofile() is None, 'do not replace existing profiler')
        sys.setprofile(meter.profile); profiled = True
        for plan in p['plans']:
            role = plan['arm']; construct(role, plan['constructor'], plan['seed'])
            for i, u in enumerate(plan['prefix_raw_u'], 1):
                step(role, i, u)
        store.event(dict(phase='all_prefixes_byte_gated'))
        for plan in p['plans']:
            for i, u in enumerate(plan['reference_suffix_raw_u'], 11):
                step(plan['arm'], i, u)
        gp = p['goal_protocol']; construct('goal', gp['constructor'], gp['seed'])
        scene = load_bound(SCENE, SCENE_SHA)
        geometry = operation(store, meter, 'goal', 'geometry', {}, lambda: scene.geometry_witness(envs['goal']))
        store.json('goal/geometry.json', geometry)
        byte_gate(store.out/'goal/geometry.json', p['goal_geometry'])
        for row, source in zip(rows, p['states']):
            rec = row['records'][-1]
            require(rec['files']['rgb.npy']['sha256'] == source['rgb']['sha256']
                    and read(store.out/rec['files']['full_public_state.json']['path']) == source['source_state'], 'five original baseline identities')
            row['baseline'] = rec; row['status'] = 'baseline_passed'
        gate = True; store.event(dict(phase='all_five_baselines_byte_gated'))
        for row in rows:
            for j, cell in enumerate(row['cells'], 1):
                render_color(envs[row['state_id']], cell['color'], pygame, base, store, meter, row['state_id'], cell)
                row['shots'][j].update(status='captured', files=dict(rgb=cell['image']['npy'],
                    png=cell['image']['png'], before=cell['before'], after=cell['after'], restored=cell['restored']))
            row['status'] = 'completed'
        stable()
    except BaseException as exc:
        errors.append(dict(phase='execute', error=type(exc).__name__+': '+str(exc)))
    finally:
        for role, env in envs.items():
            try:
                operation(store, meter, role, 'close', {}, env.close, cleanup=True)
            except BaseException as exc:
                errors.append(dict(phase='close', state_id=role, error=type(exc).__name__+': '+str(exc)))
        if profiled:
            sys.setprofile(None)
    receipt = meter.receipt() if meter else None
    if receipt and (receipt['entries'] != EXPECTED or receipt['return_events'] != EXPECTED or receipt['errors']):
        errors.append(dict(phase='census', error='native complete expected census not met'))
    try:
        store.event(dict(phase='final', errors=errors), final=True)
    except BaseException as exc:
        errors.append(dict(phase='final_journal', error=type(exc).__name__+': '+str(exc)))
    try:
        store.journal.close()
    except BaseException as exc:
        errors.append(dict(phase='journal_close', error=type(exc).__name__+': '+str(exc)))
    try:
        store.files['journal.jsonl'] = dict(path='journal.jsonl', sha256=sha(store.out/'journal.jsonl'),
                                          bytes=(store.out/'journal.jsonl').stat().st_size)
    except BaseException as exc:
        errors.append(dict(phase='journal_hash', error=type(exc).__name__+': '+str(exc)))
    report = dict(schema=NAME+'.report', prepare_sha256=prepared_sha, prepare_digest=digest(p),
        status='replay_targets_recorded' if not errors else 'partial_engineering_failure',
        protocol=protocol(), rows=rows, all_baselines_passed=gate, errors=errors, meter=receipt,
        artifacts=store.files.copy(), journal_terminal_sha256=store.previous, journal_events=store.sequence,
        wall_seconds=time.monotonic()-store.started, scientific_admission=False, task_success=None,
        profiler_released=not profiled or sys.getprofile() is None)
    try:
        store.json('report.json', report, final=True)
    except BaseException as exc:
        # Returned to MAIN/CLI even when full disk/IO failure prevents durable report.
        report['errors'].append(dict(phase='report_save', error=type(exc).__name__+': '+str(exc)))
        report['status'] = 'partial_engineering_failure'
    return report


def run(path, expected, out, authorized):
    require(authorized, 'MAIN explicit 805-step physics authorization required')
    p = verify(path, expected)
    out = Path(out).absolute()
    require(out.parent == ROOT/'results' and out.name.startswith(NAME+'_'), 'own result child')
    require(signal.getsignal(signal.SIGALRM) == signal.SIG_DFL
            and signal.getitimer(signal.ITIMER_REAL) == (0., 0.), 'no existing alarm')
    # Work deadline is enforced at every durable dispatch/write. One-shot asynchronous
    # alarm interrupts long native operations; cleanup keeps a separate byte reserve.
    def timeout(*args):
        raise TimeoutError('900s job work cap')
    signal.signal(signal.SIGALRM, timeout); signal.alarm(WALL)
    try:
        v3 = load_bound(V3, V3_SHA)
        result = execute(p, expected, out, lambda: v3.load_runtime(read(p['v3_prepare'], p['v3_sha256'])),
                         lambda: verify(path, expected))
        return result
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, signal.SIG_DFL)


def consume(path, expected, report_path, report_sha):
    """Saved-only strict rederivation. Partial evidence is retained but never released."""
    from PIL import Image
    p = verify(path, expected); report_path = Path(report_path).absolute()
    r = read(report_path, report_sha); out = report_path.parent
    require(r['schema'] == NAME+'.report' and r['prepare_sha256'] == expected
            and r['prepare_digest'] == digest(p) and r['protocol'] == protocol(), 'execution identity')
    require(r['status'] == 'replay_targets_recorded' and not r['errors'] and r['all_baselines_passed']
            and r['profiler_released'] and r['scientific_admission'] is False, 'complete engineering release only')
    require([x['state_id'] for x in r['rows']] == list(STATES), 'five fixed rows')
    actual_files = {str(f.relative_to(out)) for f in out.rglob('*') if f.is_file()}
    require(all(not f.is_symlink() and (f.is_file() or f.is_dir()) for f in out.rglob('*')), 'regular output tree only')
    require(actual_files == set(r['artifacts']) | {'report.json'}, 'complete artifact census')
    require(sum(f.stat().st_size for f in out.rglob('*') if f.is_file()) <= CAP, 'actual output cap')
    for name, rec in r['artifacts'].items():
        require(Path(name).as_posix() == name and not Path(name).is_absolute() and '..' not in Path(name).parts,
                'local artifact path')
        require(rec == dict(path=name, sha256=sha(out/name), bytes=(out/name).stat().st_size), 'artifact bytes')
    binding = read(out/'execution_binding.json')
    require(binding == dict(schema=NAME+'.execution', prepare_sha256=expected, prepare_digest=digest(p), protocol=protocol()), 'saved execution edge')
    probe = read(out/'runtime_probe.json'); v3 = load_bound(V3, V3_SHA)
    v = read(p['v3_prepare'], p['v3_sha256'])
    v3.check_origins(v, *[v['origins'][k] for k in ('environment', 'pymunk', 'pygame')], probe)
    byte_gate(out/'goal/geometry.json', p['goal_geometry'])
    events = []; previous = None
    for i, line in enumerate((out/'journal.jsonl').read_bytes().splitlines()):
        row = json.loads(line); h = row.pop('entry_sha256')
        require(row['sequence'] == i and row['previous_sha256'] == previous and digest(row) == h, 'journal chain')
        previous = h; events.append(row['event'])
    require(len(events) == r['journal_events'] and previous == r['journal_terminal_sha256'], 'terminal journal anchor')
    allowed = {'intent','returned','failed','native_call','native_return','frame_byte_gated',
               'all_prefixes_byte_gated','all_five_baselines_byte_gated','final'}
    require(all(e['phase'] in allowed for e in events) and events[-1] == dict(phase='final',errors=[])
            and Counter(e['phase'] for e in events)['final'] == 1
            and Counter(e['phase'] for e in events)['frame_byte_gated'] == 85
            and Counter(e['phase'] for e in events)['all_prefixes_byte_gated'] == 1
            and Counter(e['phase'] for e in events)['all_five_baselines_byte_gated'] == 1, 'fixed journal phase census')
    entries = Counter(e['name'] for e in events if e['phase'] == 'native_call')
    returns = Counter(e['name'] for e in events if e['phase'] == 'native_return')
    require(dict(entries) == EXPECTED and dict(returns) == EXPECTED and r['meter'] == dict(entries=EXPECTED, return_events=EXPECTED, errors=[]), 'native census independently from journal')
    require(all(e['dt'] == .01 for e in events if e['phase'] == 'native_call' and e['name'] == 'space_step'), 'all805 dt')
    pending = None; public = Counter(); gated = False; prefix = False; frames = []
    native_in = Counter(); native_out = Counter(); dispatches = []
    for e in events:
        phase = e['phase']
        if phase == 'intent':
            require(pending is None, 'no overlapping public dispatch')
            pending = (e['state_id'], e['operation']); public[e['operation']] += 1
            dispatches.append((e['state_id'],e['operation'],e['details'].get('frame'),e['details'].get('color')))
            native_in.clear(); native_out.clear()
            role = e['state_id']; op = e['operation']
            require(role in STATES, 'known exact operation role')
            config = p['goal_protocol'] if role == 'goal' else p['plans'][STATES.index(role)]
            if op == 'construct':
                require(e['details'] == dict(constructor=config['constructor']), 'original constructor dispatch')
            elif op == 'seed':
                require(e['details'] == dict(seed=config['seed']), 'original explicit seed dispatch')
            elif op == 'geometry':
                require(role == 'goal', 'only original goal extra geometry body')
            if e['operation'] == 'color':
                require(gated, 'all baselines before first color')
            if e['operation'] == 'step':
                role = e['state_id']; n = e['details']['frame']
                plan = p['plans'][STATES.index(role)]
                raw = (plan['prefix_raw_u']+plan['reference_suffix_raw_u'])[n-1]
                require(canonical(e['details']['raw_env_argument']) == canonical(raw), 'journal original raw action including signed zero')
                require(n <= 10 or prefix, 'all prefixes gate before suffix')
        elif phase == 'returned':
            require(pending == (e['state_id'], e['operation']), 'paired durable return')
            require(dict(native_in) == DELTAS[e['operation']] and dict(native_out) == DELTAS[e['operation']], 'each dispatch actual native delta')
            pending = None
        elif phase in ('native_call','native_return'):
            require(pending is not None and (e['context']['state_id'],e['context']['operation']) == pending, 'native operation context')
            (native_in if phase == 'native_call' else native_out)[e['name']] += 1
        elif phase == 'failed':
            require(False, 'failed dispatch not complete')
        elif phase == 'frame_byte_gated':
            frames.append((e['state_id'], e['frame']))
        elif phase == 'all_prefixes_byte_gated':
            require(set(frames) == {(s, n) for s in STATES[:4] for n in range(11)}, 'all44prefixes')
            prefix = True
        elif phase == 'all_five_baselines_byte_gated':
            require(set(frames) == {(s, n) for s in STATES[:4] for n in range(21)} | {('goal', 0)}, 'all85frames before gate')
            gated = True
    require(pending is None and dict(public) == protocol()['public_operations'], 'full public operation census')
    wanted = []
    for role in STATES[:4]:
        wanted.extend((role,op,None,None) for op in ('construct','seed','reset'))
        wanted.extend((role,'step',n,None) for n in range(1,11))
    for role in STATES[:4]:
        wanted.extend((role,'step',n,None) for n in range(11,21))
    wanted.extend(('goal',op,None,None) for op in ('construct','seed','reset','geometry'))
    wanted.extend((role,'color',None,color) for role in STATES for color in COLORS)
    wanted.extend((role,'close',None,None) for role in STATES)
    require(dispatches == wanted, 'original exact phase/role/frame/color dispatch sequence')
    def image(rec):
        a = np.load(out/rec['npy']['path'], allow_pickle=False)
        require(a.shape == (224,224,3) and a.dtype == np.uint8, 'RGB224')
        with Image.open(out/rec['png']['path']) as im:
            require(im.mode == 'RGB' and np.array_equal(np.asarray(im), a), 'PNG pixels exact native NPY')
        require(hashlib.sha256(a.tobytes()).hexdigest() == rec['pixel_sha256'], 'pixel receipt')
        for k in ('npy','png'):
            require(rec[k] == r['artifacts'][rec[k]['path']], 'image inventory edge')
        return a
    for row, source in zip(r['rows'], p['states']):
        role = row['state_id']; require(row['status'] == 'completed', 'completed row')
        ns = range(1) if role == 'goal' else range(21)
        require([f['frame'] for f in row['records']] == list(ns), 'full frame denominator')
        for n, frame in enumerate(row['records']):
            image(frame['image'])
            require(frame['image']['npy'] == frame['files']['rgb.npy'], 'frame image/native return edge')
            for name in NAMES:
                rec = frame['files'][name]
                require(rec['path'] == role+f'/frame_{n:02d}/'+name, 'exact frame artifact identity')
                require(rec == r['artifacts'][rec['path']], 'frame inventory edge')
                byte_gate(out/rec['path'], p['frames'][role][n][name])
            if n:
                command = read(out/role/f'command_{n:02d}.json')
                before = read(out/row['records'][n-1]['files']['full_public_state.json']['path'])
                pos = np.asarray(before['bodies']['agent']['position'], np.float64)
                plan = p['plans'][STATES.index(role)]; u = np.asarray((plan['prefix_raw_u']+plan['reference_suffix_raw_u'])[n-1], np.float64)
                want = dict(frame=n,raw_env_argument=u.tolist(),dtype='<f8',agent_xy_before=pos.tolist(),
                    absolute_target=(pos+100*u).tolist(),pre_public_sha256=row['records'][n-1]['files']['full_public_state.json']['sha256'],expected_substeps=10,dt=.01)
                require(canonical(command) == canonical(want), 'exact executable action witness')
        require(row['baseline'] == row['records'][-1], 'baseline points to real final observation')
        require(row['baseline']['files']['rgb.npy']['sha256'] == source['rgb']['sha256'], 'original baseline NPY bytes')
        require([s['shot'] for s in row['shots']] == ['baseline', *COLORS]
                and all(s['status'] == 'captured' for s in row['shots']), 'fixed baseline/visible/white shots')
        baseline_shot = row['shots'][0]['files']
        require(baseline_shot['rgb'] == row['baseline']['image']['npy']
                and baseline_shot['png'] == row['baseline']['image']['png'], 'baseline shot native image edge')
        for key in ('before','after'):
            require(baseline_shot[key] == r['artifacts'][role+'/baseline_'+key+'.json']
                    and read(out/baseline_shot[key]['path']) == source['source_state'], 'actual baseline native render state')
        require([c['color'] for c in row['cells']] == list(COLORS), 'two fixed colors')
        for j, cell in enumerate(row['cells'], 1):
            require(cell['status'] == 'rendered' and cell['requested_rgba'] == RGBA[cell['color']], 'complete native color')
            a = image(cell['image'])
            require(cell['image']['npy']['path'] == role+'/'+cell['color']+'.npy'
                    and cell['image']['png']['path'] == role+'/'+cell['color']+'.png', 'fixed cell image identity')
            for key in ('before','after','restored'):
                require(cell[key] == r['artifacts'][role+'/'+cell['color']+'_'+key+'.json'], 'invariance receipt inventory edge')
            require(row['shots'][j]['files'] == dict(rgb=cell['image']['npy'],png=cell['image']['png'],
                before=cell['before'],after=cell['after'],restored=cell['restored']), 'shot/cell exact identity')
            before, after, restored = [read(out/cell[k]['path']) for k in ('before','after','restored')]
            require(before == after == source['source_state'] == restored['public_state'], 'exact original public state/RNG/goal retained')
            require(restored['same_color_object'] and restored['topology_unchanged'] and restored['color'] == cell['original_color'] == RGBA['visible'], 'native original color restored')
            if cell['color'] == 'visible':
                require(np.array_equal(a, np.load(source['rgb']['path'], allow_pickle=False)), 'live LightGreen contact pixels baseline exact')
    require(0 <= r['wall_seconds'] <= WALL, 'recorded complete work wall')
    return dict(schema=NAME+'.receipt', report_sha256=report_sha, prepare_sha256=expected,
                status='strict_saved_complete', states=5, cells=10, native_entries=EXPECTED,
                public_operations=dict(public),png=95, scientific_admission=False, task_success=None)


def main():
    a = argparse.ArgumentParser(); sub = a.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--output', required=True)
    for cmd in ('verify','run','consume'):
        p = sub.add_parser(cmd); p.add_argument('--prepare', required=True); p.add_argument('--prepare-sha', required=True)
        if cmd == 'run':
            p.add_argument('--output', required=True); p.add_argument('--main-authorized', action='store_true')
        if cmd == 'consume':
            p.add_argument('--report', required=True); p.add_argument('--report-sha', required=True); p.add_argument('--output', required=True)
    x = a.parse_args()
    if x.command == 'prepare':
        result = build(); payload = canonical(result)+b'\n'
        require(len(payload) <= 2*1024**2 and shutil.disk_usage(Path(x.output).absolute().parent).free >= FLOOR+len(payload), 'small prepare bound plus floor')
        new(x.output, payload)
    elif x.command == 'verify':
        p = verify(x.prepare, x.prepare_sha); result = dict(status='verified_not_executed', prepare_digest=digest(p))
    elif x.command == 'run':
        result = run(x.prepare,x.prepare_sha,x.output,x.main_authorized)
    else:
        result = consume(x.prepare,x.prepare_sha,x.report,x.report_sha); new(x.output,canonical(result)+b'\n')
    print(canonical(result if x.command != 'prepare' else dict(status='prepared_no_physics',sha256=sha(x.output))).decode())
    return 2 if result.get('errors') else 0


if __name__ == '__main__':
    raise SystemExit(main())
