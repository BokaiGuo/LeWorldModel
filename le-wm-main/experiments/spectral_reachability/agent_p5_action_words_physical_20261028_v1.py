"""MAIN-only exact native replay of all 3 x 4 frozen P5 words.

prepare/verify/consume: saved NumPy/stdlib only. run: explicit MAIN physics
authorization, original isolated native runtime; no model import or execution.
Native goal observations are NOT an independently certified task witness.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

import numpy as np

TAG = 'agent_p5_action_words_physical_20261028_v1'
ROOT = Path(__file__).resolve().parents[2]
PRODUCER = 'agent_p5_native_action_words_20261028_v2'
PANEL = 'p5_jepa_factual_panel_20261027_v3'
SEEDS = (2026102702, 2026102703, 2026102704)
WORDS = ('xy', 'yx', 'x_minus_x', 'zero_zero')
NAMES = ('rgb.npy', 'state.npy', 'proprio.npy', 'full_public_state.json', 'transition.json')
PINNED = {
    f'experiments/spectral_reachability/{PRODUCER}.py': '5dc5d158ff817d9f39dca0f82ea9eb562f0f9566cc3e74a2abfd2be1af4d74ce',
    'experiments/spectral_reachability/consume_p5_words_MAIN_20261028_v1.py': 'ddbbe56a263e4311ca042862da21f73e4db6a5107bf865914e3a9cb2a3061cad',
    f'results/{PRODUCER}_prepare_MAIN01/manifest.json': '1515cb228c6ab0a4fe826074324fd566781b06812a1e5163c41652492518a905',
    f'results/{PRODUCER}_cuda_MAIN01/report.json': 'cdbccbdf894d8d6e7c078e02f7078bd8cd26e802c9a00d59421ad0e4b0883bbd',
    'results/consume_p5_words_MAIN_20261028_v1_MAIN01.json': '5237000615227a5cb6741e19c3feb4414c710077983399c46804b96c223b01db',
    f'experiments/spectral_reachability/{PANEL}.py': '0d4f1d983ab46f78bc21b79161de1105ed92f9e6ca2c910ab014667bb30019ef',
    f'results/{PANEL}/manifest_frozen.json': '67b719d332a9405b6f4f6dd1adfdf2f5bdbd5c2e8bede1e16d73360a6470d516',
}
MIB = 1024**2
PAYLOAD, CAP, RESERVE, NODES, FLOOR, WALL = 96*MIB, 128*MIB, 8*MIB, 2048, 200*MIB, 900
EXPECTED = dict(env_construct=1, seed=2, reset=1, control_step=20,
                space_step=201, native_render=21, native_get_obs=21, close=1,
                setup=1, set_state=1, space_construct=1, native_goal_body=41,
                body_construct=44)


def require(ok, message):
    if not ok:
        raise ValueError(message)


class IntegrityFailure(ValueError):
    """Identity failure stops later dispatch, never substitutes a prefix."""


class NativeDone(ValueError):
    """Keep actual early endpoint, no padding/retry."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    path = Path(path)
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
            'regular nonsymlink file required: '+str(path))
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(MIB), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path, expected=None):
    require(expected is None or sha(path) == expected, 'JSON SHA mismatch: '+str(path))
    sha(path)
    require(Path(path).stat().st_size <= 8*MIB, 'bounded JSON required')
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, 'duplicate JSON key')
            out[key] = value
        return out
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda s: require(False, 'nonfinite JSON '+s))


def import_bound(path, expected, tag):
    require(sha(path) == expected, 'source SHA mismatch: '+str(path))
    spec = importlib.util.spec_from_file_location(TAG+'_'+tag, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def policy():
    return dict(seeds=list(SEEDS), words=list(WORDS), denominator=12,
        replay='fresh constructor, explicit seed, reset, original prefix10; no warmup/setter/restore',
        prefix_gate='all five files byte-exact at every frame0..10, before suffix',
        suffix='all four fixed words/root, raw float64[10,2], never inverse-normalized',
        frame_checkpoints=[0, 5, 10, 15, 20], saved_frames=list(range(21)),
        with_target=True, goal_setter_calls=0, selection=False, retry=False,
        expected_per_branch=EXPECTED, expected_total={k:12*v for k,v in EXPECTED.items()},
        output_cap_bytes=CAP, payload_cap_bytes=PAYLOAD, final_reserve_bytes=RESERVE,
        max_nodes=NODES, overhead_bytes_per_node=8192, free_floor_bytes=FLOOR, wall_seconds=WALL,
        native_meter='actual Python call entries + Space.step dt/context, not completed C steps; intent/return separate',
        model_execution=False, scientific_admission=False, P5_admission=False,
        action_witness_established=False, task_success=None,
        blockers=['independent task-goal/protocol certification absent',
                  'predeclared contact/action-task witness acceptance absent'],
        scope='known-root physical word replay only; no goal or success gate invented; not fresh')


def dependencies(root):
    for rel, wanted in PINNED.items():
        require(sha(root/rel) == wanted, 'pinned actual input/source drift: '+rel)
    panel = import_bound(root/f'experiments/spectral_reachability/{PANEL}.py',
                         PINNED[f'experiments/spectral_reachability/{PANEL}.py'], 'panel')
    v2, base, _ = panel.bound(root)
    pm = read_json(root/f'results/{PANEL}/manifest_frozen.json')
    seals = dict(pm['seals'])
    for path, wanted in seals.items():
        require(sha(path) == wanted, 'original native helper/source drift')
    return panel, v2, base, seals


def branches_from(producer_manifest, base, schedule, seals):
    """Saved-only assembler. No endpoint, forecast, Q, contact filtering."""
    require([r['seed'] for r in producer_manifest['roots']] == list(SEEDS), 'three fixed roots')
    prefix = np.asarray(schedule, np.float64).reshape(15, 2)[:10]
    branches = []
    for root in producer_manifest['roots']:
        factual = Path(root['factual'])/'a'
        identity = []
        for frame in range(11):
            files = {}
            for name in NAMES:
                path = factual/f'frame_{frame:02d}'/name
                h = sha(path)
                require(seals.get(str(path)) == h, 'original prefix absent from producer closure')
                files[name] = dict(path=str(path), sha256=h, bytes=path.stat().st_size)
            identity.append(dict(frame=frame, files=files))
        commands = []
        for frame in range(1, 11):
            path = factual/f'command_{frame:02d}.json'
            require(seals.get(str(path)) == sha(path), 'original command outside closure')
            command = read_json(path)
            require(command['raw_env_argument'] == prefix[frame-1].tolist(), 'original schedule mismatch')
            commands.append(dict(path=str(path), sha256=sha(path)))
        rng_path = factual/'pre_reset_rng.json'
        require(seals.get(str(rng_path)) == sha(rng_path), 'original seeded RNG outside closure')
        goal_path = factual/'frame_10/full_public_state.json'
        require(root['goal_source'] == dict(path=str(goal_path), sha256=sha(goal_path)), 'source goal binding')
        native_goal = read_json(goal_path)['env']['goal_pose']
        require(set(root['plans']) == set(WORDS), 'all fixed words')
        for word in WORDS:
            suffix = np.asarray(root['plans'][word]['raw'], np.float64)
            require(suffix.shape == (2,5,2) and np.isfinite(suffix).all(), 'two raw packets')
            branches.append(dict(case=f"seed_{root['seed']}_{word}", seed=root['seed'], word=word,
                root_sha256=root['root_sha256'], goal_source=root['goal_source'], native_goal=native_goal,
                prefix=prefix.tolist(), suffix=suffix.reshape(10,2).tolist(), prefix_identity=identity,
                prefix_commands=commands, seeded_rng=dict(path=str(rng_path), sha256=sha(rng_path))))
    return branches


def prepare(root=ROOT):
    panel, v2, base, seals = dependencies(root)
    ppath = root/f'experiments/spectral_reachability/{PRODUCER}.py'
    producer = import_bound(ppath, PINNED[str(ppath.relative_to(root))], 'producer')
    mp = root/f'results/{PRODUCER}_prepare_MAIN01/manifest.json'
    rp = root/f'results/{PRODUCER}_cuda_MAIN01/report.json'
    cp = root/'results/consume_p5_words_MAIN_20261028_v1_MAIN01.json'
    pm = read_json(mp)
    # Recompute exact owner return with saved arrays only; never import CUDA runner.
    actual = producer.consume(pm, rp, PINNED[str(rp.relative_to(root))])
    require(read_json(cp) == actual and actual['status'] == 'all12_saved_program_predictions_verified',
            'actual strict consumer receipt differs from exact producer.consume')
    require(actual['physical_release'] is False, 'old model receipt must not be relabelled physics authority')
    seals.update(pm['bindings'])
    seals.update({str(root/p):h for p,h in PINNED.items()})
    branches = branches_from(pm, base, v2.SCHEDULE, seals)
    for path in (Path(__file__).resolve(), root/'tests'/f'test_{TAG}.py', root/'innerdocs'/f'{TAG}.md'):
        seals[str(path)] = sha(path)
    for path,h in seals.items():
        require(sha(path) == h, 'input/source closure drift')
    runtime_path, runtime = base.runtime_manifest(root)
    require(all(str(root/runtime_rel) in seals for runtime_rel in
                ['experiments/spectral_reachability/p5_jepa_native_physics_runtime_20261027_v1.py']),
            'native runtime source outside inherited closure')
    return dict(schema=TAG, status='prepared_MAIN_native_replay_only', policy=policy(),
        bindings=seals, constructor=base.CTOR, branches=branches,
        runtime_manifest=dict(path=str(runtime_path), sha256=sha(runtime_path)),
        native_source=dict(path=str(root/'external/jepa-wms/evals/simu_env_planning/envs/pusht_env/pusht_env.py'),
                           sha256=seals[str(root/'external/jepa-wms/evals/simu_env_planning/envs/pusht_env/pusht_env.py')]),
        upstream_consume=actual, physics_executed=False, neural_executed=False,
        scientific_admission=False, task_success=None)


def verify(manifest):
    require(manifest == prepare(), 'physical prepare/source/authority drift')


class Writer:
    """Exact payload + conservative node cap, floor and wall guard. Never overwrite."""
    def __init__(self, base, anchor, clock=time.monotonic, free=None):
        self.base, self.anchor, self.clock = base, Path(anchor), clock
        self.free = free or (lambda: shutil.disk_usage(self.anchor).free)
        self.started = clock()
        self.payload = self.nodes = 0

    def guard(self):
        require(self.clock()-self.started < WALL, 'wall cap; no retry')
        require(self.free() >= FLOOR, 'free disk floor; no dispatch')

    def charge(self, size, nodes=0, final=False):
        if not final:
            self.guard()
        require(self.payload+size <= PAYLOAD-(0 if final else RESERVE), 'payload cap')
        require(self.nodes+nodes <= NODES-(0 if final else 64), 'node cap')
        require(self.payload+size+8192*(self.nodes+nodes) <= CAP, 'accounted disk cap')
        # Intentional: failed writes are charged, never refunded.
        self.payload += size
        self.nodes += nodes

    def mkdir(self, path):
        self.charge(0, 1)
        self.base.mkdir(path)

    def save_json(self, path, value, final=False):
        self.charge(len(self.base.canonical(value))+1, 1, final)
        self.base.save_json(path, value)

    def save_array(self, path, value):
        value = np.asarray(value)
        require(value.dtype.kind in 'biuf' and value.nbytes < 200000, 'bounded numeric nonpickle array')
        buffer = io.BytesIO()
        np.save(buffer, value, allow_pickle=False)
        self.charge(buffer.tell(), 1)
        self.base.save_array(path, value)

    def journal(self, path):
        self.charge(0, 1)
        return Journal(self, self.base.Journal(path))


class Journal:
    def __init__(self, writer, native):
        self.writer, self.native = writer, native
        self.attempted, self.returned = Counter(), Counter()

    def append(self, event, final=False):
        row = dict(sequence=self.native.sequence, previous_sha256=self.native.previous,
                   event=event, entry_sha256='0'*64)
        self.writer.charge(len(self.writer.base.canonical(row))+1, final=final)
        result = self.native.append(event)
        if event.get('phase') == 'intent':
            self.attempted[event['operation']] += 1
        if event.get('phase') == 'returned':
            self.returned[event['operation']] += 1
        return result

    def close(self):
        self.native.close()


class FinalJournal:
    def __init__(self, journal):
        self.journal = journal

    def append(self, event):
        return self.journal.append(event, final=True)


def prefix_gate(path, expected):
    for name in NAMES:
        saved = expected['files'][name]
        a, b = Path(path)/name, Path(saved['path'])
        if (sha(a) != saved['sha256'] or sha(b) != saved['sha256']
                or a.stat().st_size != saved['bytes'] or saved['bytes'] > 200000
                or a.read_bytes() != b.read_bytes()):
            raise IntegrityFailure('prefix byte mismatch: '+str(expected['frame'])+'/'+name)


def command_for(frame, u, position, previous):
    return dict(frame=frame, action_index=frame-1, packet_index=(frame-1)//5,
        step_within_packet=(frame-1)%5, raw_env_argument=u.tolist(),
        pixel_displacement=(100*u).tolist(), agent_xy_before=position.tolist(),
        absolute_target=(position+100*u).tolist(), expected_substeps=10, dt=.01,
        pre_public_sha256=previous)


def collect(panel, base, writer, cls, branch, dest, journal, calls, row):
    env = None
    close = row['close']
    try:
        def construct():
            nonlocal env
            close['status'] = 'construction_attempted'
            env = cls(**base.CTOR)
            close['status'] = 'required'
            return env
        base.operation(journal, calls, branch['case'], 'construct', dict(config=base.CTOR), construct)
        writer.save_json(dest/'constructor_rng.json', base.rng_state(env))
        base.operation(journal, calls, branch['case'], 'seed', dict(seed=branch['seed']), lambda:env.seed(branch['seed']))
        writer.save_json(dest/'pre_reset_rng.json', base.rng_state(env))
        require(sha(dest/'pre_reset_rng.json') == branch['seeded_rng']['sha256'], 'seeded RNG identity')
        obs, state = base.operation(journal, calls, branch['case'], 'reset',
            dict(seed=branch['seed'], expected_space_steps=1, dt=.01), env.reset)
        record = panel.capture(base, writer, env, obs, state, dict(kind='reset'), dest/'frame_00')
        row['records'].append(record)
        prefix_gate(dest/'frame_00', branch['prefix_identity'][0])
        row['prefix_gates'].append(0)
        journal.append(dict(phase='prefix_byte_gate', frame=0))
        commands = np.concatenate((np.asarray(branch['prefix'], np.float64), np.asarray(branch['suffix'], np.float64)))
        require(commands.shape == (20,2) and np.isfinite(commands).all(), 'frozen raw20')
        for frame, u in enumerate(commands, 1):
            if frame > 10:
                require(row['prefix_gates'] == list(range(11)), 'all prefix gates BEFORE suffix')
            position = np.array(env.agent.position, np.float64)
            command = command_for(frame, u, position, record['files']['full_public_state.json']['sha256'])
            writer.save_json(dest/f'command_{frame:02d}.json', command)
            if frame <= 10 and sha(dest/f'command_{frame:02d}.json') != branch['prefix_commands'][frame-1]['sha256']:
                raise IntegrityFailure('original prefix command mismatch')
            obs, reward, done, info = base.operation(journal, calls, branch['case'], 'step', command, lambda:env.step(u.copy()))
            record = panel.capture(base, writer, env, obs, info['state'],
                                   dict(kind='step', reward=reward, done=done, info=info), dest/f'frame_{frame:02d}')
            row['records'].append(record)
            if frame <= 10:
                prefix_gate(dest/f'frame_{frame:02d}', branch['prefix_identity'][frame])
                row['prefix_gates'].append(frame)
                journal.append(dict(phase='prefix_byte_gate', frame=frame))
            if done:
                raise NativeDone('native done frame '+str(frame)+'; no padding')
    finally:
        if env is not None:
            original = sys.exc_info()[1]
            close['original_error'] = None if original is None else type(original).__name__+': '+str(original)
            def close_call():
                close['attempted'] = True
                env.close()
                close['returned'] = True
            try:
                base.operation(FinalJournal(journal), calls, branch['case'], 'close', {}, close_call)
                close.update(status='completed', durable_return=True)
            except Exception as exc:
                close.update(status='failed', error=type(exc).__name__+': '+str(exc))
                if original is None:
                    raise


def native_metrics(public):
    """Native quantities only; no task-success gate/readout/threshold fitting."""
    block, agent = public['bodies']['block'], public['bodies']['agent']
    goal = np.asarray(public['env']['goal_pose']['values'], np.float64)
    coverage = public['env']['coverage_arr']
    angle = float(block['angle'])-goal[2]
    return dict(agent_xy=agent['position'], agent_velocity=agent['velocity'],
        block_xy=block['position'], block_angle_native_rad=block['angle'], native_goal_pose=goal.tolist(),
        block_xy_native_goal_distance=float(np.linalg.norm(np.asarray(block['position'])-goal[:2])),
        block_angle_native_goal_distance=float(abs(np.arctan2(np.sin(angle), np.cos(angle)))),
        native_final_coverage=coverage[-1] if coverage else None,
        native_max_coverage=max(coverage) if coverage else None, native_contact_points=public['env']['n_contact_points'],
        native_success_threshold=public['env']['success_threshold'], task_success=None)


def inventory(out):
    records = {}
    nodes = 0
    size = 0
    for p in sorted(Path(out).rglob('*')):
        require(not p.is_symlink(), 'output symlink')
        nodes += 1
        if p.is_file():
            size += p.stat().st_size
            if p != Path(out)/'report.json':
                records[str(p.relative_to(out))] = dict(sha256=sha(p), bytes=p.stat().st_size)
    require(size+8192*(nodes+1) <= CAP and nodes+1 <= NODES, 'saved accounted output cap')
    return records


def check_calls(receipt, case):
    require(receipt['python_call_entries'] == EXPECTED, 'native call-entry census')
    wanted = [dict(context=dict(role=case, operation='reset', frame=None), dt=.01)]
    for frame in range(1,21):
        wanted += [dict(context=dict(role=case, operation='step', frame=frame), dt=.01)]*10
    require(receipt['space_step_events'] == wanted, 'native dt/context census')


def execute(manifest, manifest_sha, out, panel, base, load, verify_sources):
    """RAM-testable execution seam. Production load is sealed native-only."""
    require(manifest['policy'] == policy() and len(manifest['branches']) == 12, 'fixed replay policy')
    out = Path(out)
    require(out.is_absolute() and not out.exists() and out.parent.is_dir(), 'new output child required')
    require(not any(p.is_symlink() for p in (out.parent,*out.parent.parents)), 'output parent symlink')
    writer = Writer(base, out.parent)
    writer.mkdir(out)
    rows = [dict(case=b['case'], seed=b['seed'], word=b['word'], root_sha256=b['root_sha256'],
                 status='not_run', error=None, records=[], prefix_gates=[], meter=None,
                 close=dict(status='not_constructed', attempted=False, returned=False,
                            durable_return=False, original_error=None, error=None)) for b in manifest['branches']]
    writer.save_json(out/'dispatch.json', dict(manifest_sha256=manifest_sha, branches=manifest['branches'], policy=policy()))
    error = None
    try:
        cls, meter_factory, probe = load()
        writer.save_json(out/'runtime_probe.json', probe)
    except Exception as exc:
        error = 'runtime preflight: '+type(exc).__name__+': '+str(exc)
    for branch, row in zip(manifest['branches'], rows):
        calls = None
        journal = None
        dest = out/branch['case']
        try:
            if error is not None:
                row['error'] = error
                continue
            verify_sources()
            writer.mkdir(dest)
            journal = writer.journal(dest/'journal.jsonl')
            journal.append(dict(phase='branch', case=branch['case'], manifest_sha256=manifest_sha, branch_sha256=digest(branch)))
            calls = meter_factory()
            calls.start()
            try:
                collect(panel, base, writer, cls, branch, dest, journal, calls, row)
            finally:
                calls.stop()
            check_calls(calls.receipt(), branch['case'])
            require(len(row['records']) == 21 and row['prefix_gates'] == list(range(11)), 'complete branch')
            row['status'] = 'completed'
        except NativeDone as exc:
            row.update(status='native_early_stop', error=str(exc))
        except IntegrityFailure as exc:
            row.update(status='integrity_failure', error=str(exc))
            error = 'prefix integrity failed; later dispatch blocked'
        except Exception as exc:
            row.update(status='engineering_failure', error=type(exc).__name__+': '+str(exc))
            error = 'engineering failure; later dispatch blocked, no retry'
        finally:
            if calls is not None:
                row['meter'] = calls.receipt()
            row['operation_meter'] = None if journal is None else dict(
                durable_intents=dict(journal.attempted), durable_returns=dict(journal.returned))
            if row['close']['status'] not in ('completed','not_constructed'):
                error = 'incomplete constructor/close; later dispatch blocked'
            if row['records']:
                try:
                    last = row['records'][-1]['frame']
                    row['last_actual_frame'] = last
                    row['actual_suffix_frames'] = max(0, last-10)
                    row['last_native_metrics'] = native_metrics(read_json(dest/f'frame_{last:02d}/full_public_state.json'))
                except Exception as exc:
                    row.update(status='engineering_failure', error=str(exc))
                    error = 'endpoint receipt failure'
            try:
                verify_sources()
            except Exception as exc:
                error = 'source drift: '+str(exc)
                row.update(status='integrity_failure', error=error)
            if journal is not None:
                try:
                    journal.append(dict(phase='terminal', row_sha256=digest(row)), final=True)
                finally:
                    journal.close()
            writer.save_json(out/f"row_{branch['case']}.json", row, final=True)
    result = dict(schema=TAG+'.report', status='all12_native_replays_recorded' if all(r['status']=='completed' for r in rows) else 'native_replay_incomplete',
        manifest_sha256=manifest_sha, manifest_digest=digest(manifest), policy=policy(), rows=rows,
        global_error=error, wall_seconds=writer.clock()-writer.started, artifacts=inventory(out),
        neural_executed=False, scientific_admission=False, action_witness_established=False,
        task_success=None, blockers=policy()['blockers'])
    writer.save_json(out/'report.json', result, final=True)
    inventory(out)
    return result


def run(manifest, manifest_path, manifest_sha, out):
    require(sha(manifest_path) == manifest_sha, 'MAIN must pin actual prepare FILE SHA')
    verify(manifest)
    panel, _, base, _ = dependencies(ROOT)
    def stable():
        for p,h in manifest['bindings'].items():
            # No checkpoint deserialization. Hashing only, bounded closure, no scan.
            require(sha(p) == h, 'source/input drift: '+p)
    def load():
        runtime, parent = base.load_runtime(ROOT)
        probe = runtime.probe(ROOT, parent, Path(manifest['runtime_manifest']['path']))
        require(probe['status'] == 'native_import_ready_physics_not_executed', 'isolated runtime preflight rejected')
        native = manifest['native_source']
        module = import_bound(Path(native['path']), native['sha256'], 'native')
        cls = module.PushTEnv
        require(Path(cls.__init__.__code__.co_filename).resolve() == Path(native['path']), 'native class provenance')
        import pymunk
        return cls, lambda:base.NativeCalls(cls,pymunk), probe
    require(signal.getsignal(signal.SIGALRM) == signal.SIG_DFL
            and signal.getitimer(signal.ITIMER_REAL) == (0.0,0.0), 'existing alarm rejected')
    def timeout(signum, frame):
        raise TimeoutError('900s physical wall alarm; no retry')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(WALL)
    try:
        return execute(manifest, manifest_sha, out, panel, base, load, stable)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, signal.SIG_DFL)


def read_journal(path):
    require(Path(path).stat().st_size <= 8*MIB, 'journal cap')
    previous, rows = None, []
    with Path(path).open() as stream:
        for sequence, line in enumerate(stream):
            row = json.loads(line)
            h = row.pop('entry_sha256')
            require(row['sequence'] == sequence and row['previous_sha256'] == previous and digest(row) == h, 'journal chain')
            previous = h
            rows.append(row['event'])
    return rows


def validate_frame(path, frame, goal):
    arrays = {}
    for name, shape, dtype in (('rgb',(224,224,3),np.uint8),('state',(7,),np.float32),('proprio',(4,),np.float32)):
        p = path/(name+'.npy')
        require(p.stat().st_size < 200000, 'array byte cap')
        a = np.load(p, allow_pickle=False, max_header_size=1024)
        require(a.shape == shape and a.dtype == dtype and np.isfinite(a).all(), 'native array ABI')
        arrays[name] = a
    public = read_json(path/'full_public_state.json')
    transition = read_json(path/'transition.json')
    agent, block = public['bodies']['agent'], public['bodies']['block']
    native = np.asarray(agent['position']+block['position']+[block['angle'] % (2*np.pi)]+agent['velocity'], np.float32)
    require(np.array_equal(arrays['state'], native) and np.array_equal(arrays['proprio'], native[[0,1,5,6]]), 'public/native7/proprio identity')
    require(public['env']['goal_pose'] == goal and len(public['env']['coverage_arr']) == frame, 'unchanged native goal/coverage history')
    if frame == 0:
        require(transition == dict(kind='reset'), 'reset transition')
    else:
        info = transition['info']
        require(transition['kind'] == 'step' and transition['done'] is False,
                'bound native done=False contract (not coverage threshold)')
        for key, val in (('state', arrays['state']), ('pos_agent',np.asarray(agent['position'])),
                         ('vel_agent',np.asarray(agent['velocity'])),
                         ('block_pose',np.asarray(block['position']+[block['angle']]))):
            require(info[key]['values'] == val.tolist(), 'returned info/public state identity')
        coverage = public['env']['coverage_arr']
        require(info['goal_pose'] == goal and info['final_coverage'] == coverage[-1]
                and info['max_coverage'] == max(coverage), 'native returned coverage/goal')
        require(transition['reward'] == float(np.clip(coverage[-1]/public['env']['success_threshold'],0,1)), 'native reward, not a new success gate')
    return public, transition


def validate_saved(manifest, manifest_sha, report_path, report_sha):
    """Strict saved-only validator; authority verification supplied by consume()."""
    report_path = Path(report_path)
    r = read_json(report_path, report_sha)
    require(r['schema'] == TAG+'.report' and r['manifest_sha256'] == manifest_sha
            and r['manifest_digest'] == digest(manifest) and r['policy'] == policy(), 'report contract binding')
    require(r['neural_executed'] is False and r['scientific_admission'] is False
            and r['action_witness_established'] is False and r['task_success'] is None
            and r['blockers'] == policy()['blockers'], 'no task/scientific promotion')
    require(0 <= r['wall_seconds'] < WALL and len(r['rows']) == 12, 'wall/full denominator')
    out = report_path.parent
    require(inventory(out) == r['artifacts'], 'complete saved output inventory')
    dispatch = read_json(out/'dispatch.json')
    require(dispatch == dict(manifest_sha256=manifest_sha, branches=manifest['branches'], policy=policy()), 'predispatch full12 identity')
    verified = []
    for row, b in zip(r['rows'], manifest['branches']):
        require({k:row[k] for k in ('case','seed','word','root_sha256')} == {k:b[k] for k in ('case','seed','word','root_sha256')}, 'branch/source identity')
        require(row == read_json(out/f"row_{b['case']}.json"), 'durable row identity')
        dest = out/b['case']
        records = row['records']
        require([x['frame'] for x in records] == list(range(len(records))) and len(records) <= 21, 'actual contiguous frames only')
        for rec in records:
            frame = rec['frame']
            frame_path = dest/f'frame_{frame:02d}'
            require(set(rec['files']) == set(NAMES), 'five native files')
            for name in NAMES:
                p = frame_path/name
                require(rec['files'][name] == dict(sha256=sha(p), bytes=p.stat().st_size), 'frame asset binding')
            public, transition = validate_frame(frame_path, frame, b['native_goal'])
            require(rec['state'] == np.load(frame_path/'state.npy',allow_pickle=False).tolist()
                    and rec['proprio'] == np.load(frame_path/'proprio.npy',allow_pickle=False).tolist(), 'frame summary binding')
            if frame <= 10 and frame in row['prefix_gates']:
                prefix_gate(frame_path, b['prefix_identity'][frame])
            if frame:
                previous = read_json(dest/f'frame_{frame-1:02d}/full_public_state.json')
                raw = np.asarray((b['prefix']+b['suffix'])[frame-1],np.float64)
                expected = command_for(frame, raw, np.asarray(previous['bodies']['agent']['position'],np.float64), sha(dest/f'frame_{frame-1:02d}/full_public_state.json'))
                require(read_json(dest/f'command_{frame:02d}.json') == expected, 'exact raw command/pre-state chain')
                require(public['env']['latest_action'] == expected['absolute_target'], 'actual native target')
                if frame <= 10:
                    require(sha(dest/f'command_{frame:02d}.json') == b['prefix_commands'][frame-1]['sha256'], 'original prefix command bytes')
            if frame < len(records)-1:
                require(not transition.get('done',False), 'no step after native done')
        if records:
            last = len(records)-1
            require(row['last_actual_frame'] == last and row['actual_suffix_frames'] == max(0,last-10)
                    and row['last_native_metrics'] == native_metrics(read_json(dest/f'frame_{last:02d}/full_public_state.json')), 'actual last endpoint only')
        if dest.exists():
            events = read_journal(dest/'journal.jsonl')
            require(events[0] == dict(phase='branch',case=b['case'],manifest_sha256=manifest_sha,branch_sha256=digest(b))
                    and events[-1] == dict(phase='terminal',row_sha256=digest(row)), 'journal branch/terminal binding')
            attempts = [e for e in events if e['phase']=='intent']
            returns = [e for e in events if e['phase']=='returned']
            require(row['operation_meter'] == dict(durable_intents=dict(Counter(e['operation'] for e in attempts)),
                    durable_returns=dict(Counter(e['operation'] for e in returns))), 'actual durable intent/return meter')
            require(sum(e['operation']=='step' for e in returns) >= max(0,len(records)-1), 'return before capture required')
            if row['status'] == 'completed':
                ops = ['construct','seed','reset']+['step']*20+['close']
                require([e['operation'] for e in attempts] == [e['operation'] for e in returns] == ops, 'complete actual attempt/return census')
                require([e['frame'] for e in events if e['phase']=='prefix_byte_gate'] == list(range(11)), 'journal prefix gates')
                gate_end = max(i for i,e in enumerate(events) if e['phase']=='prefix_byte_gate')
                suffix_start = min(i for i,e in enumerate(events) if e['phase']=='intent' and e['operation']=='step' and e['details']['frame']==11)
                require(gate_end < suffix_start, 'gates precede physical suffix')
                require(returns[-1]['counters_after'] == row['meter'], 'final actual meter binding')
                # Complete branches have strict alternating intent/return pairs;
                # intervening prefix gates are not native calls.
                operations = [e for e in events if e['phase'] in ('intent','returned')]
                for pair in range(24):
                    before, after = operations[2*pair:2*pair+2]
                    require(before['phase']=='intent' and after['phase']=='returned'
                            and before['operation']==after['operation']
                            and before['role']==after['role']==b['case'], 'native operation pairing')
                    if before['operation']=='step':
                        frame = pair-2
                        require(before['details']==read_json(dest/f'command_{frame:02d}.json'), 'dispatch command binding')
                    if pair:
                        require(before['counters_before']==operations[2*pair-1]['counters_after'], 'native counter continuity')
                require(operations[0]['counters_before']==dict(python_call_entries={},space_step_events=[]), 'fresh profiler/branch')
        if row['status'] == 'completed':
            require(row['error'] is None and len(records)==21 and row['prefix_gates']==list(range(11))
                    and row['close']==dict(status='completed',attempted=True,returned=True,durable_return=True,original_error=None,error=None), 'complete cleanup/gated branch')
            require(sha(dest/'pre_reset_rng.json') == b['seeded_rng']['sha256'], 'seeded RNG replay')
            check_calls(row['meter'], b['case'])
            verified.append(b['case'])
        else:
            require(row['status'] in ('not_run','native_early_stop','engineering_failure','integrity_failure') and row['error'] is not None, 'explicit failed denominator')
    all_done = len(verified)==12
    require(r['status'] == ('all12_native_replays_recorded' if all_done else 'native_replay_incomplete'), 'status reflects full denominator')
    require(not all_done or r['global_error'] is None, 'no hidden global error')
    return dict(schema=TAG+'.consume', status='all12_saved_native_replays_verified' if all_done else 'partial_saved_native_replays_retained',
        report_sha256=report_sha, manifest_sha256=manifest_sha, denominator=12, verified_cases=verified,
        statuses={x['case']:x['status'] for x in r['rows']}, scientific_admission=False,
        action_witness_established=False, task_success=None, blockers=policy()['blockers'])


def consume(manifest_path, manifest_sha, report_path, report_sha):
    require(sha(manifest_path) == manifest_sha, 'actual MAIN prepare FILE SHA')
    manifest = read_json(manifest_path)
    verify(manifest)
    return validate_saved(manifest, manifest_sha, report_path, report_sha)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare','verify','run','consume'])
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--report-sha256')
    parser.add_argument('--confirm-physics', action='store_true')
    args = parser.parse_args(argv)
    if args.command=='prepare':
        require(not args.manifest.exists(), 'new manifest only')
        result = prepare()
        _,_,base,_ = dependencies(ROOT)
        base.save_json(args.manifest, result)
        result = dict(path=str(args.manifest), sha256=sha(args.manifest), status=result['status'])
    else:
        require(args.manifest_sha256 and sha(args.manifest)==args.manifest_sha256, 'actual manifest FILE SHA required')
        manifest = read_json(args.manifest)
        if args.command=='verify':
            verify(manifest)
            result = dict(status='saved_prepare_verified_no_physics', scientific_admission=False)
        elif args.command=='run':
            require(args.confirm_physics and args.output is not None, 'MAIN --confirm-physics and new --output required')
            result = run(manifest,args.manifest,args.manifest_sha256,args.output)
        else:
            require(args.report is not None and args.report_sha256, 'actual report path/SHA required')
            result = consume(args.manifest,args.manifest_sha256,args.report,args.report_sha256)
    print(json.dumps(result,sort_keys=True,indent=2,allow_nan=False))
    if result['status'] in ('native_replay_incomplete','partial_saved_native_replays_retained'):
        return 2
    return 0


if __name__=='__main__':
    raise SystemExit(main())
