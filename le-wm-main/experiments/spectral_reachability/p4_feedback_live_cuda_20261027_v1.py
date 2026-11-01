"""Known-root live feedback experiment. prepare/verify are saved-data only.

Only MAIN may invoke run. No admission of the old runtime or saved winners.
Every dispatch requires a new durable selection; failures are nonrefundable.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import io as byteio
import json
import math
import os
from pathlib import Path
import shutil
import signal
import time
import numpy as np
from experiments.spectral_reachability import p4_feedback_root_cuda_response_20261027_v1 as response
from experiments.spectral_reachability import p4_feedback_root_cuda_response_saved_audit_20261027_v1 as audit
from experiments.spectral_reachability.pusht_visible_command_executor import execute

NAME = 'p4_feedback_live_cuda_20261027_v1'
REPO = response.REPO
io = response.io
require = io.require
ROOT = '9173218f71898716eac8fe0f3946b8e2cf20527ac71d84555c20c4e1b7fbf333'
ARMS = ('imagined', 'actual')
BOUNDARIES = (5, 10, 15)
RESERVE = 200 * 1024**2
OUTPUT_CAP = 16 * 1024**2
PREPARE_CAP = 8 * 1024**2
PHYSICAL_PREPARE_SHA = '1cfc5eb2ddcfc563176a3cce51a0ef90b88e92d298b3cf35d415c2e420317daf'
PHYSICAL_REPORT_SHA = '6c3659e199a8151fee2fa2c020a0cfe63db00b691b947cdbeb762c3c18d6004a'
PHYSICAL_DIR = REPO / 'results/p3_joint_horizon_goal_metric_physical_20261027_v1_main01'
KINDS = dict(predict=80, action_encoder=80, encoder=8, projector=8,
             jacobian_api=6, grad_api=1152, engine_api=1152, backward_api=0)


def caps(wall_seconds, allocated_bytes, reserved_bytes):
    require(type(wall_seconds) in (int, float) and math.isfinite(wall_seconds) and wall_seconds > 0, 'explicit wall cap')
    require(type(allocated_bytes) is type(reserved_bytes) is int and 0 < allocated_bytes <= reserved_bytes, 'explicit VRAM caps')
    return dict(wall_seconds=float(wall_seconds), allocated_bytes=allocated_bytes, reserved_bytes=reserved_bytes,
                neural=KINDS, resets=34, root_collection_resets=32, branch_resets=2,
                warmup_steps=30, selected_steps=40, public_steps=70,
                per_round=dict(predict=13, action_encoder=13, jacobian_api=1, grad_api=192, engine_api=192),
                mathematical_VJP_directions=1152, processed_context_tokens=240, encoded_frames=20,
                output_bytes=OUTPUT_CAP, disk_floor=RESERVE, retries=False, FD=0, training=False,
                deadline='POSIX alarm and synchronized checks; MAIN external watchdog still required')


def material(limits):
    require(limits == caps(limits['wall_seconds'], limits['allocated_bytes'], limits['reserved_bytes']), 'exact live budgets')
    pack = response.validate_pack(response.source.verify(response.SOURCEPACK, response.SOURCEPACK_SHA))
    verified = audit.audit(audit.DEFAULT, audit.RECEIPT_SHA)
    require(verified['provenance_verified'] and pack['root_sha256'] == ROOT, 'actual known root0 sources required')
    binding = io.read(PHYSICAL_DIR / 'execution_protocol.json')
    require(binding['prepare_sha256'] == PHYSICAL_PREPARE_SHA, 'original physical prepare identity')
    folder = Path(binding['prepared']); pr = io.read(folder / 'prepare.json')
    require(io.sha(folder / 'prepare.json') == PHYSICAL_PREPARE_SHA and
            io.sha(folder / 'protocol.json') == pr['protocol_sha256'], 'original physical protocol SHA')
    for seal in pr['source_seals']:
        io.verify_seal(output_dir=seal['output_dir'], expected_manifest_sha256=seal['manifest_sha256'])
    physical = io.read(folder / 'protocol.json')
    require(physical['root_seed'] == 202753010 and physical['root_oversample_factor'] == 4 and
            len(physical['references']) == 4 and physical['model_bindings'] == pack['model_bindings'], 'original physical root/model setup')
    report_path = PHYSICAL_DIR / 'report.json'
    require(io.sha(report_path) == PHYSICAL_REPORT_SHA, 'original physical report SHA')
    report = io.read(report_path)
    require(report['physical_call_receipt']['resets']['root_collection']['attempts'] == 32, 'actual collector reset budget')
    bindings = {str(folder / 'prepare.json'): PHYSICAL_PREPARE_SHA,
                str(folder / 'protocol.json'): pr['protocol_sha256'],
                str(PHYSICAL_DIR / 'execution_protocol.json'): io.sha(PHYSICAL_DIR / 'execution_protocol.json'),
                str(report_path): PHYSICAL_REPORT_SHA, str(audit.DEFAULT / 'receipt.json'): audit.RECEIPT_SHA}
    for path, digest in {**physical['sources_sha256'], **pack['model_bindings']}.items():
        require(io.sha(path) == digest, 'physical source/model drift: ' + path)
    config = io.read(next(x for x in pack['model_bindings'] if Path(x).name == 'config.json'))
    require(config['encoder'].get('pretrained') is False, 'no model downloads')
    return dict(schema=NAME + '.protocol', root_sha256=ROOT, sourcepack=str(response.SOURCEPACK),
                sourcepack_prepare_sha256=response.SOURCEPACK_SHA, response_receipt_sha256=audit.RECEIPT_SHA,
                expected_model_hash=io.read(audit.DEFAULT / 'receipt.json')['model_hash_before'],
                initial_context=pack['initial_context'], prefix_commands=pack['prefix_commands'],
                goal=pack['goal'], q=pack['q'], normalization=pack['normalization'],
                model_bindings=pack['model_bindings'], active_metric_id=pack['active_metric_id'],
                references=physical['references'], root_seed=physical['root_seed'], bindings=bindings,
                physical_sources=physical['sources_sha256'], limits=limits,
                arm_order=list(ARMS), boundaries=list(BOUNDARIES),
                selection='fresh nominal/J/ten nonlinear candidates/exact replay; durable seal before dispatch',
                feedback='actual: own RGB3; imagined: persistent predicted context; RGB guard in BOTH',
                terminal='own RGB1 at20 is evaluation only',
                known_roots_only=True, independent_confirmation=False, P1_admission=False, P4_admission=False)


class Store:
    def __init__(self, path, cap=OUTPUT_CAP):
        self.path = Path(path).absolute(); self.path.mkdir(exist_ok=False); self.cap = cap

    def check(self, n=0, final=False):
        require(io.disk_bytes(self.path) + n <= self.cap - (0 if final else 1024**2), 'small output cap')
        require(shutil.disk_usage(self.path).free >= RESERVE + n, '200MiB disk floor')

    def blob(self, name, raw, final=False):
        require(Path(name).name == name, 'flat exclusive output filename')
        self.check(len(raw), final)
        with (self.path / name).open('xb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        io.fsync_directory(self.path)
        return io.sha(self.path / name)

    def write(self, name, value, final=False):
        return self.blob(name, (io.io.canonical(value) + '\n').encode(), final)

    def array(self, name, value):
        buf = byteio.BytesIO(); np.save(buf, np.asarray(value, dtype=np.float32), allow_pickle=False)
        return self.blob(name, buf.getvalue())

    def png(self, name, rgb):
        from PIL import Image
        rgb = valid_rgb(rgb); buf = byteio.BytesIO(); Image.fromarray(rgb).save(buf, format='PNG')
        return dict(file=name, file_sha256=self.blob(name, buf.getvalue()), pixel_sha256=pixel_sha(rgb))


def sources(p):
    extra = [Path(__file__), REPO / 'tests' / ('test_' + NAME + '.py'), REPO / 'innerdocs' / (NAME + '.md'), Path(audit.__file__)]
    return sorted(set(response.sources() + extra + [Path(x) for x in p['physical_sources']]))


def prepare(output, limits):
    output = Path(output).absolute()
    require(not output.exists() and shutil.disk_usage(output.parent).free >= RESERVE + PREPARE_CAP + OUTPUT_CAP, 'NEW prepare headroom')
    p = material(limits); files = sources(p)
    require(sum(f.stat().st_size for f in files) < 6 * 1024**2, 'bounded source closure')
    p['sources_sha256'] = {str(f): io.sha(f) for f in files}
    store = Store(output, PREPARE_CAP); store.write('protocol.json', p); seals = []
    for offset in range(0, len(files), 40):
        store.check(sum(f.stat().st_size for f in files[offset:offset+40]) + 65536)
        seals.append(io.create_seal(files=files[offset:offset+40], args_json=io.io.canonical(dict(protocol_sha256=io.sha(output / 'protocol.json'))),
                                    output_dir=output / ('source_seal_' + str(offset // 40))))
    store.write('prepare.json', dict(schema=NAME + '.prepare', status='prepared_not_run',
                protocol_sha256=io.sha(output / 'protocol.json'), source_seals=seals))
    return io.sha(output / 'prepare.json')


def verify(folder, expected):
    folder = Path(folder).resolve(); require(io.sha(folder / 'prepare.json') == expected, 'external prepare SHA')
    receipt = io.read(folder / 'prepare.json'); p = io.read(folder / 'protocol.json')
    require(receipt['schema'] == NAME + '.prepare' and receipt['status'] == 'prepared_not_run' and
            io.sha(folder / 'protocol.json') == receipt['protocol_sha256'], 'live prepare identity')
    for seal in receipt['source_seals']:
        io.verify_seal(output_dir=seal['output_dir'], expected_manifest_sha256=seal['manifest_sha256'])
        require(io.read(Path(seal['output_dir']) / 'args.json') == dict(protocol_sha256=receipt['protocol_sha256']), 'seal arguments')
    for path, digest in p['sources_sha256'].items(): require(io.sha(path) == digest, 'source drift: ' + path)
    rebuilt = material(p['limits']); rebuilt['sources_sha256'] = {str(f): io.sha(f) for f in sources(rebuilt)}
    require(io.same(rebuilt, p), 'saved live protocol exact reconstruction')
    return p


def valid_rgb(rgb):
    a = np.asarray(rgb)
    require(a.shape == (224, 224, 3) and a.dtype == np.uint8, 'fresh native uint8 RGB224')
    return a.copy()


def pixel_sha(rgb): return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()


class Ledger:
    """Global, per-arm, per-boundary, per-phase attempted/returned API ledger."""
    def __init__(self, limits, emit, clock=time.monotonic):
        self.limits, self.emit, self.clock = limits, emit, clock; self.started = clock()
        self.sample = lambda: (0, 0); self.sync = lambda: None
        self.arm = None; self.boundary = None; self.phase = None; self.frames = 0
        self.entries = []; self.active = []; self.failed = False; self.peaks = [0, 0]
        self.physics = dict(root_resets=0, branch_resets=0, warmup_steps=0, selected_steps=0)
        self.physics_returns = dict(self.physics); self.rounds = []

    def check(self):
        require(not self.failed, 'fail-sticky ledger')
        require(self.clock() - self.started <= self.limits['wall_seconds'], 'wall cap')
        values = self.sample(); self.peaks = [max(a, b) for a, b in zip(self.peaks, values)]
        require(values[0] <= self.limits['allocated_bytes'] and values[1] <= self.limits['reserved_bytes'], 'VRAM cap')

    def slot(self, arm, boundary, phase, frames=0):
        self.check(); require(not self.active and arm in ARMS, 'nonreentrant named work slot')
        self.arm, self.boundary, self.phase, self.frames = arm, boundary, phase, frames
        self.emit(dict(event='slot', arm=arm, boundary=boundary, phase=phase, frames=frames))

    def reject(self, kind, message):
        self.failed = True; self.emit(dict(event='blocked', kind=kind, error=message)); raise ValueError(message)

    def call(self, kind, callback, device='cuda', shape=None):
        try:
            self.check(); require(device == 'cuda', 'no CPU neural fallback')
            require(kind in KINDS and self.arm in ARMS, 'metered native work only')
            entries = [e for e in self.entries if e['kind'] == kind]
            require(len(entries) < KINDS[kind], 'global API cap: ' + kind)
            own = [e for e in entries if e['arm'] == self.arm and e['boundary'] == self.boundary and e['phase'] == self.phase]
            if kind in ('encoder', 'projector'):
                require(self.phase in ('observe', 'terminal') and self.frames == (1 if self.phase == 'terminal' else 3), 'encode slot')
                require((self.phase == 'terminal' and self.boundary == 20) or
                        (self.phase == 'observe' and self.boundary in BOUNDARIES), 'encode boundary')
                require(not own and not self.active and shape == ((self.frames, 3, 224, 224) if kind == 'encoder' else (self.frames, 192)), 'one visual module/native shape')
                if kind == 'projector': require(any(e['completed'] for e in self.entries if e['arm'] == self.arm and e['boundary'] == self.boundary and e['kind'] == 'encoder'), 'encoder before projector')
            elif kind in ('predict', 'action_encoder'):
                require((self.phase == 'prefix' and self.boundary == 0) or
                        (self.phase in response.PHASES and self.boundary in BOUNDARIES), 'planning slot')
                require(self.active == (['jacobian_api'] if self.phase == 'jacobian' else []) and not own, 'one forward per slot')
                require(shape == ((1, 3, 192) if kind == 'predict' else (1, 3, 10)), 'native full3 shape')
                if kind == 'predict': require(any(e['kind'] == 'action_encoder' and e['completed'] and e['arm'] == self.arm and e['boundary'] == self.boundary and e['phase'] == self.phase for e in self.entries), 'action encoder before predict')
            else:
                require(self.phase == 'jacobian' and self.boundary in BOUNDARIES, 'AD only fresh round')
                require(self.active == {'jacobian_api': [], 'grad_api': ['jacobian_api'], 'engine_api': ['jacobian_api', 'grad_api']}[kind], 'serial AD nesting')
                require(len(own) < (1 if kind == 'jacobian_api' else 192), 'per-round AD cap')
            entry = dict(index=len(self.entries), arm=self.arm, boundary=self.boundary, phase=self.phase, kind=kind, completed=False)
            self.entries.append(entry); self.emit(dict(event='attempt', **entry)); self.active.append(kind)
            try: result = callback(); self.sync(); self.check()
            finally: self.active.pop()
            entry['completed'] = True; self.emit(dict(event='return', **entry)); return result
        except BaseException:
            self.failed = True; raise

    def physical(self, key, callback):
        try:
            self.check(); cap = dict(root_resets=32, branch_resets=2, warmup_steps=30, selected_steps=40)[key]
            require(self.physics[key] < cap and not self.active, 'physics budget / no physics in neural call')
            self.physics[key] += 1
            event = dict(arm=self.arm, boundary=self.boundary, kind=key, attempt=self.physics[key])
            self.emit(dict(event='physical_attempt', **event)); result = callback()
            self.physics_returns[key] += 1; self.emit(dict(event='physical_return', **event)); self.check(); return result
        except BaseException:
            self.failed = True; raise

    def counts(self):
        return {k: dict(attempted=sum(e['kind'] == k for e in self.entries), completed=sum(e['kind'] == k and e['completed'] for e in self.entries)) for k in KINDS}

    def round_complete(self):
        counts = {k: sum(e['kind'] == k and e['completed'] and e['arm'] == self.arm and e['boundary'] == self.boundary for e in self.entries)
                  for k in self.limits['per_round']}
        require(counts == self.limits['per_round'], 'exact useful work per arm/round')
        row = dict(arm=self.arm, boundary=self.boundary, counts=counts)
        require(not any(r['arm'] == self.arm and r['boundary'] == self.boundary for r in self.rounds), 'round once only')
        self.rounds.append(row); self.emit(dict(event='round_complete', **row))

    def complete(self):
        self.check(); require(self.counts() == {k: dict(attempted=v, completed=v) for k, v in KINDS.items()}, 'exact total neural work')
        require(self.physics == self.physics_returns == dict(root_resets=32, branch_resets=2, warmup_steps=30, selected_steps=40), 'exact70 public steps /34 resets')
        require([(r['arm'], r['boundary']) for r in self.rounds] == [(a, b) for a in ARMS for b in BOUNDARIES], 'six causal rounds')


def compile_round(p, context, backend, ledger, store, arm, boundary):
    prefix = f'{arm}.step{boundary}'
    io.context(context); backend.bind(dict(context=deepcopy(context)))
    request = dict(root_sha256=p['root_sha256'], arm=arm, boundary=boundary,
                   context=context, context_sha256=io.digest(context), goal=p['goal'], q_sha256=io.digest(p['q']))
    store.write(prefix + '.request.json', request)
    ledger.slot(arm, boundary, 'nominal'); nominal = backend.nominal(); io.context(nominal)
    io.history(context, nominal, np.zeros((5, 2)), p['normalization'])
    ledger.slot(arm, boundary, 'jacobian'); j, forward = backend.jacobian(); j = io.array(j, (192, 10))
    require(io.same(forward, nominal) and np.linalg.norm(j) > 0, 'fresh AD/nominal consistency and nonzero derivative')
    bank = response.compiler.spectral_bank(j, np.asarray(nominal['latent_history'])[-1], np.asarray(p['goal']), np.asarray(p['q']))
    require(len(bank['params']) == len(bank['kinds']) == 10, 'ten candidates')
    store.array(prefix + '.J.npy', j); candidates = []
    for index, (logits, kind) in enumerate(zip(bank['params'], bank['kinds'], strict=True)):
        logits = io.array(logits, (10,)); require(np.linalg.norm(logits) <= response.compiler.CAP + 1e-10, 'shared candidate radius')
        commands = np.tanh(logits.astype(np.float32)).reshape(5, 2).tolist()
        ledger.slot(arm, boundary, 'candidate_' + str(index)); successor = backend.candidate(commands)
        io.history(context, successor, commands, p['normalization'])
        cost = float(np.mean((np.asarray(p['q']) @ (np.asarray(successor['latent_history'])[-1] - p['goal']))**2))
        require(math.isfinite(cost), 'finite candidate cost')
        candidates.append(dict(index=index, kind=kind, logits=logits.tolist(), commands=commands, context=successor, cost=cost))
    winner = min(candidates, key=lambda row: (row['cost'], row['index']))
    ledger.slot(arm, boundary, 'replay'); replay = backend.replay(winner['commands'])
    require(io.same(replay, winner['context']), 'independent full-context replay exact')
    ledger.round_complete()
    selection = dict(request=request, nominal=nominal, jacobian_forward=forward, bank=bank,
                     jacobian_file=prefix + '.J.npy', jacobian_sha256=io.sha(store.path / (prefix + '.J.npy')),
                     candidates=candidates, winner=winner, replay=replay, fresh_context_specific_AD=True)
    name = prefix + '.selection.json'; digest = store.write(name, selection)
    # Readback binds ALL scored candidates and winner before physical authorization.
    require(io.sha(store.path / name) == digest and io.same(io.read(store.path / name), selection), 'durable selection readback')
    ledger.emit(dict(event='selection_sealed', arm=arm, boundary=boundary, file=name, sha256=digest))
    return winner, dict(file=name, sha256=digest)


def workflow(p, backend, physical, ledger, store, rows):
    """No torch or simulator import: mockable orchestration with real source shape."""
    for arm in ARMS:
        row = dict(arm=arm, status='started', rounds=[]); rows.append(row)
        ledger.slot(arm, 0, 'reset'); frames = physical.start(arm)
        frames = [valid_rgb(f) for f in frames]; require(len(frames) == 3, 'three real warmup frames')
        imagined = deepcopy(p['initial_context']); observed_previous = np.asarray(imagined['latent_history'])
        prefix_name = arm + '.prefix.selection.json'
        prefix_sha = store.write(prefix_name, dict(root=p['root_sha256'], commands=p['prefix_commands'], initial_context=imagined))
        frames = frames[1:] + [physical.packet(arm, 0, p['prefix_commands'], dict(file=prefix_name, sha256=prefix_sha))]
        ledger.slot(arm, 0, 'prefix'); backend.bind(dict(context=imagined)); imagined = backend.replay(p['prefix_commands'])
        io.history(p['initial_context'], imagined, p['prefix_commands'], p['normalization'])
        store.write(arm + '.prefix.prediction.json', imagined)
        for boundary in BOUNDARIES:
            ledger.slot(arm, boundary, 'observe', frames=3)
            observed = np.asarray(backend.encode(frames), dtype=np.float32)
            require(observed.shape == (3, 192) and np.isfinite(observed).all(), 'native observed RGB3 latent')
            require(np.allclose(observed[:2], observed_previous[1:], rtol=0, atol=2e-5), 'overlapping own real frame latents')
            observed_previous = observed.copy()
            latent_name = f'{arm}.step{boundary}.observed.npy'; latent_sha = store.array(latent_name, observed)
            context = deepcopy(imagined)
            if arm == 'actual': context['latent_history'] = observed.tolist()
            evidence = dict(arm=arm, boundary=boundary, pixel_sha256=[pixel_sha(f) for f in frames],
                            latent_file=latent_name, latent_sha256=latent_sha, planning_context=context,
                            actual_observation_consumed=arm == 'actual', action_history=context['action_history'])
            store.write(f'{arm}.step{boundary}.observed.json', evidence)
            winner, sealed = compile_round(p, context, backend, ledger, store, arm, boundary)
            row['rounds'].append(dict(boundary=boundary, selection=sealed, winner_index=winner['index'], model_cost=winner['cost']))
            frames = frames[1:] + [physical.packet(arm, boundary, winner['commands'], sealed)]
            imagined = deepcopy(winner['context'])
        ledger.slot(arm, 20, 'terminal', frames=1)
        terminal = np.asarray(backend.encode([frames[-1]]), dtype=np.float32)
        require(terminal.shape == (1, 192) and np.isfinite(terminal).all(), 'native terminal192')
        latent_name = arm + '.step20.native192.npy'; latent_sha = store.array(latent_name, terminal[0])
        delta = terminal[0].astype(float) - np.asarray(imagined['latent_history'])[-1]
        evaluation = dict(latent_file=latent_name, latent_sha256=latent_sha, pixel_sha256=pixel_sha(frames[-1]),
                          predicted_context=imagined, latent_MSE=float(np.mean(delta**2)),
                          Q_prediction_MSE=float(np.mean((np.asarray(p['q']) @ delta)**2)),
                          Q_goal_MSE=float(np.mean((np.asarray(p['q']) @ (terminal[0] - np.asarray(p['goal'])))**2)),
                          physical=physical.evaluate(), consumed_for_planning=False)
        row.update(status='completed20', evaluation=evaluation)
        store.write(arm + '.result.json', row)
    ledger.complete()


class Meter:
    def __init__(self, env, ledger): self.env, self.ledger, self.phase = env, ledger, 'root_resets'
    def __getattr__(self, key): return getattr(self.env, key)
    def reset(self, *args, **kwargs):
        require(self.phase in ('root_resets', 'branch_resets'), 'reset phase')
        return self.ledger.physical(self.phase, lambda: self.env.reset(*args, **kwargs))
    def step(self, action):
        require(self.phase in ('warmup_steps', 'selected_steps'), 'no collector step / no hidden warmup')
        return self.ledger.physical(self.phase, lambda: self.env.step(action))


class Physical:
    def __init__(self, p, env, ledger, store, base):
        from experiments.spectral_reachability.p3_horizon_residual_physical_20261027_v4 import validate_roots
        self.p, self.ledger, self.store, self.base = p, ledger, store, base
        self.env = Meter(env, ledger); self.used = set(); self.arms_started = []; self.next_boundary = 0
        roots = base.collect_nonterminal_boundary_contact_roots(self.env, count=4, seed=p['root_seed'], oversample_factor=4)
        self.root = validate_roots(roots, p['references'], base.root_fingerprint)[p['root_sha256']]
        require(ledger.physics['root_resets'] == ledger.physics_returns['root_resets'] == 32, 'exact actual collector resets')

    def guarded(self, commands, name):
        def observation(step, command, transition, rgb):
            self.store.write(name + f'.step{step}.json', dict(command=command.tolist(), terminated=bool(transition[2]),
                truncated=bool(transition[3]), rgb=self.store.png(name + f'.step{step}.png', rgb)))
        receipt = execute(self.env, commands, root_id=self.p['root_sha256'], agent_rgb=(65, 105, 225), on_observation=observation)
        rgb = valid_rgb(receipt.pop('terminal_rgb')); receipt['terminal_pixel_sha256'] = pixel_sha(rgb)
        self.store.write(name + '.physics.json', receipt)
        require(receipt['status'] == 'completed' and receipt['attempted_commands'] == receipt['completed_commands'] == 5, 'guarded five steps incomplete; stop without padding')
        return rgb

    def start(self, arm):
        from experiments.spectral_reachability.run_pusht_sampling_comparison_20261027_v2 import verify_rgb_context
        require(arm == ARMS[len(self.arms_started)], 'fresh ordered branch only')
        self.arms_started.append(arm); self.arm = arm; self.next_boundary = 0
        self.env.phase = 'branch_resets'
        _, info = self.env.reset(seed=0, options=dict(state=self.root['state'].copy(), goal_state=self.root['goal'].copy()))
        goal_rgb = valid_rgb(info['goal']); self.env.phase = 'warmup_steps'
        frames = [self.guarded(np.zeros((5, 2)), arm + f'.warmup{i}') for i in range(3)]
        identity = verify_rgb_context(frames, goal_rgb, self.p['references'][self.p['root_sha256']])
        mean, std = (np.asarray(self.p['normalization'][k], dtype=np.float32) for k in ('mean', 'std'))
        zero = ((np.zeros((5, 2), np.float32) - mean) / std).reshape(10)
        require(np.array_equal(np.stack([zero, zero]), self.p['initial_context']['action_history']), 'exact original warmup history')
        self.store.write(arm + '.reset_identity.json', dict(identity=identity, root=self.p['root_sha256'],
            goal_rgb=self.store.png(arm + '.goal.png', goal_rgb), actual_warmup_steps=15))
        return frames

    def packet(self, arm, boundary, commands, sealed):
        require(arm == self.arm and boundary == self.next_boundary and boundary in (0, 5, 10, 15), 'own causal dispatch order')
        path = self.store.path / sealed['file']; require(Path(sealed['file']).name == sealed['file'], 'local selection')
        key = (arm, boundary); require(key not in self.used and io.sha(path) == sealed['sha256'], 'once-only durable authorization')
        selection = io.read(path)
        if boundary == 0:
            require(selection['root'] == self.p['root_sha256'] and io.same(commands, self.p['prefix_commands']) and io.same(commands, selection['commands']), 'original prefix dispatch')
        else:
            r = selection['request']
            require(r['root_sha256'] == self.p['root_sha256'] and r['arm'] == arm and r['boundary'] == boundary and
                    r['q_sha256'] == io.digest(self.p['q']) and r['goal'] == self.p['goal'] and
                    io.same(commands, selection['winner']['commands']), 'sealed fresh own selection')
        require(np.asarray(commands).shape == (5, 2), 'five actions exactly')
        self.used.add(key); self.next_boundary += 5
        self.ledger.emit(dict(event='dispatch_authorized', arm=arm, boundary=boundary, selection=sealed))
        self.env.phase = 'selected_steps'; return self.guarded(commands, arm + f'.from{boundary}')

    def evaluate(self):
        from experiments.spectral_reachability.run_pusht_sampling_comparison_20261027_v2 import official_checked
        require(self.next_boundary == 20, 'terminal evaluation only after20')
        state = self.base.p0.packets.state_of(self.env)
        return dict(state=state.tolist(), goal=np.asarray(self.root['goal']).tolist(),
                    official=official_checked(self.env, state, self.root['goal']),
                    pose=self.base.physical._pose_label(state, self.root['goal']))


def install_hooks(t, model, ledger, hooks):
    modules = [(model, 'predict', 'predict'), (model.action_encoder, 'forward', 'action_encoder'),
               (model.encoder, 'forward', 'encoder'), (model.projector, 'forward', 'projector'),
               (t.autograd.functional, 'jacobian', 'jacobian_api'), (t.autograd, 'grad', 'grad_api'),
               (t.autograd, '_engine_run_backward', 'engine_api'), (t.autograd.graph, '_engine_run_backward', 'engine_api')]
    for obj, key, kind in modules:
        original = getattr(obj, key)
        def wrapped(*args, _original=original, _kind=kind, **kwargs):
            tensor = args[0] if _kind in ('predict', 'action_encoder', 'encoder', 'projector') else None
            return ledger.call(_kind, lambda: _original(*args, **kwargs),
                device='cuda' if tensor is None else getattr(getattr(tensor, 'device', None), 'type', None),
                shape=None if tensor is None else tuple(tensor.shape))
        hooks.install(obj, key, wrapped)
    def denied(*args, **kwargs): return ledger.reject('backward_api', 'public backward forbidden')
    hooks.install(t.autograd, 'backward', denied)
    original_encode = model.encode
    def guarded_encode(*args, **kwargs):
        require(ledger.phase in ('observe', 'terminal') and not ledger.active, 'encode only explicit live slots')
        return original_encode(*args, **kwargs)
    hooks.install(model, 'encode', guarded_encode)


class Native(response.Native):
    def encode(self, frames):
        from experiments.spectral_reachability.preflight_pusht_d2p_p0 import _frames_tensor
        from experiments.run_pusht_counterfactual import image_transform
        with self.t.no_grad():
            values = self.model.encode(dict(pixels=_frames_tensor(frames, transform=image_transform(), device=self.device)))['emb'][0]
        require(values.dtype == self.t.float32 and values.device == self.device, 'native CUDA float32 encoding')
        return values.detach().cpu().numpy()


def run(folder, expected, output):
    p = verify(folder, expected); output = Path(output).absolute()
    require(not output.exists() and shutil.disk_usage(output.parent).free >= RESERVE + OUTPUT_CAP, 'NEW run /200MiB floor')
    store = Store(output); store.write('execution_binding.json', dict(prepared=str(Path(folder).resolve()), prepare_sha256=expected))
    journal = response.Journal(store); ledger = Ledger(p['limits'], journal); rows = []; errors = []
    hooks = io.Hooks(); model = env = None; before = after = host = None; old_alarm = None; armed = False; source_ok = False
    try:
        require(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), 'MAIN dedicated process; no existing alarm')
        def deadline(signum, frame): raise TimeoutError('live wall cap')
        old_alarm = signal.signal(signal.SIGALRM, deadline); signal.setitimer(signal.ITIMER_REAL, p['limits']['wall_seconds']); armed = True
        os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
        host = require_cuda_allocation('cuda:0'); device = torch.device('cuda:0'); torch.cuda.reset_peak_memory_stats(device)
        ledger.sync = lambda: torch.cuda.synchronize(device)
        ledger.sample = lambda: (torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device))
        total = torch.cuda.get_device_properties(device).total_memory
        require(p['limits']['reserved_bytes'] <= total, 'reserved cap exceeds device')
        torch.cuda.set_per_process_memory_fraction(p['limits']['reserved_bytes'] / total, device)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        policy = Path(next(x for x in p['model_bindings'] if Path(x).name == 'weights.pt')).parent
        model = load_compatible_pretrained(str(policy), device=device).eval().requires_grad_(False)
        io.frozen(model); require(all(b.device == device for b in model.buffers()), 'CUDA model buffers')
        before = io.tensor_hash(model); require(before == p['expected_model_hash'], 'same actual frozen response model state')
        install_hooks(torch, model, ledger, hooks)
        env = base.gym.make('swm/PushT-v1', render_mode='rgb_array')
        require(env.unwrapped.variation_space['agent']['color'].value.tolist() == [65, 105, 225], 'exact original visible agent color')
        physical = Physical(p, env, ledger, store, base)
        with sdpa_kernel(SDPBackend.MATH): workflow(p, Native(torch, model, device, p), physical, ledger, store, rows)
    except BaseException as exc: errors.append(io.error('live_execution', exc))
    finally:
        elapsed = ledger.clock() - ledger.started
        if armed: signal.setitimer(signal.ITIMER_REAL, 0)
        if old_alarm is not None: signal.signal(signal.SIGALRM, old_alarm)
        hooks.restore(errors)
        if env is not None:
            try: env.close()
            except BaseException as exc: errors.append(io.error('env_close', exc))
        if model is not None:
            try: io.frozen(model); after = io.tensor_hash(model); ledger.check()
            except BaseException as exc: errors.append(io.error('final_model_resources', exc))
        try: verify(folder, expected); source_ok = True
        except BaseException as exc: errors.append(io.error('final_source', exc))
        complete = (not errors and not ledger.failed and source_ok and before is not None and before == after and
                    len(rows) == 2 and all(r['status'] == 'completed20' for r in rows) and elapsed <= p['limits']['wall_seconds'])
        try: journal(dict(event='run_finished', complete=complete, errors=errors))
        except BaseException as exc: errors.append(io.error('final_journal', exc)); complete = False
        journal.close()
        receipt = dict(schema=NAME + '.receipt', status='two_live_20step_3round_arms_completed' if complete else 'partial_or_failed',
            root_sha256=p['root_sha256'], prepare_sha256=expected, response_receipt_sha256=p['response_receipt_sha256'],
            model_bindings=p['model_bindings'], model_hash_before=before, model_hash_after=after, source_verified=source_ok,
            rows=rows, errors=errors, counts=ledger.counts(), entries=ledger.entries, round_ledgers=ledger.rounds,
            physics_attempts=ledger.physics, physics_returns=ledger.physics_returns, limits=p['limits'],
            peaks=ledger.peaks, wall_seconds=elapsed, host=host, journal_failed=journal.failed,
            journal_sha256=io.sha(store.path / 'events.jsonl'), training=False, known_roots_only=True,
            independent_confirmation=False, P1_admission=False, P4_admission=False,
            scope='one known-root pair; engineering execution, not independent scientific confirmation')
        receipt['output_files_sha256'] = {f.name: io.sha(f) for f in store.path.iterdir() if f.is_file()}
        store.write('receipt.json', receipt, final=True)
    require(complete, 'incomplete live run; preserved partial receipt/journal; no retry'); return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'verify', 'run')); parser.add_argument('--output', type=Path)
    parser.add_argument('--prepared', type=Path); parser.add_argument('--prepare-sha')
    parser.add_argument('--wall-seconds', type=float); parser.add_argument('--allocated-bytes', type=int); parser.add_argument('--reserved-bytes', type=int)
    a = parser.parse_args()
    if a.command == 'prepare': print(prepare(a.output, caps(a.wall_seconds, a.allocated_bytes, a.reserved_bytes)))
    elif a.command == 'verify': print(json.dumps(verify(a.prepared, a.prepare_sha)['limits'], sort_keys=True))
    else: print(run(a.prepared, a.prepare_sha, a.output)['status'])


if __name__ == '__main__': main()
