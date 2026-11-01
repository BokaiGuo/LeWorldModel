"""Twelve continuous moving O/I/R episodes; MAIN owns actual CUDA/physics.

Physical state machine, own image encoding and narrow suffix handshake live
here. Suffix mathematics remains in a separately sealed delegated core.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import importlib
import io
import os
import json
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.spectral_reachability import p1_moving_baseline_physical_20261027_v1 as prior

NAME = 'p1_moving_live_feedback_physical_20261027_v1'
BACKEND = 'p1_moving_live_suffix_cuda_20261027_v1'
BACKEND_SHA = 'b421af2a74bf8af5639f043ae3a06790895f24157880e67b556695b82bb348f6'
ROOT, kernel = prior.ROOT, prior.kernel
require, sha, read, raw, source = prior.require, prior.sha, prior.read, prior.raw, prior.source
MODES = ('O', 'I', 'R')
ARMS = ('line_search_GN', 'full10_conditional_D2P')
LAYERS = tuple((s, a) for s in (0, 1) for a in ARMS)
CAP, FLOOR, FINAL, PREPARE_CAP = 64*1024**2, 200*1024**2, 256*1024, 256*1024
AUTH = 'MAIN_MOVING_LIVE_FEEDBACK_CUDA_AND_PHYSICS_ONLY'
SOURCE_SHA = prior.SOURCE_SHA
PLANNER_SHA = 'cd8432c06288577d25d9d9b92c7eb3673de592499ee64cae93238156f3f19aa0'
ENCODER_SHA = 'e7fb554327b3bb5a96190f6291f8450b2b0723cd981817715cfb7e50aa266bf6'
PHYSICAL_SHA = 'a5322dc4f483e82b5ed9d7b3996cf8a63845261d65774f90a4ae341773016aaf'


def array_sha(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def packet_digest(value):
    return hashlib.sha256((json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()).hexdigest()


def raw5(value):
    a = np.asarray(value)
    require(a.shape == (5, 2) and a.dtype.kind in 'fi' and np.isfinite(a).all() and
            np.array_equal(a, a.astype(np.float32)) and (abs(a) <= 1).all(), 'literal raw5 float32; never clip')
    return a.astype(np.float32)


def native(value):
    a = np.asarray(value)
    require(a.shape == (192,) and a.dtype == np.float32 and np.isfinite(a).all(), 'native192 float32 latent')
    return a.copy()


def contract():
    return dict(layers=[list(x) for x in LAYERS], modes=list(MODES), episodes=12, warmup=10, horizon=10,
                feedback_step=5, with_target=False, wall_seconds=600,
                physical_caps=dict(constructor=12, reset=36, set_state=72, goal_render=36,
                                   goal_setter=48, goal_bind=12, public_step=240, physics_step=2688,
                                   explicit_render=288),
                neural_caps=dict(encoder=24, projector=24, predict=40, predictor=40, action_encoder=40,
                                 pred_proj=40, ad_forward=8, jacobian=8, svd=8, jacobian_api=8,
                                 grad_api=8, vjp_directions=1536),
                output_cap_bytes=CAP, prepare_cap_bytes=PREPARE_CAP, disk_floor_bytes=FLOOR,
                final_reserve_bytes=FINAL, cuda_allocated_cap=3*1024**3, cuda_reserved_cap=4*1024**3,
                P1_admission=False, P4_admission=False, independent_confirmation=False,
                dispatch='O original suffix; I/R fixed fresh ridge, never cost fallback')


class Store(prior.Store):
    def __init__(self, path):
        self.path = Path(path).absolute()
        require(not self.path.exists() and shutil.disk_usage(self.path.parent).free >= FLOOR+CAP,
                'NEW64MiB +200MiB reserve')
        self.path.mkdir(); self.previous = None; self.index = 0

    def check(self, size=4096):
        used = sum(f.stat().st_size for f in self.path.rglob('*') if f.is_file())
        require(used+size <= CAP-FINAL and shutil.disk_usage(self.path).free >= FLOOR+size,
                '64MiB/200MiB predispatch gate')

    def write(self, name, value, append=False, final=False):
        parts = Path(name).parts
        require(1 <= len(parts) <= 2 and all(x not in ('', '.', '..') for x in parts) and not Path(name).is_absolute() and
                not (self.path/name).is_symlink() and not (self.path/name).parent.is_symlink(), 'own root/namespace output')
        used = sum(f.stat().st_size for f in self.path.rglob('*') if f.is_file())
        require(used+len(value) <= CAP-(0 if final else FINAL) and
                shutil.disk_usage(self.path).free >= FLOOR+len(value), 'whole run cap including partial/final')
        with (self.path/name).open('ab' if append else 'xb') as f:
            f.write(value); f.flush(); os.fsync(f.fileno())
        fd = os.open((self.path/name).parent, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
        return sha(self.path/name)

    def namespace(self, name):
        require(Path(name).name == name and name not in ('', '.', '..'), 'flat namespace')
        self.check(); (self.path/name).mkdir()
        fd = os.open(self.path, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
        return self.path/name

    def array(self, name, value):
        a = np.asarray(value)
        require(a.dtype.kind in 'fi' and np.isfinite(a).all() and a.nbytes <= 1024**2, 'bounded numerical array')
        buffer = io.BytesIO(); np.save(buffer, a, allow_pickle=False)
        h = self.write(name, buffer.getvalue())
        return dict(path=str(self.path/name), file_sha256=h, array_sha256=array_sha(a), shape=list(a.shape), dtype=str(a.dtype))

    def load_array(self, receipt):
        p = Path(receipt['path']); require(p.is_relative_to(self.path) and len(p.relative_to(self.path).parts) <= 2 and
                                         sha(p) == receipt['file_sha256'], 'own array path/SHA')
        with p.open('rb') as f: a = np.load(f, allow_pickle=False)
        require(list(a.shape) == receipt['shape'] and str(a.dtype) == receipt['dtype'] and
                array_sha(a) == receipt['array_sha256'] and sha(p) == receipt['file_sha256'], 'array entry/readback SHA')
        return a


class Meter:
    def __init__(self, store, job_check=lambda: None):
        self.store = store; self.started = time.monotonic(); self.region = 'setup'; self.episode = None
        c = contract(); self.caps = dict(c['physical_caps'], **c['neural_caps'])
        self.counts = {k: dict(attempted=0, completed=0) for k in self.caps}
        self.pending = {}; self.serial = 0; self.secondary = []
        self.job_check = job_check

    def check(self):
        self.job_check()
        require(time.monotonic()-self.started < 600, 'execution600s cap within whole-job deadline')
        self.store.check()

    def begin(self, resource, units=1):
        self.check(); require(type(units) is int and units > 0 and resource in self.caps, 'known resource/positive units')
        c = self.counts[resource]; require(c['attempted']+units <= self.caps[resource], 'predispatch cap: '+resource)
        ident = self.serial; self.serial += 1; c['attempted'] += units
        e = dict(operation_id=ident, episode_id=self.episode, resource=resource, units=units, region=self.region)
        self.pending[ident] = e; self.store.emit('operation_attempted', **e); return ident

    def end(self, ident):
        require(ident in self.pending, 'unique matching operation return')
        e = self.pending[ident]; self.counts[e['resource']]['completed'] += e['units']
        self.store.emit('operation_completed', **e); del self.pending[ident]

    def call(self, resource, fn):
        ident = self.begin(resource); value = fn(); self.end(ident); return value


def metered_class(base, meter):
    """Local subclass only: intercept virtual entries BEFORE native __init__.

    Native _setup contains no integration (frozen source); hook each newly
    created space before reset's first _set_state. No registry/global patches.
    """
    class Metered(base):
        def __init__(self, *args, **kwargs):
            self._live_meter = meter; self._live_reset_depth = 0; self._live_spaces = []
            self._live_constructor_complete = False
            try:
                super().__init__(*args, **kwargs); self._live_constructor_complete = True
            except BaseException:
                try: self.close()
                except BaseException as exc: meter.secondary.append(dict(stage='constructor_close', message=str(exc)))
                raise

        def _setup(self, *args, **kwargs):
            value = super()._setup(*args, **kwargs); self._live_install_space(); return value

        def _live_install_space(self):
            space = self.space
            if any(x[0] is space for x in self._live_spaces): return
            old = space.step; owned = 'step' in getattr(space, '__dict__', {})
            self._live_spaces.append((space, owned, old))
            def step(*a, **kw):
                require(meter.region in ('setup', 'warmup', 'prefix', 'suffix'), 'no physics during encode/compile/seal')
                return meter.call('physics_step', lambda: old(*a, **kw))
            space.step = step

        def reset(self, *args, **kwargs):
            require(meter.region == 'setup', 'no reset after episode begins')
            self._live_reset_depth += 1
            try: return meter.call('reset', lambda: super(Metered, self).reset(*args, **kwargs))
            finally: self._live_reset_depth -= 1

        def _set_state(self, value):
            require(meter.region == 'setup' and self._live_reset_depth > 0, 'reset-only setter; never restore prefix')
            self._live_install_space()
            return meter.call('set_state', lambda: super(Metered, self)._set_state(value))

        def _set_goal_state(self, value):
            require(self._live_reset_depth > 0 or meter.region == 'goal_bind', 'no task replacement')
            return meter.call('goal_setter', lambda: super(Metered, self)._set_goal_state(value))

        def render(self):
            return meter.call('goal_render' if self._live_reset_depth else 'explicit_render', lambda: super(Metered, self).render())

        def step(self, action):
            require(meter.region in ('warmup', 'prefix', 'suffix'), 'public step blocked until authorization')
            return meter.call('public_step', lambda: super(Metered, self).step(action))

        def close(self):
            errors = []
            try: super().close()
            except BaseException as exc: errors.append(exc)
            for space, owned, original in reversed(self._live_spaces):
                try:
                    if owned: space.step = original
                    else: delattr(space, 'step')
                except BaseException as exc: errors.append(exc)
            self._live_spaces.clear()
            if errors:
                for exc in errors[1:]: meter.secondary.append(dict(stage='restore_space', message=str(exc)))
                raise errors[0]
    return Metered


def native_factory(meter):
    """Only MAIN run imports actual physics. Reject changed native/registration."""
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    from stable_worldmodel.envs.pusht.env import PushT
    import inspect
    require(Path(inspect.getsourcefile(PushT)).resolve() == kernel.moving.ENV and
            sha(kernel.moving.ENV) == kernel.moving.ENV_SHA, 'frozen native base source')
    spec = gym.spec('swm/PushT-v1')
    require(spec.entry_point == 'stable_worldmodel.envs.pusht.env:PushT' and spec.max_episode_steps is None and
            spec.additional_wrappers == (), 'known wrapper/constructor lineage only')
    local = metered_class(PushT, meter)
    return gym.make(replace(spec, entry_point=local), render_mode='rgb_array', resolution=224,
                    with_target=False, render_action=False)


class Sink:
    """Write-only backend namespace. No env, state, sourcepack or evaluator."""
    def __init__(self, store, meter, episode, stage):
        self.__store = store; self.__meter = meter; self.prefix = episode+'_'+stage
        self.output = store.namespace(self.prefix)
        self.operations = set(); self.artifacts = []; self.entries = {}; self.files = {}

    def save_array(self, name, value):
        require(Path(name).name == name and name.endswith('.npy'), 'flat named backend NPY')
        self.__meter.check(); receipt = self.__store.array(self.prefix+'/'+name, value)
        self.artifacts.append(deepcopy(receipt)); return receipt

    def put(self, name, value):
        require(Path(name).name == name and name.endswith('.json'), 'flat backend JSON')
        self.__meter.check(); return self.__store.put(self.prefix+'/'+name, value)

    def emit(self, kind, **fields):
        self.__meter.check(); self.__store.emit('handler_event', namespace=self.prefix, event=dict(kind=kind, **fields))

    def _write(self, name, value):
        require(Path(name).name == name and name not in self.files, 'exclusive backend artifact')
        self.__meter.check(); h = self.__store.write(self.prefix+'/'+name, value)
        self.files[name] = dict(path=str(self.output/name), sha256=h); return h

    def event(self, value):
        """Map actual suffix API/module/math events; never invent neural calls."""
        self.emit('native_event', native=deepcopy(value))
        kind = value['kind']; entry = value.get('entry', {})
        if kind in ('module_attempt', 'module_return'):
            resource = entry['resource']; resource = 'predictor' if resource == 'predict' else resource
            resources = [(resource, 1)]
            if resource == 'predictor' and entry.get('stage') == 'jacobian': resources.append(('ad_forward', 1))
            key = ('module', entry['index']); attempted = kind.endswith('attempt')
        elif kind in ('predict_API_attempt', 'predict_API_return'):
            resources = [('predict', 1)]; key = ('predict_API', entry['index']); attempted = kind.endswith('attempt')
        elif kind in ('math_attempt', 'math_return'):
            resources = [(entry['resource'], 1)]
            if entry['resource'] == 'jacobian':
                require(entry['shape'] == [192, 10] and entry['mathematical_VJP'] == 192, 'declared192 mathematical directions')
                resources.append(('vjp_directions', 192))
            key = ('math', entry['index']); attempted = kind.endswith('attempt')
        elif kind in ('AD_API_attempt', 'AD_API_completed'):
            require(entry['resource'] in ('grad', 'jacobian'), 'no extra backward API')
            resources = [('grad_api' if entry['resource'] == 'grad' else 'jacobian_api', 1)]
            key = ('AD_API', entry['index']); attempted = kind.endswith('attempt')
        else:
            return
        if attempted:
            require(key not in self.entries, 'unique actual entry')
            self.entries[key] = [self.begin(r, units) for r, units in resources]
        else:
            require(key in self.entries and self.entries[key], 'matching actual return')
            for token in self.entries.pop(key): self.end(token)

    def begin(self, resource, units=1):
        require(resource in contract()['neural_caps'], 'handler cannot dispatch physical resource')
        ident = self.__meter.begin(resource, units); self.operations.add(ident); return ident

    def end(self, ident):
        require(ident in self.operations, 'handler owns operation token')
        self.__meter.end(ident); self.operations.remove(ident)

    def complete(self): require(not self.operations and not self.entries, 'all handler operations returned')


class Handler:
    """Own visual encoding, unchanged separately sealed suffix mathematics."""
    def __init__(self, suffix): self.suffix = suffix

    def compile_suffix(self, request, artifact_sink, *, model, device, check_limits):
        return self.suffix.compile_suffix(request, artifact_sink, model=model, device=device, check_limits=check_limits)

    def verify_suffix_seal(self, request, response):
        return self.suffix.verify_suffix_seal(request, response)

    def encode_observation(self, request, artifact_sink, *, model, device, check_limits):
        require(getattr(device, 'type', None) == 'cuda', 'HOST CUDA visual encoder only')
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from PIL import Image
        from experiments.spectral_reachability import p1_constructive_blind_encode_cuda_20261027_v1 as enc
        from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
        check_limits()
        require(request['schema'] == NAME+'.encode_request' and request['observed_step'] in (5, 10) and
                request['request_sha256'] == source.digest({k: v for k, v in request.items() if k != 'request_sha256'}),
                'own current observation encoding request')
        entry = request['png']; path = Path(entry['path'])
        require(path.parent == artifact_sink.output.parent and path.name ==
                request['episode_id']+'_plan_'+f"{request['observed_step']:02}.png" and
                sha(path) == entry['file_sha256'], 'own episode fresh PNG path and external entry SHA')
        with Image.open(path) as image: rgb = np.asarray(image).copy()
        require(rgb.dtype == np.uint8 and rgb.shape == (224, 224, 3) and array_sha(rgb) == entry['pixel_sha256'] and
                sha(path) == entry['file_sha256'], 'actual PNG pixels unchanged')
        enc.frozen(model); before = enc.tensor_hash(model)
        require(before == enc.MODEL_TENSOR_SHA, 'original frozen encoder model')
        # Reuse only frozen preprocessing and visual guard; no suffix or physical work here.
        ledger = enc.Ledger(1, artifact_sink.event, lambda: torch.cuda.synchronize(device),
                            lambda: not torch.is_grad_enabled()); ledger.image = 0; errors = []
        transform = p0.image_transform()
        with sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type='cuda', enabled=False), \
             enc.guard(model, torch, ledger, errors), torch.inference_mode():
            x = p0._image_tensor(rgb, transform=transform, device=device)
            require(x.device == device and tuple(x.shape) == (1, 1, 3, 224, 224), 'one actual CUDA RGB frame')
            artifact_sink.save_array('preprocessed_pixels.npy', x.detach().cpu().numpy().copy())
            z = model.encode({'pixels': x})['emb']; torch.cuda.synchronize(device)
            require(z.device == device and z.dtype == torch.float32 and tuple(z.shape) == (1, 1, 192), 'original projected CLS192')
            latent = native(z[0, 0].detach().cpu().numpy().copy())
        require(not errors and not ledger.blocked and ledger.counts() ==
                {k: dict(attempted=1, completed=1) for k in ('encoder', 'projector')}, 'exact visual calls/guard restore')
        enc.frozen(model); after = enc.tensor_hash(model)
        require(before == after and sha(path) == entry['file_sha256'], 'visual model and pixels unchanged')
        receipt = artifact_sink.save_array('native192.npy', latent)
        response = dict(schema=NAME+'.encode_response', episode_id=request['episode_id'], observed_step=request['observed_step'],
                        request_sha256=request['request_sha256'], latent=receipt, source_verified=True,
                        model_unchanged=True, model_hash_before=before, model_hash_after=after,
                        device_type='cuda', resources=ledger.counts())
        artifact_sink.put('receipt.json', response); check_limits(); return response


def counts_delta(meter, before):
    return {k: {c: meter.counts[k][c]-before[k][c] for c in ('attempted', 'completed')} for k in meter.counts}


def check_delta(delta, wanted):
    require(delta == {k: dict(attempted=wanted.get(k, 0), completed=wanted.get(k, 0)) for k in delta},
            'exact fresh handler operation census; no retry/refund/hidden work')


def state_seal(body):
    return dict(snapshot=kernel.moving.snapshot(body), goal=np.asarray(body.goal_state).tolist(),
                goal_rgb_sha256=kernel.moving.image_sha(body._goal), goal_pose=np.asarray(body.goal_pose).tolist())


def capture(env, obs, row, store, phase, step):
    before = state_seal(env.unwrapped); image = np.asarray(env.render())
    name = row['episode_id']+'_'+phase+f'_{step:02}.png'
    if image.dtype != np.uint8 or image.shape != (224, 224, 3):
        artifact = None
        if image.dtype.kind in 'buif' and image.nbytes < 1024**2:
            artifact = name+'_invalid.npz'; store.arrays(artifact, raw_render=image)
        store.emit('invalid_render', episode_id=row['episode_id'], shape=list(image.shape), dtype=str(image.dtype),
                   raw_render=artifact, state=np.asarray(obs['state']).tolist(), snapshot=state_seal(env.unwrapped))
        raise ValueError('invalid fresh RGB')
    image = image.copy(); h = store.frame(name, image); row['PNG'] += 1
    receipt = dict(phase=phase, step=step, path=str(store.path/name), file_sha256=h,
                   pixel_sha256=array_sha(image), state=np.asarray(obs['state']).tolist(), **state_seal(env.unwrapped))
    receipt['observed_event_index'] = store.index
    row['last_observation'] = receipt; store.emit('fresh_observation', episode_id=row['episode_id'], receipt=receipt)
    require(state_seal(env.unwrapped) == before and
            kernel.moving.exact(np.asarray(obs['state']), np.asarray(env.unwrapped._get_obs())), 'render/observation exactness')
    pixels = kernel.moving.guard(image, obs['state'])
    store.emit('guard_passed', episode_id=row['episode_id'], phase=phase, step=step, canonical_agent_pixels=pixels)
    if phase == 'plan' and step in (0, 5, 10): row['checkpoints'].append(deepcopy(receipt))
    return image, receipt


def identity(layer, row):
    return dict(episode_id=row['episode_id'], source_slot=layer['source_slot'], prefix_arm=layer['arm'], mode=row['mode'],
                parent_task_id=layer['task_id'], task_export_sha256=layer['task_sha256'],
                parent_selection_sha256=layer['selection_sha256'], remaining_public_steps=5)


def native_transition(body, goal, obs, reward, terminated, truncated, row):
    """Evaluator only: completion is not success; retain native stop flags."""
    metrics = kernel.metric(goal, obs['state'])
    success, distance = body.eval_state(goal, obs['state'])
    row.update(final_state=source.vector(obs['state']).tolist(), final_metrics=metrics,
               native_success=bool(success), terminated=bool(terminated), truncated=bool(truncated))
    require(bool(success) == metrics['native_success'] and float(distance) == metrics['native_distance'] and
            bool(terminated) == bool(success) and float(reward) == -float(distance), 'native eval/reward/terminal unchanged')


def encode(backend, runtime, task, layer, row, observation, body, meter, store, step):
    request = dict(schema=NAME+'.encode_request', **identity(layer, row), observed_step=step,
                   png={k: observation[k] for k in ('path', 'file_sha256', 'pixel_sha256')},
                   task_export=layer['task_export'], model_binding_sha256=source.digest(task['model_bindings']))
    request['request_sha256'] = source.digest(request)
    store.put(row['episode_id']+f'_encode{step}_request.json', request)
    before_state = state_seal(body); before = deepcopy(meter.counts); meter.region = 'encoding'
    sink = Sink(store, meter, row['episode_id'], f'encode{step}')
    response = backend.encode_observation(deepcopy(request), sink, **runtime); sink.complete()
    require(state_seal(body) == before_state, 'encoder cannot alter live environment')
    check_delta(counts_delta(meter, before), dict(encoder=1, projector=1))
    require(response['request_sha256'] == request['request_sha256'] and response['episode_id'] == row['episode_id'] and
            response['source_verified'] is True and response['model_unchanged'] is True and
            response['device_type'] == 'cuda', 'fresh CUDA encoder response identity/frozen source')
    require(response['latent'] in sink.artifacts, 'new own encoder artifact; never saved-prefix substitute')
    latent = native(store.load_array(response['latent']))
    store.put(row['episode_id']+f'_encode{step}_response.json', response)
    row[f'encoding{step}'] = deepcopy(response)
    row[f'encoded{step}_event_index'] = store.index
    store.emit('prefix_observed_and_encoded' if step == 5 else 'terminal_encoded', episode_id=row['episode_id'],
               observation_sha256=source.digest(observation), response_sha256=source.digest(response))
    return latent


def compile_and_seal(backend, runtime, task, layer, row, prefix_raw, actual, observation, body, meter, store):
    require(row['mode'] in ('I', 'R') and row['prefix_returned'] == 5 and 'encoding5' in row, 'fresh step5 before compile')
    raw_history = np.concatenate((kernel.moving.warmup_commands()[5:], prefix_raw))
    z = native(actual) if row['mode'] == 'R' else native(np.asarray(layer['imagined_prefix'], np.float32))
    t = task['gn_task']; context = deepcopy(t['root_context'])
    context['latent_history'] = t['root_context']['latent_history'][1:]+[z.tolist()]
    # Public float32 subtract/divide, time-major; not a second /100 or tanh.
    normalized = ((raw_history-np.asarray(t['action_mean'], np.float32))/np.asarray(t['action_std'], np.float32)).reshape(2, 10)
    context['action_history'] = normalized.tolist()
    request = dict(schema=BACKEND+'.packet', mode=row['mode'], context3=context,
                   actualrawhistory10=raw_history.tolist(), oldrawsuffix5=raw5(np.asarray(layer['commands'])[5:]).tolist(),
                   goal=deepcopy(t['goal']), identityQ=dict(kind='identity', dimension=192), modelbinding=deepcopy(task['modelbinding']))
    actual_receipt = row['encoding5']; actual_artifact = actual_receipt['latent']
    seal = dict(**identity(layer, row), step=5, context_kind='observed' if row['mode'] == 'R' else 'imagined',
                context_sha256=packet_digest(context), rawhistory_sha256=packet_digest(request['actualrawhistory10']),
                oldsuffix_sha256=packet_digest(request['oldrawsuffix5']), goal_sha256=packet_digest(request['goal']),
                Q_sha256=packet_digest(request['identityQ']), modelbinding_sha256=packet_digest(request['modelbinding']),
                rgb_file_sha256=observation['file_sha256'], rgb_pixels_sha256=observation['pixel_sha256'],
                encoded_npy_sha256=actual_artifact['file_sha256'],
                encoded_latent_sha256=packet_digest(native(actual).tolist()), encoder_receipt_sha256=packet_digest(actual_receipt),
                imagined_context_sha256=None if row['mode'] == 'R' else packet_digest(context),
                observed_event_index=observation['observed_event_index'], encoded_event_index=row['encoded5_event_index'])
    seal['seal_sha256'] = packet_digest(seal); request['episode_observation_seal'] = seal
    # I carries only HASHES of the paid actual encoding; never its latent/path.
    request_sha = packet_digest(request)
    store.put(row['episode_id']+'_suffix_request.json', request)
    before_state = state_seal(body); before = deepcopy(meter.counts); meter.region = 'compiling'
    sink = Sink(store, meter, row['episode_id'], 'suffix')
    response = backend.compile_suffix(deepcopy(request), sink, **runtime); sink.complete()
    require(state_seal(body) == before_state, 'compiler cannot mutate/reset live environment')
    wanted = dict(predict=5, predictor=5, action_encoder=5, pred_proj=5, ad_forward=1, jacobian=1,
                  svd=1, jacobian_api=1, grad_api=1, vjp_directions=192)
    check_delta(counts_delta(meter, before), wanted)
    require(response['packet_sha256'] == request_sha and response['episode_observation_seal'] == seal and
            response['source_verified'] is True and response['model_unchanged'] is True,
            'own fresh suffix source/model/episode')
    for name, h in response['artifact_sha256'].items():
        require(name in sink.files and sink.files[name]['sha256'] == h, 'durable actual suffix artifacts')
    commands = raw5(response['dispatch_raw5'])
    # Pure verifier receives only the SAME whitelisted request/response.
    verified = backend.verify_suffix_seal(deepcopy(request), deepcopy(response))
    require(np.array_equal(raw5(verified), commands) and state_seal(body) == before_state and counts_delta(meter, before) ==
            {k: dict(attempted=wanted.get(k, 0), completed=wanted.get(k, 0)) for k in meter.counts},
            'pure own verifier; no unmetered replay or response mutation')
    store.put(row['episode_id']+'_suffix_response.json', response)
    seal = dict(schema=NAME+'.dispatch_seal', **identity(layer, row),
                request_sha256=request_sha, response_sha256=packet_digest(response),
                own_observation_sha256=source.digest(observation), encoding_sha256=source.digest(row['encoding5']),
                commands_sha256=array_sha(commands), commands=commands.tolist(), control_returned=5)
    seal['seal_sha256'] = source.digest(seal)
    store.put(row['episode_id']+'_dispatch_seal.json', seal)
    store.emit('suffix_sealed', episode_id=row['episode_id'], seal_sha256=seal['seal_sha256'])
    row.update(suffix_request_sha256=request_sha, dispatch_seal=seal, suffix_namespace=str(sink.output))
    return commands


class DispatchGate:
    def __init__(self, episode, seal, commands):
        require(seal['episode_id'] == episode and seal['remaining_public_steps'] == 5 and
                seal['commands_sha256'] == array_sha(commands) and seal['seal_sha256'] ==
                source.digest({k: v for k, v in seal.items() if k != 'seal_sha256'}), 'same episode durable seal')
        self.episode = episode; self.commands = commands.copy(); self.index = 0

    def take(self, episode, step):
        require(episode == self.episode and step == 6+self.index and self.index < 5, 'single-use ordered step6..10 authorization')
        value = self.commands[self.index].copy(); self.index += 1; return value


def episode(env, ref, source_plan, layer, task, row, backend, runtime, meter, store, paired):
    body = env.unwrapped; slot = ref['slot']; meter.region = 'setup'
    obs, info = env.reset(seed=slot['reset_seed']); original_goal = source.vector(info['goal_state'])
    requested = source.sample_state(original_goal, list(body.block.shapes), dict(root_seed=slot['root_seed']), 0, 0, source_plan)
    obs, info = env.reset(seed=0, options=dict(state=requested.copy(), goal_state=original_goal.copy()))
    image, receipt = capture(env, obs, row, store, 'warmup', 0)
    history, states, snapshots, actual_warm = [image], [source.vector(obs['state'])], [receipt['snapshot']], []
    meter.region = 'warmup'
    for step, command in enumerate(kernel.raw_history(ref['raw_history']), 1):
        obs, reward, terminated, truncated, info = env.step(command.copy()); row['warmup_returned'] += 1
        image, receipt = capture(env, obs, row, store, 'warmup', step)
        actual = np.asarray(body.latest_action).copy(); actual_warm.append(actual)
        store.emit('warmup_return', episode_id=row['episode_id'], step=step, raw=actual.tolist(), raw_bits=actual.tobytes().hex())
        require(kernel.moving.exact(actual, command), 'actual warmup raw command bits')
        if terminated or truncated:
            row.update(status='native_stopped_warmup', terminated=bool(terminated), truncated=bool(truncated),
                       stopped_before_task_goal_bind=True)
            return
        if step in (5, 10):
            history.append(image)
            states.append(source.vector(obs['state'])); snapshots.append(receipt['snapshot'])
    require(kernel.moving.exact(np.asarray(actual_warm), ref['raw_history']) and
            kernel.moving.exact(np.asarray(history), ref['history']) and
            kernel.moving.exact(np.asarray(states), ref['states']) and snapshots == ref['dynamics'], 'exact own0/5/10 moving root reconstruction')
    meter.region = 'goal_bind'; before = kernel.moving.snapshot(body); pose = np.asarray(body.goal_pose).copy()
    meter.call('goal_bind', lambda: kernel.moving.attach_goal(body, ref['goal'], ref['goal_rgb']))
    require(kernel.moving.snapshot(body) == before and np.array_equal(body.goal_pose, pose), 'goal bind no state/pose intervention')
    _, root_observation = capture(env, obs, row, store, 'plan', 0)
    require(root_observation['pixel_sha256'] == array_sha(ref['history'][-1]) and
            source.root_hash(source.vector(obs['state']), ref['goal']) == ref['root_sha256'], 'same source root/goal')
    row['initial_exact'] = True; prefix = raw5(np.asarray(layer['commands'])[:5]); actual_prefix = []
    store.put(row['episode_id']+'_prefix_commit.json', dict(commands=prefix.tolist(), parent_selection_sha256=layer['selection_sha256']))
    meter.region = 'prefix'
    for step, command in enumerate(prefix, 1):
        obs, reward, terminated, truncated, info = env.step(command.copy()); row['prefix_returned'] += 1
        _, observation = capture(env, obs, row, store, 'plan', step)
        actual = np.asarray(body.latest_action).copy(); actual_prefix.append(actual)
        require(kernel.moving.exact(actual, command), 'actual prefix literal raw')
        native_transition(body, ref['goal'], obs, reward, terminated, truncated, row)
        require(np.array_equal(info['goal_state'], ref['goal']) and np.array_equal(info['goal'], ref['goal_rgb']), 'prefix goal unchanged')
        store.emit('control_return', episode_id=row['episode_id'], step=step, command=actual.tolist(),
                   raw_bits=actual.tobytes().hex(), terminated=bool(terminated), truncated=bool(truncated))
        if terminated or truncated:
            row.update(status='native_stopped_prefix', final_state=source.vector(obs['state']).tolist()); return
    row['prefix_pairing'] = dict(pixel_sha256=observation['pixel_sha256'], state=observation['state'],
                                 snapshot=observation['snapshot'], actual_raw_sha256=array_sha(np.asarray(actual_prefix)),
                                 goal_sha256=source.digest(ref['goal'].tolist()))
    key = (layer['source_slot'], layer['arm'])
    if row['mode'] == 'O': paired[key] = deepcopy(row['prefix_pairing'])
    else: require(key in paired and paired[key] == row['prefix_pairing'], 'same-layer O/I/R fresh prefix exactness')
    z5 = encode(backend, runtime, task, layer, row, observation, body, meter, store, 5)
    if row['mode'] == 'O':
        commands = raw5(np.asarray(layer['commands'])[5:])
        seal = dict(schema=NAME+'.dispatch_seal', **identity(layer, row), commands=commands.tolist(),
                    commands_sha256=array_sha(commands), control_returned=5,
                    own_observation_sha256=source.digest(observation))
        seal['seal_sha256'] = source.digest(seal); store.put(row['episode_id']+'_dispatch_seal.json', seal)
        row['dispatch_seal'] = seal; store.emit('suffix_sealed', episode_id=row['episode_id'], seal_sha256=seal['seal_sha256'])
    else:
        commands = compile_and_seal(backend, runtime, task, layer, row, np.asarray(actual_prefix), z5, observation, body, meter, store)
    gate = DispatchGate(row['episode_id'], row['dispatch_seal'], commands); meter.region = 'suffix'
    for step in range(6, 11):
        command = gate.take(row['episode_id'], step)
        store.emit('authorized_dispatch', episode_id=row['episode_id'], step=step, seal_sha256=row['dispatch_seal']['seal_sha256'], command=command.tolist())
        obs, reward, terminated, truncated, info = env.step(command.copy()); row['suffix_returned'] += 1
        _, observation = capture(env, obs, row, store, 'plan', step)
        require(kernel.moving.exact(np.asarray(body.latest_action), command), 'actual sealed suffix literal raw')
        require(np.array_equal(info['goal_state'], ref['goal']) and np.array_equal(info['goal'], ref['goal_rgb']), 'task goal unchanged')
        native_transition(body, ref['goal'], obs, reward, terminated, truncated, row)
        store.emit('control_return', episode_id=row['episode_id'], step=step, command=command.tolist(),
                   raw_bits=np.asarray(body.latest_action).tobytes().hex(), reward=float(reward),
                   terminated=bool(terminated), truncated=bool(truncated))
        if terminated or truncated:
            row['status'] = 'native_stopped_suffix'
            if step < 10: return
            break
    z10 = encode(backend, runtime, task, layer, row, observation, body, meter, store, 10)
    goal = np.asarray(task['gn_task']['goal'], np.float32)
    row.update(actual_Q5=float(np.mean((z5.astype(float)-goal.astype(float))**2)),
               actual_Q10=float(np.mean((z10.astype(float)-goal.astype(float))**2)),
               full_horizon=True, status='completed10')


def execute(p, data, store, backend, factory, *, authorization, runtime, final_model_check, final_evaluator):
    require(authorization == AUTH and p['contract'] == contract(), 'MAIN authorization/fixed complete contract')
    require([(x['source_slot'], x['arm']) for x in p['layers']] == list(LAYERS), 'fixed four layers, no subset')
    for name in ('encode_observation', 'compile_suffix', 'verify_suffix_seal'):
        require(callable(getattr(backend, name, None)), 'approved handler method: '+name)
    rows = [dict(episode_id=f'e{i:02}_{mode}', source_slot=s, prefix_arm=a, mode=mode, status='not_started',
                 first_error=None, secondary_errors=[], warmup_returned=0, prefix_returned=0, suffix_returned=0,
                 initial_exact=False, full_horizon=False, native_success=False, terminated=False, truncated=False,
                 PNG=0, checkpoints=[]) for i, (s, a) in enumerate(LAYERS) for mode in MODES]
    result = dict(schema=NAME+'.report', status='running', rows=rows, errors=[], contract=contract(),
                  fixed_denominator=12, independent_tasks=2, P1_admission=False, P4_admission=False,
                  independent_confirmation=False, training=False)
    require(set(runtime) == {'model', 'device', 'check_limits'} and getattr(runtime['device'], 'type', None) == 'cuda',
            'explicit shared HOST CUDA runtime, never CPU fallback')
    original_check = runtime['check_limits']
    meter = Meter(store, original_check); paired = {}
    def checked_limits(): original_check(); meter.check()
    runtime = dict(runtime, check_limits=checked_limits)
    try:
        store.emit('run_started', prepare=p)
        for row in rows:
            meter.episode = row['episode_id']; meter.region = 'setup'; env = None
            layer = next(x for x in p['layers'] if x['source_slot'] == row['source_slot'] and x['arm'] == row['prefix_arm'])
            try:
                env = meter.call('constructor', lambda: factory(meter))
                require(env.unwrapped._live_meter is meter and env.unwrapped._live_constructor_complete is True,
                        'constructor must be metered before native initialization')
                body = env.unwrapped
                require(body.with_target is False and body.render_action is False and body.render_size == 224 and
                        body.relative is True and body.action_scale == 100 and body.control_hz == 10 and body.dt == .01,
                        'original target-free native physical semantics')
                row['status'] = 'running'
                episode(env, data['references'][row['source_slot']], data['plan'], layer,
                        data['tasks'][row['source_slot']], row, backend, runtime, meter, store, paired)
            except BaseException as exc:
                kernel.retain(row, exc, 'episode')
                if hasattr(exc, 'receipt'):
                    row['handler_failure'] = deepcopy(exc.receipt)
            finally:
                meter.region = 'closing'
                if env is not None:
                    try: env.close()
                    except BaseException as exc: kernel.retain(row, exc, 'close')
                store.emit('episode_closed', receipt=row)
    except BaseException as exc:
        result['errors'].append(dict(stage='execution', type=type(exc).__name__, message=str(exc)))
    try:
        for f, h in p['bindings'].items(): require(sha(f) == h, 'final source seal: '+f)
    except BaseException as exc: result['errors'].append(dict(stage='final_seal', message=str(exc)))
    if meter.secondary: result['errors'].extend(meter.secondary)
    try:
        result['model_final_check'] = final_model_check()
        require(result['model_final_check']['model_unchanged'] is True, 'whole-job frozen model final hash')
    except BaseException as exc: result['errors'].append(dict(stage='final_model', message=str(exc)))
    try:
        result['prefix_lineage'] = final_evaluator(rows)
        require(result['prefix_lineage']['matched'] is True, 'saved original prefix lineage after all control seals')
    except BaseException as exc: result['errors'].append(dict(stage='final_evaluator', message=str(exc)))
    clean = not result['errors'] and not meter.pending and all(r['status'] in
            ('completed10', 'native_stopped_warmup', 'native_stopped_prefix', 'native_stopped_suffix') for r in rows)
    if clean:
        try:
            c = meter.counts
            require(all(v['attempted'] == v['completed'] for v in c.values()) and c['constructor']['completed'] == 12 and
                    c['public_step']['completed'] == sum(r['warmup_returned']+r['prefix_returned']+r['suffix_returned'] for r in rows) and
                    c['physics_step']['completed'] == 2*c['reset']['completed']+10*c['public_step']['completed'] and
                    c['set_state']['completed'] == 2*c['reset']['completed'] and c['goal_render']['completed'] == c['reset']['completed'] and
                    c['goal_setter']['completed'] == c['reset']['completed']+c['goal_bind']['completed'] and
                    c['explicit_render']['completed'] == sum(r['PNG'] for r in rows), 'actual full-chain constructor/reset/physics census')
            if all(r['full_horizon'] for r in rows):
                require(all(c[k] == dict(attempted=v, completed=v) for k, v in contract()['neural_caps'].items()), 'complete12 neural census')
        except BaseException as exc:
            clean = False; result['errors'].append(dict(stage='final_census', message=str(exc)))
    result.update(status='live_feedback_calibration_recorded' if clean else 'partial_or_failed',
                  counts=meter.counts, full_horizons=sum(r['full_horizon'] for r in rows),
                  first_error=next((r['first_error'] for r in rows if r['first_error']), result['errors'][0] if result['errors'] else None),
                  summary=dict(native_success=sum(r['native_success'] and r['terminated'] for r in rows),
                               completed10=sum(r['full_horizon'] for r in rows), denominator=12),
                  execution_wall_seconds=time.monotonic()-meter.started)
    result.update(wall_seconds=time.monotonic()-data.get('job_started', meter.started),
                  wall_scope='run entry through final checks, including source verification, model loading and execution',
                  execution_wall_scope='execute entry through final checks; excludes source verification/model loading')
    if result['wall_seconds'] >= 600:
        result['errors'].append(dict(stage='whole_job_deadline', message='600s including model loading exceeded'))
        result['status'] = 'partial_or_failed'
    try:
        store.emit('run_finished', status=result['status'], counts=meter.counts, errors=result['errors'])
        result.update(journal_sha256=sha(store.path/'journal.jsonl'), journal_chain_tail=store.previous)
        store.put('report.json', result, final=True)
    except BaseException as exc:
        result['errors'].append(dict(stage='final_write', message=str(exc))); result['status'] = 'partial_or_failed'
    return result


def material(manifest, expected):
    path = Path(prior.absolute(str(manifest))); require(path.stat().st_size < PREPARE_CAP and sha(path) == expected, 'external manifest SHA')
    m = read(path)
    prior.fields(m, 'schema sourcepack sourcepack_sha256 source_prepared source_prepare_sha256 planner_output planner_report_sha256 encoder_report encoder_report_sha256 historical_physical historical_physical_sha256 backend_source backend_sha256')
    require(m['schema'] == NAME+'.selection' and m['sourcepack_sha256'] == SOURCE_SHA and
            m['planner_report_sha256'] == PLANNER_SHA and m['encoder_report_sha256'] == ENCODER_SHA and
            m['historical_physical_sha256'] == PHYSICAL_SHA, 'exact approved source/task/plan authority')
    for k in ('sourcepack', 'source_prepared', 'planner_output', 'encoder_report', 'historical_physical', 'backend_source'):
        prior.absolute(m[k])
    backend_path = ROOT/'experiments/spectral_reachability'/(BACKEND+'.py')
    require(Path(m['backend_source']).resolve() == backend_path and m['backend_sha256'] == BACKEND_SHA and
            sha(backend_path) == BACKEND_SHA, 'MAIN exact frozen suffix source')
    from experiments.spectral_reachability import moving_history_planners_cuda_20261027_v1 as planner
    from experiments.spectral_reachability import p1_constructive_moving_history_encode_cuda_20261027_v1 as encoder
    data = kernel.load_source(m['sourcepack'], SOURCE_SHA, m['source_prepared'], m['source_prepare_sha256'])
    plans = planner.consume(m['planner_output'], PLANNER_SHA)['plans']
    require([(x['source_slot'], x['arm']) for x in plans] == list(LAYERS), 'all four original selected plans')
    bindings = dict(data['bindings']); layers = []
    backend = importlib.import_module('experiments.spectral_reachability.'+BACKEND)
    for name in ('compile_suffix', 'verify_suffix_seal'):
        require(callable(getattr(backend, name, None)), 'suffix implementation not yet complete: '+name)
    modelbinding = backend.modelbinding()
    bindings.update(modelbinding['source_closure']); bindings.update(modelbinding['model_files'])
    rpath = Path(m['planner_output'])/'report.json'; r = read(rpath); prep = read(r['prepared'])
    bindings.update(prep['bindings_sha256']); bindings[str(rpath)] = PLANNER_SHA
    bindings[r['prepared']] = r['prepare_sha256']
    bindings.update({str(rpath.parent/n): h for n, h in r['artifact_sha256'].items()})
    for plan in plans:
        ref = data['references'][plan['source_slot']]; prior.task_bridge(plan, ref, bindings)
        interface = encoder.load_task(plan['task_export'], plan['task_sha256'])
        prediction = plan['prediction']
        require(prediction['latent_history'][0] == interface['gn_task']['root_context']['latent_history'][-1] and
                len(prediction['latent_history']) == 3, 'H2 final states chronology [root,prefix,terminal]')
        imagined = native(np.asarray(prediction['latent_history'][1], np.float32))
        layers.append(dict(source_slot=plan['source_slot'], arm=plan['arm'], task_id=plan['task_id'], task_export=plan['task_export'],
                           task_sha256=plan['task_sha256'], selection_sha256=plan['selection_sha256'],
                           commands=kernel.commands(plan['commands']).tolist(), imagined_prefix=imagined.tolist()))
    for f, h in ((path, expected), (Path(m['encoder_report']), ENCODER_SHA), (Path(m['historical_physical']), PHYSICAL_SHA),
                 (backend_path, m['backend_sha256']), (ROOT.parent/'docs/D2P_MOVING_OBSERVED_FEEDBACK_PROTOCOL_20261027.md', None),
                 (Path(__file__), None), (ROOT/'tests'/('test_'+NAME+'.py'), None), (ROOT/'innerdocs'/(NAME+'.md'), None)):
        h = sha(f) if h is None else h; require(sha(f) == h, 'prepare binding'); bindings[str(f.resolve())] = h
    for f in prior.physical_files(): bindings[str(f)] = sha(f)
    for f, h in bindings.items(): require(sha(f) == h, 'prepare closure SHA')
    return dict(schema=NAME+'.prepare', manifest=str(path), manifest_sha256=expected, layers=layers,
                contract=contract(), bindings=bindings, modelbinding=modelbinding,
                backend_source=str(backend_path), backend_sha256=m['backend_sha256'])


def prepare(manifest, expected, output):
    p = material(manifest, expected); payload = raw(p); path = Path(output)
    require(len(payload) < PREPARE_CAP and shutil.disk_usage(path.parent).free >= FLOOR+len(payload), 'small exclusive prepare')
    with path.open('xb') as f: f.write(payload); f.flush(); os.fsync(f.fileno())
    fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
    return dict(prepared=str(path.resolve()), prepare_sha256=sha(path), bytes=len(payload))


def verify(path, expected):
    require(Path(path).stat().st_size < PREPARE_CAP and sha(path) == expected, 'external prepare SHA')
    p = read(path); require(p == material(p['manifest'], p['manifest_sha256']), 'entire prepare reconstruction'); return p


def run(prepared, expected, output, *, authorization):
    job_started = time.monotonic()
    require(authorization == AUTH, 'MAIN explicit CUDA+physics authorization')
    p = verify(prepared, expected); m = read(p['manifest'])
    data = kernel.load_source(m['sourcepack'], SOURCE_SHA, m['source_prepared'], m['source_prepare_sha256'])
    from experiments.spectral_reachability import p1_constructive_moving_history_encode_cuda_20261027_v1 as encoder
    data['tasks'] = {}
    for layer in p['layers']:
        interface = encoder.load_task(layer['task_export'], layer['task_sha256'])
        data['tasks'][layer['source_slot']] = dict(gn_task=interface['gn_task'], model_bindings=encoder.MODELS,
                                                 modelbinding=p['modelbinding'])
    store = Store(output); store.put('execution_binding.json', dict(prepared=str(Path(prepared).resolve()), prepare_sha256=expected))
    require(sha(p['backend_source']) == p['backend_sha256'], 'backend SHA before import')
    suffix = importlib.import_module('experiments.spectral_reachability.'+BACKEND)
    backend = Handler(suffix)
    model = None
    try:
        import torch
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
        require_cuda_allocation('cuda:0'); device = torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        def check_limits():
            require(time.monotonic()-job_started < 600 and torch.cuda.memory_allocated(device) <= 3*1024**3 and
                    torch.cuda.memory_reserved(device) <= 4*1024**3, '600s/3GiBallocated/4GiBreserved')
            store.check()
        check_limits()
        model = load_compatible_pretrained(encoder.POLICY, device=device).eval().requires_grad_(False)
        check_limits()
        encoder.low.frozen(model); before = encoder.low.tensor_hash(model)
        require(before == p['modelbinding']['tensor_sha256'], 'original frozen model actual hash')
        def final_model():
            check_limits(); encoder.low.frozen(model); after = encoder.low.tensor_hash(model)
            return dict(before=before, after=after, model_unchanged=before == after)
        def final_evaluator(rows):
            # Post-control evaluator only. Historical terminal outcomes never enter request/model/selection.
            require(sha(m['historical_physical']) == PHYSICAL_SHA, 'historical physical external SHA')
            old = read(m['historical_physical']); old_rows = {(x['root_slot'], x['arm']): x for x in old['rows']}
            checks = []
            for row in rows:
                actual = next((x for x in row['checkpoints'] if x['step'] == 5), None)
                if actual is None: continue
                previous = next(x for x in old_rows[row['source_slot'], row['prefix_arm']]['checkpoints'] if x['step'] == 5)
                checks.append(dict(episode_id=row['episode_id'], matched=all(actual[k] == previous[k]
                                                                           for k in ('state', 'snapshot', 'pixel_sha256'))))
            require(sha(m['historical_physical']) == PHYSICAL_SHA, 'historical final TOCTOU')
            return dict(matched=all(x['matched'] for x in checks), checks=checks,
                        missing_episodes=[r['episode_id'] for r in rows if not any(x['step'] == 5 for x in r['checkpoints'])])
        return execute(p, dict(data, job_started=job_started), store, backend, native_factory, authorization=authorization,
                       runtime=dict(model=model, device=device, check_limits=check_limits),
                       final_model_check=final_model, final_evaluator=final_evaluator)
    except BaseException as exc:
        failure = dict(schema=NAME+'.startup_failure', errors=[dict(stage='startup', type=type(exc).__name__, message=str(exc))],
                       fixed_denominator=12, rows=[dict(source_slot=s, prefix_arm=a, mode=mode, status='not_started')
                                                   for s, a in LAYERS for mode in MODES], model_loaded=model is not None)
        store.put('startup_failure.json', failure, final=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('command', choices=('prepare', 'verify', 'run'))
    for key in ('manifest', 'manifest-sha', 'prepared', 'prepare-sha', 'output'): parser.add_argument('--'+key)
    parser.add_argument('--authorize-cuda-and-physics', action='store_true'); a = parser.parse_args()
    if a.command == 'prepare': result = prepare(a.manifest, a.manifest_sha, a.output)
    elif a.command == 'verify': result = dict(schema=verify(a.prepared, a.prepare_sha)['schema'])
    else:
        require(a.authorize_cuda_and_physics, 'MAIN CLI authorization')
        result = run(a.prepared, a.prepare_sha, a.output, authorization=AUTH)
    print(source.canonical({k: v for k, v in result.items() if k != 'rows'}))
    if a.command == 'run' and result['status'] != 'live_feedback_calibration_recorded': raise SystemExit(2)


if __name__ == '__main__': main()
