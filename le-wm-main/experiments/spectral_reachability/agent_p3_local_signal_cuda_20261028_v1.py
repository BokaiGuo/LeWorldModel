"""Same-node H1/H3 saved local signals. Only MAIN's explicit run loads CUDA.

No generate, search, physics, fitting, or scientific admission. Import, prepare,
verify and consume use metadata/NumPy only. All output is new and exclusive.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from experiments.spectral_reachability import agent_p3_runtime_contract_20261028_v1 as core

NAME = 'agent_p3_local_signal_cuda_20261028_v1'
REVISION = 'same_task_runtime_decisions_v2'
ROOT = core.ROOT
UPSTREAM = ROOT/'results/agent_p3_runtime_contract_20261028_v1_prepare01.json'
UPSTREAM_SHA = '52abdf575ba6fdff35112fda17624b356efb9141c4b339e31e38e7545a4c7448'
PRIOR = ROOT/'results/agent_p3_resume_20261027_next_prepare01.json'
PREDECESSOR = ROOT/f'results/{NAME}_prepare01.json'
PREDECESSOR_SHA = '9dccc2f8193c263e1738b7dbf4a0ba2aaa6d1c0f4c01ddc0a270c78bccd6c71b'
FLAGS = dict(core.FLAGS, scientific_admission=False, execution_authorized=False)
require, canonical, digest = core.require, core.canonical, core.digest


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024**2), b''): h.update(b)
    return h.hexdigest()


def unique_pairs(items):
    out = {}
    for k, v in items:
        require(k not in out, 'duplicate JSON key'); out[k] = v
    return out


def read(path, cap=16*1024**2):
    p = Path(path); require(p.stat().st_size <= cap, 'bounded JSON')
    result = json.loads(p.read_bytes(), object_pairs_hook=unique_pairs)
    canonical(result); return result


def blob(value):
    a = np.ascontiguousarray(value)
    require(a.dtype in (np.dtype('float32'), np.dtype('float64')) and np.isfinite(a).all(), 'native finite float array')
    raw = a.tobytes()
    return dict(shape=list(a.shape), dtype=a.dtype.str, sha256=hashlib.sha256(raw).hexdigest(),
                data=base64.b64encode(raw).decode('ascii'))


def unblob(b):
    core.keys(b, ('shape', 'dtype', 'sha256', 'data'))
    require(b['dtype'] in ('<f4', '<f8') and type(b['shape']) is list and
            all(type(x) is int and x > 0 for x in b['shape']), 'exact positive native float layout')
    raw = base64.b64decode(b['data'], validate=True)
    require(len(raw) <= 8*1024**2 and len(raw) == int(np.prod(b['shape']))*np.dtype(b['dtype']).itemsize and
            hashlib.sha256(raw).hexdigest() == b['sha256'], 'complete exact buffer')
    a = np.frombuffer(raw, dtype=b['dtype']).reshape(b['shape'])
    require(np.isfinite(a).all(), 'finite saved native buffer'); return a


def profile():
    caps = dict.fromkeys(core.AXES, 0)
    caps.update(predict_API=16, predict_samples=16, predict_context_tokens=48,
        action_encoder_calls=16, action_encoder_samples=16, action_encoder_context_tokens=48,
        predictor_calls=16, predictor_samples=16, predictor_context_tokens=48,
        pred_proj_calls=16, pred_proj_rows=48, jacobian_API=4, VJP_directions=768,
        grad_API=768, raw_backward_callbacks=768, array_SVD_calls=4, signal_compile_calls=4,
        schedule_decisions=4)
    return dict(resource_caps=caps, wall_seconds=900, output_bytes=256*1024**2,
        final_reserve_bytes=16*1024**2, disk_floor_bytes=200*1024**2,
        cuda_allocated_bytes=3*1024**3, cuda_reserved_bytes=4*1024**3,
        engineering_limits_only=True, TF32=False, SDP='MATH', CPU_neural=False,
        derivatives='192 scalar VJPs/J; 4 J; raw float32 coordinates at zero, no tanh or clipping',
        model_SVD_calls=0, encoder_calls=0, projector_calls=0, physics_steps=0,
        no_retries=True, no_outcome_selection=True)


def own_files():
    return [Path(__file__).resolve(), ROOT/f'tests/{NAME}_test.py', ROOT/f'innerdocs/{NAME}.md']


def prepare_material():
    up = core.verify(UPSTREAM, UPSTREAM_SHA)
    prior = read(PRIOR)
    tasks = [core.validate_task(t) for t in up['tasks']]
    require(len(tasks) == 2 and all(len(t['goal']) == 192 and
            np.array_equal(t['projection'], np.eye(192)) for t in tasks), 'original two known identity-Q tasks')
    require(all(t['binding']['model_sha256'] == prior['expected_model_hash'] for t in tasks), 'original frozen model identity')
    policy = Path(prior['policy']).resolve()
    files = set(map(Path, up['source_sha256']))
    require(sha(PREDECESSOR) == PREDECESSOR_SHA, 'preserve original local-signal prepare01 bytes')
    files.update([*own_files(), UPSTREAM, PRIOR, PREDECESSOR, policy/'weights.pt', policy/'config.json',
        ROOT/'experiments/checkpoint_compat.py', ROOT/'experiments/gpu_preflight.py'])
    # Source-only discovery, without importing torch or allocating/loading a model.
    site = ROOT/'.venv/lib/python3.12/site-packages'
    for package in ('stable_worldmodel', 'stable_pretraining', 'hydra', 'einops', 'transformers', 'torch'):
        directory = site/package
        require(directory.is_dir(), 'installed source required: '+package)
        files.update(directory.rglob('*.py'))
    for rel in ('torch/autograd/__init__.py', 'torch/autograd/graph.py', 'torch/__init__.py',
                'torch/nn/modules/module.py', 'torch/nn/attention/__init__.py'):
        files.add(site/rel)
    bindings = {str(f.resolve()): sha(f) for f in sorted(files)}
    return dict(schema=NAME+'.prepare', contract_revision=REVISION, upstream=str(UPSTREAM), upstream_sha256=UPSTREAM_SHA,
        tasks=tasks, policy=str(policy), expected_model_hash=prior['expected_model_hash'],
        sources=bindings, profile=profile(), denominator=[dict(slot=s, horizon=h) for s in (0, 1) for h in (1, 3)],
        runtime_rule=runtime_rule(tasks), decision_denominator=decision_census(),
        predecessor_prepare=dict(path=str(PREDECESSOR), sha256=PREDECESSOR_SHA,
            reusable_with_current_sources=False, reason='preserved historical source seal; prepare02 binds this authorized continuation'),
        history=dict(historical_resources=up['historical_resources'],
            historical_encoder=up['historical_encoder'], historical_anchor_receipts=up['historical_anchor_receipts']),
        input_scope='fixed original two known tasks; no fresh certificate; no outcome input',
        missing_dependencies=['MAIN audit and explicit HOST CUDA scheduling',
            'actual CUDA trace and strict saved consumption', 'generate/reconfiguration/replay adapters remain absent',
            'independent P1 and applicable P2 admission; certified fresh cohort and full acquisition budget'], **FLAGS)


def exclusive(path):
    p = Path(path).absolute()
    require(p.parent == ROOT/'results' and p.name.startswith(NAME) and not p.exists() and not p.is_symlink(),
            'new exclusive own result prefix required')
    return p


def write_new(path, value):
    p = exclusive(path); raw = canonical(value)
    require(len(raw) <= 8*1024**2 and shutil.disk_usage(p.parent).free >= profile()['disk_floor_bytes']+len(raw), 'metadata cap/floor')
    with p.open('xb') as f:
        require(f.write(raw) == len(raw), 'short write'); f.flush(); os.fsync(f.fileno())
    fsync_dir(p.parent)
    return dict(path=str(p), sha256=sha(p))


def fsync_dir(p):
    fd = os.open(p, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def verify(prepared, expected):
    core.hash_value(expected); require(sha(prepared) == expected, 'external prepare SHA')
    p = read(prepared); actual = prepare_material()
    require(canonical(p) == canonical(actual) and sha(prepared) == expected, 'full original input/source/profile rebuild')
    return p


class Store:
    """One append-only, chained, fsynced native-buffer journal; bounded output."""
    def __init__(self, output, limits, *, clock=time.monotonic, free=None):
        self.output = Path(output); self.limits = limits; self.clock = clock; self.started = clock()
        self.free = free or (lambda: shutil.disk_usage(self.output.parent).free)
        self.memory = None; self.bytes = 0; self.index = 0; self.tail = '0'*64; self.closed = False
        self.output.mkdir(exist_ok=False); fsync_dir(self.output.parent)
        self.stream = (self.output/'events.jsonl').open('xb'); fsync_dir(self.output)

    def check(self, size=0, final=False):
        require(self.bytes+size <= self.limits['output_bytes']-(0 if final else self.limits['final_reserve_bytes']), 'output engineering cap')
        require(self.free() >= self.limits['disk_floor_bytes']+size, 'disk engineering floor')
        if not final:
            require(self.clock()-self.started <= self.limits['wall_seconds'], 'wall engineering cap')
            if self.memory:
                a, r = self.memory()
                require(a <= self.limits['cuda_allocated_bytes'] and r <= self.limits['cuda_reserved_bytes'], 'CUDA memory engineering caps')

    def event(self, value):
        require(not self.closed, 'journal open')
        record = dict(index=self.index, previous=self.tail, payload=deepcopy(value))
        record['sha256'] = digest(record); raw = canonical(record)+b'\n'
        self.check(len(raw)); require(self.stream.write(raw) == len(raw), 'short journal write')
        self.stream.flush(); os.fsync(self.stream.fileno()); self.bytes += len(raw)
        self.tail = record['sha256']; self.index += 1

    def put(self, name, value):
        require(name in ('report.json', 'binding.json'), 'fixed artifact name')
        raw = canonical(value); self.check(len(raw), final=name == 'report.json')
        with (self.output/name).open('xb') as f:
            require(f.write(raw) == len(raw), 'short artifact write'); f.flush(); os.fsync(f.fileno())
        self.bytes += len(raw); fsync_dir(self.output)

    def close(self):
        if not self.closed:
            self.stream.flush(); os.fsync(self.stream.fileno()); self.stream.close(); self.closed = True
            fsync_dir(self.output)


def journal(path):
    records = []; tail = '0'*64
    require(Path(path).stat().st_size <= profile()['output_bytes'], 'bounded journal')
    with Path(path).open('rb') as f:
        for i, line in enumerate(f):
            require(len(line) <= 8*1024**2 and line.endswith(b'\n'), 'bounded complete durable journal line')
            r = json.loads(line, object_pairs_hook=unique_pairs); core.keys(r, ('index', 'previous', 'payload', 'sha256'))
            h = r.pop('sha256')
            require(type(r['index']) is int and r['index'] == i and r['previous'] == tail and digest(r) == h,
                    'complete journal chain/order')
            tail = h; records.append(r['payload'])
    return records, tail


def spectra(task, responses, svd=lambda a: np.linalg.svd(a, full_matrices=False)):
    require([r['horizon'] for r in responses] == [1, 3], 'ordered full H1/H3 denominator')
    q = np.asarray(task['projection'], np.float64); goal = np.asarray(task['goal'], np.float64)
    short = goal-np.asarray(responses[0]['nominal'], np.float64); out = []
    for r in responses:
        h = r['horizon']; n = core.finite(r['nominal'], (len(goal),)); j = core.finite(r['jacobian'], (len(goal), 10*h))
        k = q@j; u, s, _ = svd(k)
        rank = int(np.count_nonzero(s > s[0]*1e-6)) if len(s) and s[0] > 0 else 0
        def metric(residual):
            v = q@residual; energy = float(v@v); perp = v-u[:, :rank]@(u[:, :rank].T@v)
            return dict(residual=v.tolist(), energy=energy,
                chi=0. if energy == 0 else float(np.clip((perp@perp)/energy, 0, 1)))
        out.append(dict(horizon=h, singular_values=s.tolist(), rank=rank,
            condition_number=None if rank < min(k.shape) else float(s[0]/s[-1]),
            own_nominal=metric(goal-n), short_nominal_reference=metric(short)))
    signed = out[0]['short_nominal_reference']['chi']-out[1]['short_nominal_reference']['chi']
    return dict(spectra=out, signed_gain=signed, scheduling_gain=max(0., signed),
        coordinate_units='d latent / d raw action; Q=original identity; float64 saved-array SVD',
        svd_relative_cutoff=1e-6, cutoff_is_inherited_engineering_policy=True)


def runtime_rule(tasks):
    """Seal existing same-task constants/functions, never tune after signals."""
    return dict(schema=NAME+'.runtime_rule', runtime_source_sha256=sha(core.__file__),
        policies=[asdict(core.policy(core.validate_task(t))) for t in tasks],
        functions=['core.policy', 'core.choose', 'core.request'], methods=['fixed', 'adaptive'],
        acquisition='one shared H1/H3 pair per original task; not two independently executed search arms',
        root_only=True, require_new_paid_signals_at_each_future_child=True,
        composition='only after real compiler-returned exact-closure prefix edges; no child is invented here',
        release='requests only; generate/replay/profile for full search remain missing', **FLAGS)


def decision_census():
    return [dict(slot=s, mode=mode, status='not_started', signal=None, queue=None, error=None,
        model_plan=None) for s in (0, 1) for mode in ('fixed', 'adaptive')]


def runtime_signal(task, responses, diagnostic):
    """Bridge paid same-node arrays to FrozenSignals without another SVD/model.

    Live caller just computed diagnostic; saved caller MUST first strictly
    recompute it from the complete saved arrays. This is not an input verifier
    or a certificate accepting user-provided chi/coverage without provenance.
    """
    task = core.validate_task(task); node = dict(context=task['root_context'], depth=0,
        indices=[], actions=[], path=[], edges=[]); policy = core.policy(task, node)
    core.keys(diagnostic, ('spectra', 'signed_gain', 'scheduling_gain', 'coordinate_units',
        'svd_relative_cutoff', 'cutoff_is_inherited_engineering_policy'))
    require(diagnostic['svd_relative_cutoff'] == policy.svd_relative_cutoff and
        diagnostic['cutoff_is_inherited_engineering_policy'] is True, 'same frozen SVD policy')
    require([r['horizon'] for r in responses] == [1, 3] == [s['horizon'] for s in diagnostic['spectra']], 'complete same-node H1/H3 pair')
    residual = np.asarray(task['projection'], float)@(np.asarray(task['goal'])-np.asarray(responses[0]['nominal']))
    reference = dict(residual=residual.tolist(), energy=float(residual@residual))
    for r, s in zip(responses, diagnostic['spectra'], strict=True):
        core.keys(r, ('binding_sha256', 'parent_context_sha256', 'horizon', 'nominal', 'jacobian', 'producer_sha256'))
        require(type(r['horizon']) is int and r['binding_sha256'] == digest(task['binding']) and
            r['parent_context_sha256'] == digest(node['context']) and r['producer_sha256'] == sha(__file__), 'same original task/model/parent signal provenance')
        core.finite(r['nominal'], (192,), True); core.finite(r['jacobian'], (192, 10*r['horizon']), True)
        require({k: s['short_nominal_reference'][k] for k in reference} == reference, 'both horizons use exact same short reference')
    chi = diagnostic['spectra'][0]['short_nominal_reference']['chi']
    signed = chi-diagnostic['spectra'][1]['short_nominal_reference']['chi']
    require(signed == diagnostic['signed_gain'] and max(0., signed) == diagnostic['scheduling_gain'], 'signed gain preserved, never own-nominal drift')
    signals = core.FrozenSignals(policy.source_seal, chi, diagnostic['scheduling_gain'])
    receipt = dict(schema=NAME+'.runtime_signal', binding_sha256=digest(task['binding']),
        parent_context_sha256=digest(node['context']), responses=deepcopy(responses),
        scheduling_signal=asdict(signals), signed_gain=signed,
        diagnostic_sha256=digest(diagnostic), policy=asdict(policy), no_control_or_safety_guarantee=True)
    return node, signals, receipt


def decide_root(task, mode, node, signals, receipt):
    require(mode in ('fixed', 'adaptive') and node == dict(context=task['root_context'], depth=0,
        indices=[], actions=[], path=[], edges=[]), 'original root only; no invented composition prefix')
    require(receipt['binding_sha256'] == digest(task['binding']) and receipt['parent_context_sha256'] == digest(node['context']) and
        receipt['scheduling_signal'] == asdict(signals) and receipt['policy'] == asdict(core.policy(task, node)), 'exact same-node frozen signal seal and policy')
    decision = core.choose(task, node, signals, mode)
    requests = [core.request(task, node, operation='compile_program_extension', **decision, candidate_index=i,
        prefix_edges=[], prefix_actions=[], signal_sha256=digest(receipt), local_response=deepcopy(receipt['responses']))
        for i in range(decision['candidates'])] if decision['continue_expanding'] else []
    return dict(parent_context_sha256=digest(node['context']), depth=0, decision=decision, requests=requests,
        status='blocked_missing_generate_and_replay', program_plan=None,
        future_children_require_new_paid_signals=True, **FLAGS)


class NativeSignal:
    """Runnable signal-only adapter. Temporary wrappers are instance-local.

    No old module/global changes. Native predictor/projection forwards remain
    untouched delegates, observed at their actual entries and restored on exit.
    """
    kind = 'HOST_CUDA'

    def __init__(self, model, device, task, store):
        require(getattr(device, 'type', None) == 'cuda', 'HOST CUDA only; no CPU fallback')
        self.model, self.device, self.task, self.store = model, device, core.validate_task(task), store
        self.event_sink = store.event; self.ledger = None; self.phase = None; self.step = None
        self.frozen()

    def frozen(self):
        require(all(not m.training for m in self.model.modules()) and
            all(not p.requires_grad and p.device == self.device for p in self.model.parameters()) and
            all(b.device == self.device for b in self.model.buffers()), 'entire native model frozen eval CUDA')

    def arrays(self, kind, operation_index=None, **values):
        import torch
        arrays = {}
        for name, value in values.items():
            require(isinstance(value, torch.Tensor) and value.device == self.device and value.dtype == torch.float32,
                    'actual native CUDA float32 tensor; no cast/precision repair')
            arrays[name] = blob(value.detach().contiguous().cpu().numpy())
        self.store.event(dict(kind=kind, scope=deepcopy(self.ledger.scope), phase=self.phase, step=self.step,
            operation_index=operation_index, arrays=arrays))

    def call(self, resource, fn, inputs, *, shape=None, directions=None):
        index = len(self.ledger.events)
        def actual():
            self.arrays('io_input', operation_index=index, **{f'{resource}:{k}': v for k, v in inputs.items()})
            out = fn()
            self.arrays('io_return', operation_index=index, **{resource+':output': out})
            return out
        return self.ledger.call(resource, actual, shape=shape, directions=directions)

    @contextmanager
    def observe(self):
        restored = []
        try:
            for name in ('action_encoder', 'predictor', 'pred_proj', 'encoder', 'projector'):
                module = getattr(self.model, name); original = module.forward
                had = 'forward' in module.__dict__; old = module.__dict__.get('forward')
                def wrapped(*args, _name=name, _fn=original, **kwargs):
                    require(not kwargs and len(args) == (2 if _name == 'predictor' else 1), 'native positional ABI')
                    shape = list(args[0].shape)
                    def invoke():
                        require(_name not in ('encoder', 'projector'), 'cached signal forbids visual acquisition')
                        return _fn(*args)
                    return self.call(_name, invoke, {str(i): x for i, x in enumerate(args)}, shape=shape)
                restored.append((module, had, old)); module.forward = wrapped
            yield
        finally:
            errors = []
            for module, had, old in reversed(restored):
                try:
                    if had: module.forward = old
                    else: del module.forward
                except BaseException as exc: errors.append(repr(exc))
            require(not errors, 'native wrapper cleanup failure: '+repr(errors))

    def signal(self, req, ledger):
        core.keys(req, ('binding', 'binding_sha256', 'parent_context', 'parent_context_sha256', 'operation', 'horizon'))
        h = req['horizon']; require(type(h) is int and h in (1, 3), 'fixed H1/H3')
        require(req == core.request(self.task, dict(context=self.task['root_context']),
            operation='compile_finite_horizon_J', horizon=h), 'exact fixed same-node binding; no nearby/stale/outcome inputs')
        import torch
        self.frozen(); self.ledger = ledger; self.step = None
        emb = torch.tensor(req['parent_context']['latent_history'], dtype=torch.float32, device=self.device)[None]
        hist = torch.tensor(req['parent_context']['action_history'], dtype=torch.float32, device=self.device)[None]
        mean = torch.tensor(self.task['action_mean'], dtype=torch.float32, device=self.device)
        std = torch.tensor(self.task['action_std'], dtype=torch.float32, device=self.device)
        def terminal(raw_flat, phase):
            self.phase = phase; state, history = emb, hist
            self.arrays('origin', latent=state, history=history, raw_flat=raw_flat, mean=mean, std=std)
            for i, raw in enumerate(raw_flat.reshape(h, 5, 2)):
                self.step = i
                token = ((raw-mean)/std).reshape(10)
                outgoing = torch.cat((history, token[None, None]), dim=1)
                self.arrays('packet', raw=raw, token=token, outgoing=outgoing)
                encoded = self.model.action_encoder(outgoing)
                predicted = self.call('predict', lambda: self.model.predict(state, encoded),
                    dict(latent=state, act_emb=encoded), shape=list(state.shape))
                require(tuple(predicted.shape) == (1, 3, 192), 'full native prediction output')
                state = torch.cat((state[:, 1:], predicted[:, -1:]), dim=1); history = outgoing[:, 1:]
                self.arrays('boundary', latent=state, history=history)
            return state[0, -1]
        with self.observe():
            zero = emb.new_zeros(10*h)
            with torch.no_grad(): nominal = terminal(zero, 'nominal')
            self.arrays('nominal', value=nominal)
            def jacobian():
                with torch.enable_grad():
                    raw = zero.detach().clone().requires_grad_(True)
                    def raw_hook(value):
                        return ledger.call('raw_backward', lambda: self._raw_return(value))
                    handle = raw.register_hook(raw_hook)
                    try:
                        ad = terminal(raw, 'AD'); self.step = None; self.arrays('AD_terminal', value=ad)
                        require(torch.equal(ad, nominal), 'independent nominal/AD forward exact at same raw buffer')
                        columns = []
                        for direction in range(192):
                            basis = torch.zeros_like(ad); basis[direction] = 1
                            self.store.event(dict(kind='direction', scope=deepcopy(ledger.scope), direction=direction))
                            def grad():
                                return self.call('grad', lambda: torch.autograd.grad(ad, raw, grad_outputs=basis,
                                    retain_graph=direction < 191, create_graph=False, allow_unused=False)[0],
                                    dict(output=ad, raw=raw, basis=basis))
                            columns.append(ledger.call('vjp', grad, directions=1))
                        return torch.stack(columns)
                    finally: handle.remove()
            jac = ledger.call('jacobian', jacobian)
            self.arrays('jacobian', value=jac)
        self.frozen()
        return dict(binding_sha256=req['binding_sha256'], parent_context_sha256=req['parent_context_sha256'],
            horizon=h, nominal=nominal.detach().cpu().tolist(), jacobian=jac.detach().cpu().tolist(), producer_sha256=sha(__file__))

    def _raw_return(self, value):
        self.arrays('raw_backward', operation_index=self.ledger.events[-1]['index'], value=value); return value


def census():
    return [dict(slot=s, horizon=h, status='not_started', response=None, error=None) for s in (0, 1) for h in (1, 3)]


def execute_panel(p, factory, ledger, store, rows, decision_rows=None):
    """Shared real runner; RAM fixtures exercise this exact denominator loop."""
    responses = {0: [], 1: []}; diagnostics = []
    decision_rows = decision_census() if decision_rows is None else decision_rows
    for row in rows:
        s, h = row['slot'], row['horizon']; ledger.scope = dict(slot=s, horizon=h)
        row['status'] = 'attempted'; store.event(dict(kind='signal_request', row=deepcopy(row)))
        try:
            backend = factory(s)
            req = core.request(p['tasks'][s], dict(context=p['tasks'][s]['root_context']), operation='compile_finite_horizon_J', horizon=h)
            row['response'] = ledger.call('signal_compile', lambda: backend.signal(req, ledger))
            responses[s].append(row['response']); row['status'] = 'returned'
            store.event(dict(kind='signal_return', row=deepcopy(row)))
            if h == 3:
                result = spectra(p['tasks'][s], responses[s], lambda a: ledger.call('array_svd', lambda: np.linalg.svd(a, full_matrices=False)))
                diagnostics.append(dict(slot=s, **result)); store.event(dict(kind='diagnostics', result=diagnostics[-1]))
                node, signals, receipt = runtime_signal(p['tasks'][s], responses[s], result)
                for branch in decision_rows[2*s:2*s+2]:
                    branch.update(status='attempted', signal=deepcopy(receipt))
                    store.event(dict(kind='decision_request', slot=s, mode=branch['mode'], signal_sha256=digest(receipt)))
                    try:
                        branch['queue'] = ledger.call('schedule', lambda: decide_root(p['tasks'][s], branch['mode'], node, signals, receipt))
                        branch['status'] = 'decision_prepared_not_executable'
                        store.event(dict(kind='decision_return', row=deepcopy(branch)))
                    except BaseException as exc:
                        branch.update(status='failed', error=type(exc).__name__+': '+str(exc)); raise
        except BaseException as exc:
            row.update(status='failed', error=type(exc).__name__+': '+str(exc)); raise
    return diagnostics


def model_hash(model):
    # Same digest as original sealed source, plus all buffers incl nonpersistent.
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        a = value.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(a.dtype).encode()); h.update(str(tuple(a.shape)).encode()); h.update(a.numpy().tobytes())
    all_buffers = hashlib.sha256()
    for name, v in model.named_buffers():
        a = v.detach().cpu().contiguous()
        all_buffers.update(canonical([name, str(a.dtype), list(a.shape)])); all_buffers.update(a.numpy().tobytes())
    return dict(original_state_dict=h.hexdigest(), all_buffers=all_buffers.hexdigest())


def run(prepared, expected, output, authorization):
    require(authorization == 'MAIN_APPROVED_HOST_CUDA', 'MAIN source audit and HOST CUDA scheduling required')
    started = time.monotonic()
    p = verify(prepared, expected); output = exclusive(output); store = Store(output, p['profile'])
    store.started = started  # include preallocation saved verification, not only model dispatch.
    ledger = core.Ledger(p['profile']['resource_caps'], sink=store.event); rows = census(); decision_rows = decision_census()
    errors = []; diagnostics = []; model = None; before = after = host = None; sources_ok = False; cleanup_ok = False
    source_before = digest(p['sources']); source_after = None
    try:
        store.put('binding.json', dict(prepared=str(Path(prepared).resolve()), prepare_sha256=expected))
        store.event(dict(kind='denominator', rows=deepcopy(rows)))
        store.event(dict(kind='decision_denominator', rows=deepcopy(decision_rows)))
        store.event(dict(kind='source_before', manifest_sha256=source_before))
        os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        # Loaded native classes/functions must be within the presealed source set.
        host = require_cuda_allocation('cuda:0'); device = torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        store.memory = lambda: (torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device))
        store.check(); model = load_compatible_pretrained(p['policy'], device=device).eval().requires_grad_(False)
        before = model_hash(model); require(before['original_state_dict'] == p['expected_model_hash'], 'original source frozen model hash')
        store.event(dict(kind='model_before', hashes=before)); store.check()
        for module in model.modules():
            file = str(Path(inspect.getfile(type(module))).resolve())
            require(file in p['sources'], 'native model class outside presealed sources: '+file)
        adapters = [NativeSignal(model, device, t, store) for t in p['tasks']]
        with torch.autocast(device_type='cuda', enabled=False), sdpa_kernel(SDPBackend.MATH):
            diagnostics = execute_panel(p, lambda s: adapters[s], ledger, store, rows, decision_rows)
        torch.cuda.synchronize(device); store.check(); cleanup_ok = True
        require(ledger.attempted == p['profile']['resource_caps'] == ledger.completed, 'exact complete native resource vector')
    except BaseException as exc:
        errors.append(dict(stage='execution', type=type(exc).__name__, message=str(exc)))
    finally:
        for row in rows:
            if row['status'] == 'not_started': row['status'] = 'blocked_not_dispatched'
            elif row['status'] == 'attempted': row['status'] = 'failed'
        for branch in decision_rows:
            if branch['status'] == 'not_started': branch['status'] = 'blocked_missing_complete_signal'
            elif branch['status'] == 'attempted': branch['status'] = 'failed'
        if model is not None:
            try:
                after = model_hash(model)
                require(before == after and all(not m.training for m in model.modules()) and
                    all(not v.requires_grad for v in model.parameters()), 'unchanged frozen model including buffers')
                store.event(dict(kind='model_after', hashes=after))
            except BaseException as exc: errors.append(dict(stage='final_model', message=str(exc)))
        try:
            final = verify(prepared, expected); source_after = digest(final['sources'])
            require(source_before == source_after, 'actual source manifests before/after')
            store.event(dict(kind='source_after', manifest_sha256=source_after)); sources_ok = True
        except BaseException as exc: errors.append(dict(stage='final_sources', message=str(exc)))
        try: store.check()
        except BaseException as exc: errors.append(dict(stage='final_limits', message=str(exc)))
        try: store.close()
        except BaseException as exc: errors.append(dict(stage='journal_close', message=str(exc)))
        complete = not errors and sources_ok and cleanup_ok and before is not None and before == after and all(r['status'] == 'returned' for r in rows) and all(b['status'] == 'decision_prepared_not_executable' for b in decision_rows)
        report = dict(schema=NAME+'.report', contract_revision=REVISION, status='signals_saved_not_admitted' if complete else 'partial_or_failed',
            prepared=str(Path(prepared).resolve()), prepare_sha256=expected, rows=rows, diagnostics=diagnostics,
            decision_rows=decision_rows, runtime_rule=p['runtime_rule'],
            resource_receipt=ledger.receipt(), errors=errors, model_before=before, model_after=after,
            source_manifest_before=source_before, source_manifest_after=source_after,
            sources_verified=sources_ok, cleanup_ok=cleanup_ok, host=host, journal_tail=store.tail,
            artifact_sha256={n: sha(output/n) for n in ('events.jsonl', 'binding.json') if (output/n).exists()},
            wall_seconds=store.clock()-store.started, CUDA_peak_bytes=store.memory() if store.memory else None,
            total_pipeline_cost_claim=False, **FLAGS)
        store.put('report.json', report)
    require(complete, 'partial/failed retained; no retry; '+repr(errors))
    return dict(output=str(output), report_sha256=sha(output/'report.json'))


def consume(output, expected):
    output = Path(output).absolute(); core.hash_value(expected)
    require(sha(output/'report.json') == expected, 'external report SHA')
    r = read(output/'report.json')
    core.keys(r, ('schema', 'contract_revision', 'status', 'prepared', 'prepare_sha256', 'rows', 'diagnostics', 'decision_rows', 'runtime_rule', 'resource_receipt',
        'errors', 'model_before', 'model_after', 'source_manifest_before', 'source_manifest_after', 'sources_verified', 'cleanup_ok', 'host', 'journal_tail',
        'artifact_sha256', 'wall_seconds', 'CUDA_peak_bytes', 'total_pipeline_cost_claim', *FLAGS))
    p = verify(r['prepared'], r['prepare_sha256'])
    require(r['contract_revision'] == p['contract_revision'] == REVISION and r['runtime_rule'] == p['runtime_rule'], 'frozen same-task rule revision; no threshold tuning')
    require(r['schema'] == NAME+'.report' and r['status'] == 'signals_saved_not_admitted' and not r['errors'], 'complete engineering result only')
    require(all(r[k] is False for k in FLAGS) and r['sources_verified'] is True and r['cleanup_ok'] is True,
            'no admission or source/cleanup failure')
    require(r['total_pipeline_cost_claim'] is False, 'no unmeasured total pipeline advantage claim')
    require(set(r['artifact_sha256']) == {'events.jsonl', 'binding.json'} and
        {f.name for f in output.iterdir()} == {'report.json', 'events.jsonl', 'binding.json'}, 'complete exact artifact set')
    for name, h in r['artifact_sha256'].items(): require(sha(output/name) == h, 'saved artifact SHA')
    binding = read(output/'binding.json')
    require(binding == dict(prepared=r['prepared'], prepare_sha256=r['prepare_sha256']), 'execution binding')
    events, tail = journal(output/'events.jsonl'); require(tail == r['journal_tail'], 'report journal chain binding')
    require(r['model_before'] == r['model_after'] and r['model_before']['original_state_dict'] == p['expected_model_hash'], 'frozen model/source identity')
    core.audit_ledger(r['resource_receipt'])
    require(r['resource_receipt']['caps'] == p['profile']['resource_caps'] == r['resource_receipt']['attempted'] == r['resource_receipt']['completed'], 'exact complete actual resource vector')
    validate_saved(p, r, events)
    for name, h in r['artifact_sha256'].items(): require(sha(output/name) == h, 'artifact unchanged after consume')
    require(all(sha(f) == h for f, h in p['sources'].items()), 'source closure unchanged after consume')
    require(sha(output/'report.json') == expected, 'report unchanged after consume')
    return dict(schema=NAME+'.consume', status='local_signals_saved_verified_not_admitted', report_sha256=expected,
        prepare_sha256=r['prepare_sha256'], denominator=4, diagnostics=r['diagnostics'],
        decision_denominator=4, decision_summary=[dict(slot=b['slot'], mode=b['mode'], status=b['status'],
            binding_sha256=b['signal']['binding_sha256'], parent_context_sha256=b['queue']['parent_context_sha256'],
            scheduling_signal=b['signal']['scheduling_signal'], signed_gain=b['signal']['signed_gain'],
            decision=b['queue']['decision'], candidate_requests=len(b['queue']['requests']),
            queue_sha256=digest(b['queue'])) for b in r['decision_rows']],
        resources=r['resource_receipt']['completed'], **FLAGS)


def validate_saved(p, r, events):
    allowed = {'denominator', 'decision_denominator', 'decision_request', 'decision_return', 'model_before', 'model_after', 'source_before', 'source_after', 'signal_request', 'signal_return', 'diagnostics',
        'resource_attempt', 'resource_return', 'io_input', 'io_return', 'origin', 'packet',
        'boundary', 'nominal', 'AD_terminal', 'jacobian', 'direction', 'raw_backward'}
    require(all(e.get('kind') in allowed for e in events), 'no unknown/leaking/failed event kinds')
    for e in events:
        if 'arrays' in e: continue
        fields = ('kind', 'entry') if e['kind'].startswith('resource_') else {
            'denominator': ('kind', 'rows'), 'decision_denominator': ('kind', 'rows'),
            'decision_request': ('kind', 'slot', 'mode', 'signal_sha256'), 'decision_return': ('kind', 'row'),
            'model_before': ('kind', 'hashes'), 'model_after': ('kind', 'hashes'),
            'source_before': ('kind', 'manifest_sha256'), 'source_after': ('kind', 'manifest_sha256'),
            'signal_request': ('kind', 'row'), 'signal_return': ('kind', 'row'),
            'diagnostics': ('kind', 'result'), 'direction': ('kind', 'scope', 'direction')}[e['kind']]
        core.keys(e, fields)
    require(events and events[0] == dict(kind='denominator', rows=census()), 'denominator first')
    require([e['rows'] for e in events if e['kind'] == 'decision_denominator'] == [decision_census()], 'full fixed task/method decision denominator')
    require([e['hashes'] for e in events if e['kind'] == 'model_before'] == [r['model_before']], 'actual before-model journal binding')
    require([e['hashes'] for e in events if e['kind'] == 'model_after'] == [r['model_after']], 'actual after-model journal binding')
    require(r['source_manifest_before'] == r['source_manifest_after'] == digest(p['sources']), 'actual before/after source manifest')
    for phase in ('before', 'after'):
        require([e['manifest_sha256'] for e in events if e['kind'] == 'source_'+phase] ==
            [r['source_manifest_'+phase]], 'source journal/report binding')
    for hashes in (r['model_before'], r['model_after']):
        core.keys(hashes, ('original_state_dict', 'all_buffers'))
        for value in hashes.values(): core.hash_value(value)
    host = r['host']
    core.keys(host, ('device', 'index', 'name', 'allocation_sum', 'torch_version', 'torch_cuda', 'visible_nvidia_nodes'))
    require(type(host) is dict and host.get('device') == 'cuda:0' and type(host.get('index')) is int and host['index'] == 0 and
        type(host.get('name')) is str and host['name'] and host.get('allocation_sum') == 4. and
        type(host.get('torch_version')) is str and type(host.get('torch_cuda')) is str and
        type(host.get('visible_nvidia_nodes')) is list and host['visible_nvidia_nodes'], 'actual HOST CUDA preflight metadata, not proof by metadata alone')
    for row in r['rows']:
        core.keys(row, ('slot', 'horizon', 'status', 'response', 'error'))
        require(row['error'] is None, 'no row failure')
    require([e['row'] for e in events if e['kind'] == 'signal_request'] ==
        [dict(v, status='attempted', response=None) for v in census()], 'every fixed request, no retry')
    for e in events:
        if 'arrays' in e:
            core.keys(e, ('kind', 'scope', 'phase', 'step', 'operation_index', 'arrays'))
            require(e['scope'] in [dict(slot=s, horizon=h) for s in (0, 1) for h in (1, 3)] and
                e['phase'] in ('nominal', 'AD') and type(e['arrays']) is dict and e['arrays'], 'scoped native arrays only')
            require(all(unblob(b).dtype == np.float32 for b in e['arrays'].values()), 'no native precision casts')
            if e['kind'] not in ('io_input', 'io_return'):
                names = {'origin': ('latent', 'history', 'raw_flat', 'mean', 'std'), 'packet': ('raw', 'token', 'outgoing'),
                    'boundary': ('latent', 'history'), 'nominal': ('value',), 'AD_terminal': ('value',),
                    'jacobian': ('value',), 'raw_backward': ('value',)}[e['kind']]
                core.keys(e['arrays'], names)
            index = e['operation_index']
            if e['kind'] in ('io_input', 'io_return', 'raw_backward'):
                require(type(index) is int and 0 <= index < len(r['resource_receipt']['events']), 'IO linked to actual paid operation')
                operation = r['resource_receipt']['events'][index]
                require(operation['scope'] == e['scope'] and operation['resource'] in
                    (('raw_backward',) if e['kind'] == 'raw_backward' else ('predict', 'action_encoder', 'predictor', 'pred_proj', 'grad')),
                    'native IO matches exact operation and scope')
                if e['kind'] == 'io_input':
                    names = {'predict': ('latent', 'act_emb'), 'action_encoder': ('0',),
                        'predictor': ('0', '1'), 'pred_proj': ('0',), 'grad': ('output', 'raw', 'basis')}[operation['resource']]
                    require(set(e['arrays']) == {operation['resource']+':'+n for n in names}, 'exact native input signature')
                    if operation['resource'] != 'grad':
                        require(e['arrays'][operation['resource']+':'+names[0]]['shape'] == operation['shape'], 'actual buffer shape equals charged API/sample/token axes')
                elif e['kind'] == 'io_return':
                    require(set(e['arrays']) == {operation['resource']+':output'}, 'exact native output signature')
                else: require(set(e['arrays']) == {'value'}, 'exact raw callback array')
            else: require(index is None, 'non-IO not misbound as operation')
    require([(v['slot'], v['horizon'], v['status']) for v in r['rows']] == [(s, h, 'returned') for s in (0, 1) for h in (1, 3)], 'fixed full denominator; no selected roots')
    require([e['rows'] for e in events if e['kind'] == 'denominator'] == [census()], 'denominator saved before work')
    resource = [e for e in events if e['kind'].startswith('resource_')]
    core.keys(r['resource_receipt'], ('caps', 'attempted', 'completed', 'events', 'wall_ns', 'backward_engine_calls',
        'cuda_ns', 'FLOPs', 'inclusive_operation_wall_ns', 'wall_semantics', 'physics_steps', 'unknowns_are_not_zero'))
    for operation in r['resource_receipt']['events']:
        core.keys(operation, ('index', 'scope', 'resource', 'shape', 'directions', 'delta', 'status', 'started_ns', 'ended_ns', 'error'))
    stack = []
    for e in resource:
        if e['kind'] == 'resource_attempt': stack.append(e['entry']['index'])
        else:
            require(stack and stack.pop() == e['entry']['index'], 'real nested attempt/return order')
    require(not stack, 'all attempts returned')
    linked = {}
    for e in events:
        if e.get('operation_index') is not None:
            linked.setdefault(e['operation_index'], []).append(e['kind'])
    for operation in r['resource_receipt']['events']:
        if operation['resource'] in ('predict', 'action_encoder', 'predictor', 'pred_proj', 'grad'):
            require(linked.get(operation['index']) == ['io_input', 'io_return'],
                    'every actual API has exactly one linked full input and return')
        elif operation['resource'] == 'raw_backward':
            require(linked.get(operation['index']) == ['raw_backward'], 'every raw callback has a saved buffer')
    expected = []
    for e in r['resource_receipt']['events']:
        expected.extend([dict(kind='resource_attempt', entry=dict(e, status='attempted', ended_ns=None)),
                         dict(kind='resource_return', entry=e)])
    # Calls nest, so compare by operation id rather than flattening nested order.
    require(len(resource) == len(expected), 'all actual attempt-return journal records')
    require(sorted(map(canonical, resource)) == sorted(map(canonical, expected)), 'journal vs resource ledger exact equality')
    for slot in (0, 1):
        responses = []
        for row in r['rows'][2*slot:2*slot+2]:
            h = row['horizon']; scope = dict(slot=slot, horizon=h); t = p['tasks'][slot]
            response = row['response']; core.keys(response, ('binding_sha256', 'parent_context_sha256', 'horizon', 'nominal', 'jacobian', 'producer_sha256'))
            require(response['binding_sha256'] == digest(t['binding']) and response['parent_context_sha256'] == digest(t['root_context']) and
                response['horizon'] == h and response['producer_sha256'] == sha(__file__), 'exact signal provenance')
            local = [e for e in events if e.get('scope') == scope]
            def arrays(kind, phase=None):
                return [({k: unblob(v) for k, v in e['arrays'].items()}, e) for e in local
                        if e['kind'] == kind and (phase is None or e.get('phase') == phase)]
            ns = arrays('nominal'); js = arrays('jacobian'); ads = arrays('AD_terminal')
            require(len(arrays('io_input')) == len(arrays('io_return')) == 8*h+192,
                    'all and only paid native module/grad IO')
            require(len(ns) == len(js) == len(ads) == 1, 'one full nominal/J/AD buffer')
            n = ns[0][0]['value']; j = js[0][0]['value']
            require(n.shape == (192,) and j.shape == (192, 10*h) and n.dtype == j.dtype == np.float32 and
                np.array_equal(n, response['nominal']) and np.array_equal(j, response['jacobian']) and np.array_equal(n, ads[0][0]['value']), 'native precision response buffers')
            for phase in ('nominal', 'AD'):
                origin = arrays('origin', phase); packets = arrays('packet', phase); boundaries = arrays('boundary', phase)
                require(len(origin) == 1 and len(packets) == len(boundaries) == h, 'complete original ABI trajectory')
                o = origin[0][0]
                require(np.array_equal(o['latent'], np.asarray(t['root_context']['latent_history'], np.float32)[None]) and
                    np.array_equal(o['history'], np.asarray(t['root_context']['action_history'], np.float32)[None]) and
                    np.array_equal(o['raw_flat'], np.zeros(10*h, np.float32)) and
                    np.array_equal(o['mean'], t['action_mean']) and np.array_equal(o['std'], t['action_std']), 'exact root/normalizer/raw origin')
                latent, hist = o['latent'], o['history']
                inputs, outputs = arrays('io_input', phase), arrays('io_return', phase)
                for i, ((packet, pe), (after, be)) in enumerate(zip(packets, boundaries, strict=True)):
                    require(pe['step'] == be['step'] == i and packet['raw'].dtype == np.float32 and
                        np.array_equal(packet['raw'], o['raw_flat'].reshape(h, 5, 2)[i]), 'actual raw float32 packet')
                    token = ((packet['raw']-o['mean'])/o['std']).reshape(10)
                    outgoing = np.concatenate((hist, token[None, None]), axis=1)
                    require(np.array_equal(token, packet['token']) and np.array_equal(outgoing, packet['outgoing']), 'exact subtract/divide normalization; no /100/clipping')
                    def io(resource, returned=False):
                        candidates = [(a, e) for a, e in (outputs if returned else inputs) if e['step'] == i and any(k.startswith(resource+':') for k in a)]
                        require(len(candidates) == 1, 'one actual module IO per step: '+resource)
                        return candidates[0][0]
                    ae_i, ae_o = io('action_encoder'), io('action_encoder', True)
                    pr_i, pr_o = io('predict'), io('predict', True)
                    pred_i, pred_o = io('predictor'), io('predictor', True)
                    proj_i, proj_o = io('pred_proj'), io('pred_proj', True)
                    require(np.array_equal(ae_i['action_encoder:0'], outgoing) and
                        np.array_equal(pr_i['predict:latent'], latent) and np.array_equal(pr_i['predict:act_emb'], ae_o['action_encoder:output']) and
                        np.array_equal(pred_i['predictor:0'], latent) and np.array_equal(pred_i['predictor:1'], ae_o['action_encoder:output']) and
                        np.array_equal(proj_i['pred_proj:0'], pred_o['predictor:output'].reshape(3, 192)) and
                        np.array_equal(proj_o['pred_proj:output'].reshape(1, 3, 192), pr_o['predict:output']), 'native AE/predict/predictor/pred_proj wiring')
                    require(np.array_equal(after['latent'], np.concatenate((latent[:, 1:], pr_o['predict:output'][:, -1:]), axis=1)) and
                        np.array_equal(after['history'], outgoing[:, 1:]), 'exact latent/history endpoint closure')
                    latent, hist = after['latent'], after['history']
                require(np.array_equal(latent[0, -1], n), 'full saved path terminal')
            require([e['direction'] for e in local if e['kind'] == 'direction'] == list(range(192)), 'all 192 VJP directions')
            grad_i = [a for a, _ in arrays('io_input', 'AD') if 'grad:basis' in a]
            grad_o = [a['grad:output'] for a, _ in arrays('io_return', 'AD') if 'grad:output' in a]
            raw = [a['value'] for a, _ in arrays('raw_backward', 'AD')]
            require(len(grad_i) == len(grad_o) == len(raw) == 192, 'complete AD/raw callbacks')
            for i, (gi, go, rw) in enumerate(zip(grad_i, grad_o, raw, strict=True)):
                require(np.array_equal(gi['grad:basis'], np.eye(192, dtype=np.float32)[i]) and
                    np.array_equal(gi['grad:raw'], np.zeros(10*h, np.float32)) and np.array_equal(gi['grad:output'], n) and
                    np.array_equal(go, rw) and np.array_equal(go, j[i]), 'actual scalar target/raw gradient/J row witness')
            responses.append(response)
        actual = dict(slot=slot, **spectra(p['tasks'][slot], responses))
        require(canonical(actual) == canonical(r['diagnostics'][slot]), 'exact own/common chi, signed gain and full singular spectrum recomputation')
    require([e['row'] for e in events if e['kind'] == 'signal_return'] == r['rows'] and
        [e['result'] for e in events if e['kind'] == 'diagnostics'] == r['diagnostics'], 'saved final rows and diagnostics binding')
    validate_decisions(p, r, events)
    require(type(r['wall_seconds']) in (float, int) and 0 <= r['wall_seconds'] <= p['profile']['wall_seconds'] and
        type(r['CUDA_peak_bytes']) in (list, tuple) and len(r['CUDA_peak_bytes']) == 2 and
        all(type(v) is int and 0 <= v <= p['profile'][key] for v, key in zip(r['CUDA_peak_bytes'], ('cuda_allocated_bytes', 'cuda_reserved_bytes'))), 'actual wall/memory profile')


def validate_decisions(p, r, events):
    require(p['runtime_rule'] == r['runtime_rule'] == runtime_rule(p['tasks']), 'original frozen same-task policies/source')
    require(len(r['decision_rows']) == 4 and [(b['slot'], b['mode']) for b in r['decision_rows']] ==
        [(s, mode) for s in (0, 1) for mode in ('fixed', 'adaptive')], 'all same-task fixed/adaptive branches, no cross-root allocation')
    expected_requests = []
    for slot in (0, 1):
        diagnostic = {k: v for k, v in r['diagnostics'][slot].items() if k != 'slot'}
        responses = [v['response'] for v in r['rows'][2*slot:2*slot+2]]
        node, signals, receipt = runtime_signal(p['tasks'][slot], responses, diagnostic)
        for b in r['decision_rows'][2*slot:2*slot+2]:
            core.keys(b, ('slot', 'mode', 'status', 'signal', 'queue', 'error', 'model_plan'))
            require(b['status'] == 'decision_prepared_not_executable' and b['error'] is None and b['model_plan'] is None,
                'no generated plan or hidden failed decision')
            require(canonical(b['signal']) == canonical(receipt) and canonical(b['queue']) ==
                canonical(decide_root(p['tasks'][slot], b['mode'], node, signals, receipt)), 'exact saved signal to sealed type/horizon/depth and queue recomputation')
            expected_requests.append(dict(kind='decision_request', slot=slot, mode=b['mode'], signal_sha256=digest(receipt)))
    require([e for e in events if e['kind'] == 'decision_request'] == expected_requests and
        [e['row'] for e in events if e['kind'] == 'decision_return'] == r['decision_rows'], 'actual attempted/returned decision journal binding')
    require(sum(e['resource'] == 'schedule' and e['status'] == 'returned' for e in r['resource_receipt']['events']) == 4,
        'all four decisions paid once; signal acquisition not multiplied or free')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); subs = parser.add_subparsers(dest='command', required=True)
    a = subs.add_parser('prepare'); a.add_argument('--output', required=True)
    for name in ('verify', 'run'):
        a = subs.add_parser(name); a.add_argument('--prepared', required=True); a.add_argument('--sha256', required=True)
        if name == 'run':
            a.add_argument('--output', required=True); a.add_argument('--authorization', required=True)
    a = subs.add_parser('consume'); a.add_argument('--output', required=True); a.add_argument('--sha256', required=True)
    args = vars(parser.parse_args(argv)); command = args.pop('command')
    if command == 'prepare': result = write_new(args['output'], prepare_material())
    elif command == 'verify':
        p = verify(args['prepared'], args['sha256']); result = dict(status='prepared_signal_only_not_admitted', denominator=p['denominator'], profile=p['profile'], **FLAGS)
    elif command == 'run': result = run(args['prepared'], args['sha256'], args['output'], args['authorization'])
    else: result = consume(args['output'], args['sha256'])
    print(canonical(result).decode())


if __name__ == '__main__': main()
