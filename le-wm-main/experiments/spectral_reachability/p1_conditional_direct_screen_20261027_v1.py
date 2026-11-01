"""Finite genuine H2 direct optimization; NEVER P1 admission or a bank search.

NativeCuda is an explicit MAIN-only backend, with no model loading at import.
This module supplies direct evidence, NOT the missing independent witness or
threshold authority. All configuration must be frozen before model execution.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import numpy as np

from experiments.spectral_reachability import p1_conditional_task_admission_20261027_v1 as core

NAME = 'p1_conditional_direct_screen_20261027_v1'
require = core.require


def validate_configuration(c, task):
    core.validate_task(task)
    core.fields(c, 'method B1 B2 initial_logits step_scale damping extra_line_search_trials', 'direct configuration')
    require(c['method'] in ('full_H2_GN', 'full_H2_gradient'), 'full H2 direct method required')
    require(core.integer(c['B2'], 2) > core.integer(c['B1'], 1), 'continuous B1 < B2 required')
    require(type(c['extra_line_search_trials']) is int and c['extra_line_search_trials'] == 0,
            'old admission contract permits one fixed proposal only; no extra line-search')
    for k in ('step_scale', 'damping'):
        require(type(c[k]) in (float, int) and np.isfinite(c[k]) and c[k] > 0, 'explicit positive '+k)
    require(isinstance(c['initial_logits'], list) and bool(c['initial_logits']), 'finite explicit restarts required')
    for x in c['initial_logits']:
        a = core.array(x, (2, 10))
        require(np.array_equal(a, a.astype(np.float32)), 'initial logits must be float32 representable')
        require((np.linalg.norm(a, axis=1) <= task['packet_logit_cap']).all(), 'initial per-packet ball')


def budget(c):
    """Exact completed direct work, per task. Native images are separate overhead."""
    r, b = len(c['initial_logits']), c['B2']
    return dict(predict=r*(6*(b+1)+2*b), action_encoder=r*(6*(b+1)+2*b),
                jacobian=r*b, svd=r*b if c['method'] == 'full_H2_GN' else 0,
                vjp_directions=192*r*b, linear_solve=r*b if c['method'] == 'full_H2_GN' else 0,
                proposal=r*b, selection=r*b)


class DiskJournal:
    """Exclusive small output; fsync every attempted work/array/trajectory event.

    No overwrites/resume. Returned failure cannot promise disk durability after
    an I/O failure. Main owns final report/source/model-file revalidation.
    """
    def __init__(self, directory, *, byte_limit, reserve_bytes=200*1024**2):
        self.directory = Path(directory)
        core.integer(byte_limit, 1); core.integer(reserve_bytes, 200*1024**2)
        require(not self.directory.exists(), 'NEW output directory required')
        require(shutil.disk_usage(self.directory.parent).free >= byte_limit+reserve_bytes, 'storage reserve')
        self.directory.mkdir()
        self.path = self.directory/'direct_journal.jsonl'
        self.limit, self.reserve, self.written = byte_limit, reserve_bytes, 0
        with self.path.open('x') as f:
            f.flush(); os.fsync(f.fileno())

    def __call__(self, event):
        raw = (core.canonical(event)+'\n').encode()
        require(self.written+len(raw) <= self.limit, 'artifact byte limit')
        require(shutil.disk_usage(self.directory).free >= self.reserve+len(raw), 'storage reserve')
        with self.path.open('ab') as f:
            f.write(raw); f.flush(); os.fsync(f.fileno())
        self.written += len(raw)


class Meter:
    def __init__(self, caps, sink):
        self.caps, self.sink = caps, sink
        self.events, self.extended = [], []
        self.group = None

    def emit(self, kind, **values):
        self.sink(dict(kind=kind, **deepcopy(values)))

    def begin(self, resource):
        used = sum(e['samples'] for e in self.extended if e['resource'] == resource)
        require(resource in self.caps and used < self.caps[resource], 'predispatch cap '+resource)
        e = dict(index=len(self.extended), phase='direct', evaluation_id=self.group,
                 resource=resource, samples=1, completed=False)
        self.extended.append(e)
        if resource in core.RESOURCES:
            self.events.append(dict(e, index=len(self.events)))
            idx = len(self.events)-1
        else:
            idx = None
        self.emit('operation_attempted', event=e)
        return e, idx

    def end(self, token):
        e, idx = token
        e['completed'] = True
        if idx is not None:
            self.events[idx]['completed'] = True
        self.emit('operation_completed', event=e)

    def call(self, resource, fn):
        token = self.begin(resource)
        value = fn()
        self.end(token)
        return value


def projected_logits(x, cap):
    """Float32 feasible packet balls; numeric inward rounding, not a Q threshold."""
    a = core.array(x, (2, 10)).copy()
    for i in range(2):
        n = np.linalg.norm(a[i])
        if n > cap:
            a[i] *= cap/n
    a = a.astype(np.float32)
    for i in range(2):
        if np.linalg.norm(a[i].astype(float)) > cap:
            a[i] *= np.float32(1-4*np.finfo(np.float32).eps)
    require((np.linalg.norm(a.astype(float), axis=1) <= cap).all(), 'float32 feasible projection')
    return a.astype(float)


def evaluate(task, logits, backend, meter):
    raw = np.tanh(np.asarray(logits, np.float32)).reshape(10, 2)
    # Three genuinely recomputed H2 trajectories. No cached terminal masquerades
    # as replay. Backend receives raw controls, never privileged physical state.
    paths = []
    for mode in ('nominal', 'whole_replay', 'segmented_replay'):
        path = backend.rollout(task, raw, meter, mode)
        meter.emit('prediction_path', mode=mode, contexts=path)
        paths.append(path)
    p = dict(task_id=core.task_id(task), logits=np.asarray(logits).tolist(), commands=raw.tolist(),
             contexts=paths[0], whole_replay=paths[1][-1], segmented_replay=paths[2],
             cost=core.model_cost(task, paths[0][-1]['latent_history'][-1]))
    require(paths[1] == paths[0], 'whole replay chronology mismatch')
    core.check_prediction(task, p)
    meter.emit('prediction_complete', prediction=p)
    return p


def run_direct(task, configuration, backend, sink):
    """Execute finite direct component, catch failures with partial receipts.

    Backend protocol: guard(meter), rollout(task, raw10x2, meter, mode),
    jacobian(task, incumbent, meter)->(terminal192, logitJ192x20).
    Custom backends are trusted engineering mocks, NOT authenticated CUDA runs.
    """
    validate_configuration(configuration, task)
    c, t = deepcopy(configuration), deepcopy(task)
    meter = Meter(budget(c), sink)
    report = dict(schema=NAME+'.direct_component', status='started', task_id=core.task_id(t),
                  configuration=c, configuration_sha256=core.digest(c), direct_runs=[],
                  journal=meter.events, extended_journal=meter.extended, budget=budget(c),
                  error=None, secondary_errors=[], P1_admission=False, model_task_eligible=False,
                  physical_future_labels_read=False,
                  missing_components=['independently_validated_fresh_source_scan',
                                      'independent_witness_producer_and_replay', 'independent_threshold_authority'])
    try:
        meter.emit('direct_started', task=t, configuration=c, budget=budget(c))
        with backend.guard(meter):
            for r, initial in enumerate(c['initial_logits']):
                run = []; report['direct_runs'].append(run)
                meter.group = f'run_{r}_iter_0'
                incumbent = evaluate(t, initial, backend, meter)
                run.append(dict(iteration=0, incumbent=incumbent, proposal=None, accepted=False, derivative=None))
                meter.emit('trajectory_step', restart=r, step=run[-1])
                for i in range(1, c['B2']+1):
                    meter.group = f'opt_{r}_iter_{i}'
                    terminal, j = meter.call('jacobian', lambda: backend.jacobian(t, incumbent, meter))
                    j = core.array(j, (192, 20)); terminal = core.array(terminal, (192,))
                    require(np.array_equal(terminal, incumbent['contexts'][-1]['latent_history'][-1]),
                            'AD forward must bind exact incumbent terminal')
                    q = np.asarray(t['projection']); residual = q@(terminal-np.asarray(t['goal']))
                    a = q@j
                    g = 2*a.T@residual/192
                    d = dict(task_id=report['task_id'], at_logits=deepcopy(incumbent['logits']),
                             at_terminal=terminal.tolist(), raw_terminal_jacobian=j.tolist(), objective_gradient=g.tolist())
                    core.check_derivative(t, d, incumbent)
                    meter.emit('derivative_complete', derivative=d)
                    if c['method'] == 'full_H2_GN':
                        # Ridge for MSE-linearization + damping*||delta||².
                        u, s, vh = meter.call('svd', lambda: np.linalg.svd(a/np.sqrt(192), full_matrices=False))
                        delta = meter.call('linear_solve', lambda: -vh.T@(s*(u.T@(residual/np.sqrt(192)))/(s*s+c['damping'])))
                    else:
                        delta = -g
                    x = meter.call('proposal', lambda: projected_logits(
                        np.asarray(incumbent['logits'])+c['step_scale']*delta.reshape(2, 10), t['packet_logit_cap']))
                    meter.emit('proposal_logits', logits=x.tolist())
                    meter.group = f'run_{r}_iter_{i}'
                    proposal = evaluate(t, x, backend, meter)
                    accepted = meter.call('selection', lambda: bool(proposal['cost'] < incumbent['cost']))
                    if accepted:
                        incumbent = proposal
                    run.append(dict(iteration=i, incumbent=deepcopy(incumbent), proposal=proposal,
                                    accepted=accepted, derivative=d))
                    meter.emit('trajectory_step', restart=r, step=run[-1])
        for resource, cap in meter.caps.items():
            events = [e for e in meter.extended if e['resource'] == resource]
            require(len(events) == cap and all(e['completed'] for e in events), 'complete exact budget '+resource)
        report['status'] = 'direct_component_completed_not_admitted'
    except BaseException as exc:
        report['status'] = 'failed'
        report['error'] = dict(type=type(exc).__name__, message=str(exc))
    try:
        meter.emit('direct_finished', report=report)
    except BaseException as exc:
        error = dict(type=type(exc).__name__, message=str(exc))
        if report['error'] is None:
            report['error'] = error; report['status'] = 'failed'
        else:
            report['secondary_errors'].append(error)
    return report


class NativeCuda:
    """MAIN-owned frozen JEPA native backend, NO loading/fallback/simulator.

    Loader must independently bind model/config files and transform provenance.
    bound_task pins all Q/goal/normalizer/context fields for this invocation.
    Actual entry hooks charge predict/action_encoder incl ADforward. VJP counts
    are mathematical reverse directions, NOT kernel/FLOP equivalence.
    """
    def __init__(self, model, device, bound_task):
        require(getattr(device, 'type', None) == 'cuda', 'explicit CUDA only; no CPU neural fallback')
        import torch
        core.validate_task(bound_task)
        require(not any(m.training for m in model.modules()), 'all modules frozen eval')
        require(all(not p.requires_grad and p.device == device for p in model.parameters()), 'frozen CUDA parameters')
        require(all(b.device == device for b in model.buffers()), 'CUDA buffers')
        self.model, self.device, self.torch = model, device, torch
        self.bound_id = core.task_id(bound_task)

    @contextmanager
    def guard(self, meter):
        hooks = []; pending = {}; primary = None
        require(not hasattr(self, '_active_meter'), 'native backend already active')
        self._active_meter = meter
        def pre(resource):
            def enter(module, args):
                require(module not in pending, 'recursive module entry forbidden')
                pending[module] = meter.begin(resource)
            return enter
        def post(module, args, output):
            token = pending.pop(module, None)
            if token is not None and output is not None:
                meter.end(token)
        def forbid(module, args):
            raise ValueError('new image encoding forbidden during direct optimization')
        try:
            for module, resource in ((self.model.predictor, 'predict'), (self.model.action_encoder, 'action_encoder')):
                hooks.append(module.register_forward_pre_hook(pre(resource)))
                hooks.append(module.register_forward_hook(post, always_call=True))
            for module in (self.model.encoder, self.model.projector):
                hooks.append(module.register_forward_pre_hook(forbid))
            yield
        except BaseException as exc:
            primary = exc
            raise
        finally:
            cleanup = []
            for hook in reversed(hooks):
                try:
                    hook.remove()
                except BaseException as exc:
                    cleanup.append(str(exc))
            del self._active_meter
            if cleanup:
                if primary is not None:
                    primary.add_note('hook cleanup: '+repr(cleanup))
                else:
                    raise RuntimeError('hook cleanup: '+repr(cleanup))

    def _tensor(self, a):
        return self.torch.tensor(a, dtype=self.torch.float32, device=self.device)

    def _start(self, task):
        require(core.task_id(task) == self.bound_id, 'native backend task drift')
        c = task['root_context']
        return self._tensor(c['latent_history'])[None], self._tensor(c['action_history'])[None]

    def _step(self, task, z, h, raw):
        torch = self.torch
        token = (raw.reshape(10)-self._tensor(np.tile(task['action_mean'], 5)))/self._tensor(np.tile(task['action_std'], 5))
        # Receipt-only host comparison; token itself remains in the joint graph.
        expected = core.normalized_tokens(task, raw.detach().cpu().numpy())[0]
        require(np.array_equal(token.detach().cpu().numpy(), expected), 'exact float32 normalization before native forward')
        outgoing = torch.cat((h, token[None, None]), 1)
        meter = self._active_meter
        before = {r: sum(e['samples'] for e in meter.events if e['resource'] == r)
                  for r in ('predict', 'action_encoder')}
        successor = self.model.predict(z, self.model.action_encoder(outgoing))[:, -1:]
        for r in before:
            require(sum(e['samples'] for e in meter.events if e['resource'] == r) == before[r]+1,
                    'native model.predict must enter predictor/action_encoder exactly once')
        require(bool(torch.isfinite(successor).all()), 'nonfinite native successor')
        return torch.cat((z[:, 1:], successor), 1), outgoing[:, 1:]

    def _saved(self, z, h):
        return dict(layout='states_with_between_state_actions_v1', context_length=3, latent_dim=192,
                    action_dim=10, latent_history=z[0].detach().cpu().tolist(), action_history=h[0].detach().cpu().tolist())

    def rollout(self, task, raw, meter, mode):
        with self.torch.no_grad():
            z, h = self._start(task); commands = self._tensor(raw); paths = []
            for k in range(2):
                if mode == 'segmented_replay' and k:
                    # Independent packet boundary reconstruction of the full
                    # saved context. Derivative path NEVER takes this branch.
                    z = self._tensor(paths[-1]['latent_history'])[None]
                    h = self._tensor(paths[-1]['action_history'])[None]
                z, h = self._step(task, z, h, commands[5*k:5*k+5])
                paths.append(self._saved(z, h))
                meter.emit('native_packet', mode=mode, packet=k, context=paths[-1])
            return paths

    def jacobian(self, task, incumbent, meter):
        torch = self.torch
        # NumPy float32 tanh defines exact commands in the admission contract.
        # Differentiate the raw controls at THESE commands, then analytic tanh
        # chain ONCE. No detach/prefix recompilation in the joint H2 graph.
        with torch.enable_grad():
            raw = self._tensor(incumbent['commands']).requires_grad_(True)
            z, h = self._start(task)
            for k in range(2):
                z, h = self._step(task, z, h, raw[5*k:5*k+5])
                meter.emit('jacobian_forward_packet', packet=k, context=self._saved(z, h))
            terminal = z[0, -1]
            value = terminal.detach().cpu().numpy().astype(float)
            meter.emit('jacobian_forward_complete', terminal=value.tolist(), logits=incumbent['logits'])
            require(np.array_equal(value, incumbent['contexts'][-1]['latent_history'][-1]), 'AD nominal mismatch')
            rows = []
            for i in range(192):
                row = meter.call('vjp_directions', lambda: torch.autograd.grad(
                    terminal[i], raw, retain_graph=i < 191, create_graph=False)[0])
                rows.append(row.detach().cpu().numpy().reshape(20).astype(float))
                meter.emit('jacobian_row', row=i, raw_control_derivative=rows[-1].tolist())
            commands = np.asarray(incumbent['commands']).reshape(20)
            j = np.stack(rows)*(1-commands**2)[None]
            meter.emit('joint_logit_jacobian', terminal=value.tolist(), jacobian=j.tolist())
            return value, j


def encode_source(path, plan, exclusions, expected_raw_sha256, template, encode_rgb):
    """Ready-source bridge; encoder callback sees RGB only, NEVER state/pose.

    Source census/journal MUST subsequently be checked by to_admission_scan
    using the returned true task_id map; this receipt alone is not freshness.
    encode_rgb must be native model.encode; mocks validate engineering only.
    Native image cost: 4 encoder +4 projector samples, separately from direct.
    """
    from experiments.spectral_reachability import p1_conditional_procedural_source_20261027_v1 as source
    checked = source.validate_plan(plan, exclusions)
    require(checked['ready'], 'fresh source plan/exclusion readiness required')
    core.hash_value(expected_raw_sha256)
    require(not Path(path).is_symlink() and source.file_sha(path) == expected_raw_sha256, 'raw source SHA drift')
    packet = source.load_raw_source(path, plan)
    require('root_context' not in template and 'goal' not in template and 'root_raw_action_history' not in template,
            'no transplanted context/goal in source task template')
    require(template['environment'] == plan['environment'], 'same source environment')
    goal_bytes = np.ascontiguousarray(packet['goal_rgb']).tobytes()
    require(template['goal_source_sha256'] == core.sha(goal_bytes), 'goal RGB byte binding')
    # Validate ALL template keys/scales before the first native call. Unknown
    # privileged fields cannot slip into encoder work before rejection.
    bootstrap = dict(deepcopy(template), goal=[0.]*192,
                     root_raw_action_history=packet['raw_action_history'].tolist())
    require((core.array(bootstrap['action_std'], (2,)) > 0).all(), 'positive normalizer before token computation')
    bootstrap['root_context'] = dict(layout='states_with_between_state_actions_v1', context_length=3,
        latent_dim=192, action_dim=10, latent_history=np.zeros((3, 192)).tolist(),
        action_history=core.normalized_tokens(bootstrap, packet['raw_action_history']).tolist())
    core.validate_task(bootstrap)
    values = [core.array(encode_rgb(rgb.copy()), (192,)) for rgb in
              [*packet['rgb_history'], packet['goal_rgb']]]
    task = dict(deepcopy(template), goal=values[-1].tolist(), root_raw_action_history=packet['raw_action_history'].tolist())
    task['root_context'] = dict(layout='states_with_between_state_actions_v1', context_length=3, latent_dim=192,
                                action_dim=10, latent_history=[x.tolist() for x in values[:3]],
                                action_history=core.normalized_tokens(task, packet['raw_action_history']).tolist())
    ident = core.task_id(task)
    return dict(task=task, task_id=ident, root_sha256=packet['metadata']['root_sha256'],
                raw_source_sha256=expected_raw_sha256, plan_id=source.digest(plan),
                source_purpose=plan['purpose'], freshness_certified=False,
                history_semantics=packet['metadata']['history_semantics'],
                native_image_overhead=dict(encoder=4, projector=4),
                census_validated=False, P1_admission=False)


def native_image_encoder(model, device, transform, sink):
    """Explicit MAIN factory; four bridge invocations pay four native forwards.

    transform MUST be separately source-bound native image_transform, including
    ImageNet normalization/Resize224. No image calls allowed in run_direct.
    Receipts track actual encoder/projector module entries, not assumptions.
    """
    require(getattr(device, 'type', None) == 'cuda', 'native encoding CUDA only')
    import torch
    require(not any(m.training for m in model.modules()) and
            all(not p.requires_grad and p.device == device for p in model.parameters()), 'frozen CUDA model')
    def encode(rgb):
        require(isinstance(rgb, np.ndarray) and rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[-1] == 3, 'uint8 RGB only')
        counts = dict(encoder=0, projector=0); hooks = []
        def pre(name):
            def enter(module, args):
                require(counts[name] == 0 and args[0].shape[0] == 1, 'one native image per entry')
                counts[name] += 1
                sink(dict(kind='native_image_attempted', resource=name, samples=1))
            return enter
        primary = None
        try:
            for name in counts:
                hooks.append(getattr(model, name).register_forward_pre_hook(pre(name)))
            with torch.no_grad():
                pixels = transform(torch.from_numpy(rgb.copy()).permute(2, 0, 1)).to(device)[None, None]
                value = model.encode({'pixels': pixels})['emb'][0, 0].detach().cpu().numpy()
            require(counts == dict(encoder=1, projector=1), 'native encoder/projector entry count')
            value = core.array(value, (192,))
            sink(dict(kind='native_image_completed', counts=counts, rgb_sha256=core.sha(rgb.tobytes()), latent=value.tolist()))
            return value
        except BaseException as exc:
            primary = exc
            raise
        finally:
            errors = []
            for hook in reversed(hooks):
                try:
                    hook.remove()
                except BaseException as exc:
                    errors.append(str(exc))
            if errors:
                if primary is not None:
                    primary.add_note('image hook cleanup: '+repr(errors))
                else:
                    raise RuntimeError('image hook cleanup: '+repr(errors))
    return encode
