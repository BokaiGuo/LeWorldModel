"""Fixed-root three-boundary feedback runtime. prepare is model/physics free.

Only the main host should execute run after review. This is calibration, never
P1/P4 admission. full10 denotes five primitive commands, NOT ten steps.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from experiments.spectral_reachability import pusht_geometry_feedback_response_20261027_v1 as response
from experiments.spectral_reachability import pusht_feedback_suffix_compile_20261027_v1 as compiler
from experiments.spectral_reachability import p4_feedback_matched_20261027_v2 as feedback
from experiments.spectral_reachability import p3_matched_physical_20261027_v1 as durable
from experiments.spectral_reachability.pusht_rgb_observability_guard import RgbObservabilityGuard

TAG = 'p4_multiround_feedback_runtime_20261027_v1'
REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO/'results/pusht_geometry_feedback_response_20261027_v1_prepare_v1'
ARMS = ('imagined', 'actual')
BOUNDARIES = (5, 10, 15)
ROOT = response.ROOT
sha, read = response.sha, response.read


def content_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sync_dir(path):
    fd = os.open(Path(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write(path, value):
    durable.write(path, value)
    sync_dir(Path(path).parent)


def contract():
    return dict(root_sha256=ROOT, root_index=1, prefix_role='selected', arms=list(ARMS),
        boundaries=list(BOUNDARIES), suffix_packets=3, primitive_commands_per_packet=5,
        planned_steps_per_arm=20, candidates_per_compile=10, chart='full10 tanh logits',
        metric='mean((original_Q @ (original_goal - terminal))**2)',
        selection='stable bank-order nonlinear argmin; independent exact replay before dispatch',
        damping=compiler.packet.DAMPING, logit_l2_cap=compiler.packet.CAP,
        radii=list(compiler.packet.RADII),
        actual_context='own last three observed RGB boundaries and two executed packets',
        imagined_context='own predicted rolling context only; no observed latent substitution',
        observation_cost='both arms encode three RGB frames at each boundary; control ignores embeddings',
        prefix_gate='exact saved selected step5 pixels before first suffix compile/dispatch',
        guard='canonical RoyalBlue guard before each command; partial packet never advances history',
        scheduled_predict_samples=80, scheduled_compiler_predict_samples=78,
        scheduled_jacobians=6, scheduled_scalar_vjp_directions=1152, scheduled_svd_calls=12,
        scheduled_encode_calls=6, scheduled_encode_frames=18, scheduled_selected_steps=40,
        setup_physics='every root-collection/reset/warmup env.step counted separately',
        total_compute_matched=False, outcome_gating=False, training=False,
        P1_admission=False, P4_admission=False, scope='one reused known-root calibration')


def roster():
    return [dict(arm=a, boundary=b, status='blocked_not_reached', root_sha256=ROOT)
            for a in ARMS for b in BOUNDARIES]


def census(slots):
    if len(slots) != 6 or [(s['arm'], s['boundary']) for s in slots] != [(a,b) for a in ARMS for b in BOUNDARIES]:
        raise ValueError('all six ordered compile slots required, including blocked slots')
    if any(s['root_sha256'] != ROOT for s in slots):
        raise ValueError('fixed root changed')
    return dict(branches=2, compile_slots=6, planned_selected_steps=40)


class Journal:
    """Exclusive append-only journal; images and directory entries fsynced first."""
    def __init__(self, output):
        self.output = Path(output)
        self.stream = (self.output/'runtime.events.jsonl').open('x')
        sync_dir(self.output)
        self.index = 0

    def emit(self, value, rgb=None):
        from PIL import Image
        value = deepcopy(value)
        value.update(event_index=self.index, monotonic_seconds=time.monotonic())
        if rgb is not None:
            if np.asarray(rgb).shape != (224,224,3) or np.asarray(rgb).dtype != np.uint8:
                raise ValueError('official uint8 RGB required')
            path = self.output/f'rgb_{self.index:06d}.png'
            with path.open('xb') as stream:
                Image.fromarray(rgb).save(stream, format='PNG')
                stream.flush(); os.fsync(stream.fileno())
            sync_dir(self.output)
            value.update(rgb_path=str(path), rgb_sha256=sha(path),
                pixel_sha256=hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest())
        self.stream.write(json.dumps(value, sort_keys=True, allow_nan=False)+'\n')
        self.stream.flush(); os.fsync(self.stream.fileno())
        self.index += 1
        return value

    def close(self):
        self.stream.close()


def prepare(output, source=SOURCE):
    """Verify original data/witnesses, seal all dependencies; never instantiate env/model."""
    from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal
    from experiments.spectral_reachability import pusht_composition_cuda_replay_v2 as adapter
    from experiments.spectral_reachability.d2p_model_call_instrumentation import PredictInstrumentation
    output, source = Path(output).resolve(), Path(source).resolve()
    if not output.name.startswith(TAG) or output.exists():
        raise ValueError('new exclusive basename output required')
    frozen = response.verify_frozen(source)
    # Recompute original input bindings without opening a simulator or model.
    current, context = response.prepare(frozen['capability'], frozen['root_rgb'])
    if current != frozen:
        raise ValueError('original protocol no longer reproduces')
    base, checked, upstream, config = context[:4]
    case = next(c for c in frozen['cases'] if c['role'] == 'selected')
    original_row = next(r for r in read(frozen['capability'])['rows'] if r['method']=='selected_zero')
    bindings = dict(frozen['bindings'])
    files = set(checked.sources_before_allocate(upstream, config))
    receipt = read(source/'prepare_receipt.json')
    for s in receipt['source_seals']:
        files.update(Path(e['source']) for e in read(Path(s['output_dir'])/'manifest.json')['files'])
        for name in ('manifest.json', 'args.json'):
            p = Path(s['output_dir'])/name; bindings[str(p)] = sha(p)
    for name in ('protocol.json', 'prepare_receipt.json'):
        p = source/name; bindings[str(p)] = sha(p)
    # Resolve installed environment entry point and seal its loaded Python contract
    # BEFORE any root collection/reset/physics; no env constructor here.
    entry = base.gym.spec('swm/PushT-v1').entry_point
    if not isinstance(entry, str) or ':' not in entry:
        raise ValueError('explicit installed environment source entry point required')
    env_module = importlib.import_module(entry.split(':')[0])
    files.add(Path(inspect.getfile(env_module)))
    for name, mod in tuple(sys.modules.items()):
        path = getattr(mod, '__file__', '') or ''
        if path.endswith('.py') and name.startswith(('stable_worldmodel', 'gymnasium.wrappers', 'pymunk')):
            files.add(Path(path))
    files.update([Path(__file__), Path(response.__file__), Path(compiler.__file__),
        Path(compiler.packet.__file__), Path(feedback.__file__), Path(durable.__file__), Path(adapter.__file__),
        Path(inspect.getfile(RgbObservabilityGuard)), Path(inspect.getfile(PredictInstrumentation)),
        REPO/f'tests/test_{TAG}.py', REPO/f'innerdocs/{TAG}.md'])
    if any(sha(p) != h for p,h in bindings.items()):
        raise ValueError('input changed during prepare')
    protocol = dict(schema=TAG, source=str(source), contract=contract(), bindings=bindings,
        root_sha256=ROOT, root_index=1, role='selected', prefix=case['prefix'],
        prefix_rgb=case['prefix_rgb'], between_raw=case['between_raw'],
        root_rgb=frozen['root_rgb'], initial_context=frozen['initial_context'],
        saved_predicted_prefix=case['saved_predicted_prefix'], goal=frozen['goal'], q=frozen['q'],
        policy=frozen['policy'], root_seed=int(context[6]['protocol']['root_seed']),
        physical_goal=original_row['context_identity']['goal'],
        original_root_order=context[8]['roots'], env_entry_point=entry, slots=roster())
    output.mkdir(exist_ok=False)
    write(output/'protocol.json', protocol)
    write(output/'planned_census.json', dict(slots=roster(), **census(roster())))
    files = sorted({Path(p).resolve() for p in files})
    seals = [create_seal(files=files[i:i+32], args_json=json.dumps(dict(protocol_sha256=sha(output/'protocol.json'))),
        output_dir=output/f'source_seal_{i//32}') for i in range(0,len(files),32)]
    result = dict(status='prepared_not_run', protocol_sha256=sha(output/'protocol.json'), source_seals=seals,
        census_sha256=sha(output/'planned_census.json'), actual_predict_calls=0,
        actual_encode_frames=0, actual_physics_steps=0, P1_admission=False, P4_admission=False)
    write(output/'prepare_receipt.json', result)
    return result


def verify_prepared(output):
    from experiments.spectral_reachability.d2p_prerun_source_seal import verify_seal
    output = Path(output)
    r, p = read(output/'prepare_receipt.json'), read(output/'protocol.json')
    if (r['status'] != 'prepared_not_run' or r['protocol_sha256'] != sha(output/'protocol.json')
            or r['census_sha256'] != sha(output/'planned_census.json') or p['contract'] != contract()
            or p['schema'] != TAG or p['slots'] != roster() or not r['source_seals']):
        raise ValueError('prepared protocol/census differs')
    census(read(output/'planned_census.json')['slots'])
    for seal in r['source_seals']:
        verify_seal(output_dir=seal['output_dir'], expected_manifest_sha256=seal['manifest_sha256'])
    if any(sha(path) != h for path,h in p['bindings'].items()):
        raise ValueError('bound source/data/witness changed')
    return p, r


def validate_transition(before, after, terminal=None):
    """Check predicted state shift; action normalization is the sealed CUDA adapter's job."""
    response.array(before['latent_history'], (3,192)); response.array(before['action_history'], (2,10))
    response.array(after['latent_history'], (3,192)); response.array(after['action_history'], (2,10))
    if (after['latent_history'][:2] != before['latent_history'][1:]
            or after['action_history'][0] != before['action_history'][1]
            or terminal is not None and after['latent_history'][-1] != terminal):
        raise ValueError('explicit rolling history shift/replay differs')


def execute_five(env, commands, *, guard, start, state_of, emit, cost):
    raw = response.array(commands, (5,2))
    if np.any(np.abs(raw)>1):
        raise ValueError('bounded five-command packet required')
    rgb = None
    for j, command in enumerate(raw):
        step = start+j+1
        authorized = guard.dispatch(command)
        cost['selected_attempts'] += 1
        emit(dict(kind='step_attempt', primitive_step=step, command=command.tolist(), cost=deepcopy(cost)))
        started = time.monotonic()
        result = env.step(authorized)
        cost['selected_returns'] += 1
        emit(dict(kind='step_returned', primitive_step=step, seconds=time.monotonic()-started, cost=deepcopy(cost)))
        if len(result) != 5:
            raise ValueError('Gymnasium five-field transition required')
        rgb = np.asarray(env.render()).copy()
        emit(dict(kind='observation', primitive_step=step, state=np.asarray(state_of(env)).tolist(),
            reward=float(result[1]), terminated=bool(result[2]), truncated=bool(result[3]), cost=deepcopy(cost)), rgb)
        guard.observe(rgb, primitive_step=step)
        if result[2] or result[3]:
            raise RuntimeError('environment terminated; do not replace with shortened endpoint')
    return rgb


def arm_loop(p, arm, *, env, reset, state_of, backend, emit, slots, cost):
    """Shared real/non-neural-test orchestration. No shortened packet or reselection."""
    row = dict(arm=arm, root_sha256=ROOT, status='blocked', completed_packets=0, failure=None)
    start_attempts, start_returns = cost['selected_attempts'], cost['selected_returns']
    guard = RgbObservabilityGuard(root_id=ROOT, agent_rgb=(65,105,225))
    def event(value, rgb=None):
        return emit(dict(value, arm=arm), rgb)
    try:
        frames, between = reset()
        frames, between = np.array(frames, copy=True), np.array(between, copy=True)
        predicted = deepcopy(p['initial_context'])
        event(dict(kind='initial', state=np.asarray(state_of(env)).tolist()), np.asarray(env.render()).copy())
        guard.observe(np.asarray(env.render()), primitive_step=0)
        prefix = p['prefix']
        # Fixed control prediction prepared before the common original prefix.
        prefix_prediction = backend.replay(predicted, prefix, arm+'_prefix')
        validate_transition(predicted, prefix_prediction)
        response.require_close(prefix_prediction['latent_history'][-1], p['saved_predicted_prefix'], 'original prefix replay')
        terminal_rgb = execute_five(env, prefix, guard=guard, start=0, state_of=state_of, emit=event, cost=cost)
        from PIL import Image
        with Image.open(p['prefix_rgb']) as im:
            expected = np.asarray(im).copy()
        exact = np.array_equal(terminal_rgb, expected)
        event(dict(kind='prefix_identity', primitive_step=5, pixels_exact=bool(exact)))
        if not exact:
            raise ValueError('original selected prefix pixels differ; no suffix compilation')
        frames, between = feedback.advance_observed(frames, between, terminal_rgb, prefix,
            dict(status='completed', completed_commands=5, remaining_commands_not_executed=0))
        predicted = prefix_prediction
        row['completed_packets'] = 1
        for boundary in BOUNDARIES:
            slot = next(s for s in slots if s['arm']==arm and s['boundary']==boundary)
            slot['status'] = 'encoding'
            event(dict(kind='boundary_begin', boundary=boundary, frames_pixel_sha256=[
                hashlib.sha256(np.ascontiguousarray(f).tobytes()).hexdigest() for f in frames],
                between_raw=np.asarray(between).tolist()))
            observed = backend.encode(frames, between, f'{arm}_{boundary}')
            if observed['action_history'] != predicted['action_history']:
                raise ValueError('observed/predicted action history normalization differs')
            chosen_context = observed if arm=='actual' else predicted
            slot['status'] = 'compiling'
            event(dict(kind='planning_context', boundary=boundary, observed_context=observed,
                predicted_context=predicted, chosen_context=chosen_context,
                control_uses_observed_latents=False))
            plan, successor = backend.compile(deepcopy(chosen_context), deepcopy(p), arm, boundary)
            if (plan['context'] != chosen_context or plan['goal'] != p['goal'] or plan['q'] != p['q']
                    or plan['prefix'] != p['prefix'] or plan['arm'] != arm or plan['role'] != 'selected'):
                raise ValueError('compiler live context/original metric/prefix binding differs')
            winner = compiler.verify_plan(plan)
            commands = winner['commands'][5:]
            validate_transition(chosen_context, successor, winner['predicted_terminal'])
            selection = dict(root_sha256=ROOT, arm=arm, boundary=boundary,
                context_sha256=content_sha(chosen_context), metric_sha256=content_sha([p['goal'],p['q']]),
                incoming_between_raw=np.asarray(between).tolist(), compiler_plan=plan,
                commands=commands, successor=successor)
            selection_hash = content_sha(selection)
            # Full candidates/Jacobian and selection receipt fsynced BEFORE dispatch.
            event(dict(kind='selection', selection_sha256=selection_hash, **selection))
            slot.update(status='selected', selection_sha256=selection_hash, selected_candidate_id=winner['candidate_id'])
            terminal_rgb = execute_five(env, commands, guard=guard, start=boundary,
                state_of=state_of, emit=event, cost=cost)
            frames, between = feedback.advance_observed(frames, between, terminal_rgb, commands,
                dict(status='completed', completed_commands=5, remaining_commands_not_executed=0))
            predicted = successor
            slot['status'] = 'completed'
            row['completed_packets'] += 1
            event(dict(kind='packet_complete', boundary=boundary, next_boundary=boundary+5,
                next_between_raw=np.asarray(between).tolist(), predicted_context=predicted))
        row['status'] = 'completed'
    except (Exception, KeyboardInterrupt) as exc:
        row.update(status='aborted', failure=durable.failure(exc, 'arm_loop'))
        for slot in slots:
            if slot['arm']==arm and slot['status'] in ('encoding','compiling','selected'):
                slot.update(status='aborted', failure=row['failure'])
        event(dict(kind='arm_failure', failure=row['failure'], cost=deepcopy(cost)))
    finally:
        row.update(selected_attempts=cost['selected_attempts']-start_attempts,
            selected_returns=cost['selected_returns']-start_returns)
        row['guard'] = guard.receipt()
        event(dict(kind='arm_complete', row=row, slots=slots, cost=deepcopy(cost)))
    return row


class CudaBackend:
    def __init__(self, model, device, base, counter, emit, cost):
        from experiments.spectral_reachability.pusht_composition_cuda_replay_v2 import CudaReplay
        self.model, self.device, self.base = model, device, base
        self.counter, self.emit, self.cost = counter, emit, cost
        self.transition = compiler.packet.make_transition(model, device)
        self.adapter = CudaReplay(model, device)
        self.transform = base.p0.image_transform()

    def snapshot(self, kind, **kwargs):
        self.cost['predict_samples'] = sum(r['predict_samples'] for r in self.counter.records)
        self.emit(dict(kind=kind, predict=self.counter.records, cost=deepcopy(self.cost), **kwargs))

    def replay(self, context, commands, method):
        self.snapshot('replay_begin', method=method)
        try:
            with self.counter.phase('evaluation', method=method, horizon_tokens=1):
                return self.adapter(deepcopy(context), commands)
        finally:
            self.snapshot('predict_receipt', method=method)

    def encode(self, frames, between, method):
        import torch
        self.cost['encode_attempts'] += 1
        self.cost['encode_frames_attempted'] += len(frames)
        self.snapshot('encode_begin', method=method)
        try:
            result = feedback.encode_cuda(self.model, self.device, frames, between, self.base.p0, self.transform)
            torch.cuda.synchronize(self.device)
            self.cost['encode_completed'] += 1
            return result
        finally:
            self.snapshot('encode_receipt', method=method)

    def compile(self, context, protocol, arm, boundary):
        import torch
        method = f'{arm}_{boundary}'
        def values(t):
            return t.detach().cpu().double().numpy().tolist()
        emb = torch.tensor(context['latent_history'], dtype=torch.float32, device=self.device)[None]
        history = torch.tensor(context['action_history'], dtype=torch.float32, device=self.device)[None]
        def terminal(u):
            return self.transition(emb, history, torch.tanh(u).reshape(5,2))[0][0,-1]
        try:
            with self.counter.phase('planning', method=method+'_compile', horizon_tokens=1):
                with torch.no_grad():
                    nominal = terminal(emb.new_zeros(10))
                self.cost['jacobian_attempts'] += 1
                self.snapshot('jacobian_begin', method=method)
                jac = torch.autograd.functional.jacobian(terminal, emb.new_zeros(10).requires_grad_(True), vectorize=True)
                self.cost['jacobian_completed'] += 1
            self.snapshot('derivative_receipt', method=method, nominal_terminal=values(nominal), jacobian=values(jac))
            q, goal = np.asarray(protocol['q']), np.asarray(protocol['goal'])
            self.cost['bank_attempts'] += 1
            self.snapshot('bank_begin', method=method)
            bank = compiler.packet.spectral_bank(values(jac), values(nominal), goal, q)
            self.cost['bank_completed'] += 1
            candidates = []
            with self.counter.phase('planning', method=method+'_bank', horizon_tokens=1), torch.no_grad():
                for i, params in enumerate(bank['params']):
                    u = torch.tensor(params, dtype=emb.dtype, device=self.device)
                    pred = values(terminal(u))
                    candidate = dict(candidate_id=f'suffix_{i}', kind=bank['kinds'][i], suffix_params=values(u),
                        commands=protocol['prefix']+values(torch.tanh(u).reshape(5,2)), predicted_terminal=pred,
                        nonlinear_terminal_cost=float(np.mean((q@(goal-np.asarray(pred)))**2)))
                    candidates.append(candidate)
                    self.snapshot('candidate', method=method, candidate=candidate)
            winner = min(candidates, key=lambda c:c['nonlinear_terminal_cost'])
            successor = self.replay(context, winner['commands'][5:], method+'_independent_replay')
            plan = dict(root_sha256=ROOT, root_index=1, role='selected', arm=arm, prefix=protocol['prefix'],
                context=deepcopy(context), goal=protocol['goal'], q=protocol['q'], nominal_terminal=values(nominal),
                jacobian=values(jac), bank=bank, candidates=candidates, selected_candidate_id=winner['candidate_id'],
                independent_replay_exact=successor['latent_history'][-1]==winner['predicted_terminal'])
            compiler.verify_plan(plan)
            return plan, successor
        finally:
            self.snapshot('compile_receipt', method=method)


def new_cost():
    return dict(predict_samples=0, selected_attempts=0, selected_returns=0, setup_attempts=0, setup_returns=0,
        encode_attempts=0, encode_completed=0, encode_frames_attempted=0,
        visual_encoder_attempts=0, visual_encoder_completed=0,
        visual_encoder_frames_attempted=0, visual_encoder_frames_completed=0,
        jacobian_attempts=0, jacobian_completed=0, bank_attempts=0, bank_completed=0,
        action_encoder_attempts=0, action_encoder_completed=0, action_encoder_input_tokens=0)


def run(output):
    """Main-host-only CUDA+physics entry point; NEVER called by prepare/tests."""
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
    from experiments.spectral_reachability.d2p_model_call_instrumentation import PredictInstrumentation
    from experiments.spectral_reachability.run_pusht_sampling_comparison_20261027_v2 import verify_rgb_context
    output = Path(output).resolve()
    p, prepared = verify_prepared(output)
    # Exclusive launch prevents rerunning a partially dispatched protocol.
    write(output/'launch_receipt.json', dict(protocol_sha256=prepared['protocol_sha256']))
    slots, rows, cost = roster(), [], new_cost()
    journal = Journal(output)
    journal.emit(dict(kind='all_slots_before_allocation', slots=slots, branches=list(ARMS), cost=cost))
    env = model = counter = before = after = None
    handles, failures = [], []
    phase = {'value':'setup'}
    try:
        # Repeat model-free provenance validation before any environment allocation.
        frozen, ctx = response.prepare(read(Path(p['source'])/'protocol.json')['capability'], p['root_rgb'])
        if frozen['initial_context'] != p['initial_context'] or frozen['goal'] != p['goal'] or frozen['q'] != p['q']:
            raise ValueError('original input context/metric changed')
        refs = ctx[7]
        import torch
        base.require_cuda_allocation('cuda')
        device = torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32 = False
        model = base.load_compatible_pretrained(Path(p['policy']), device=device).eval().requires_grad_(False)
        before = checked.tensor_state_hash(model)
        def action_pre(module, args):
            cost['action_encoder_attempts'] += 1
            cost['action_encoder_input_tokens'] += int(args[0].shape[0]*args[0].shape[1])
        def action_post(module, args, result):
            cost['action_encoder_completed'] += 1
        def visual_pre(module, args):
            cost['visual_encoder_attempts'] += 1
            cost['visual_encoder_frames_attempted'] += int(args[0].shape[0])
        def visual_post(module, args, result):
            cost['visual_encoder_completed'] += 1
            cost['visual_encoder_frames_completed'] += int(args[0].shape[0])
        handles = [model.action_encoder.register_forward_pre_hook(action_pre),
                   model.action_encoder.register_forward_hook(action_post),
                   model.encoder.register_forward_pre_hook(visual_pre),
                   model.encoder.register_forward_hook(visual_post)]
        env = base.gym.make('swm/PushT-v1', render_mode='rgb_array')
        if tuple(env.unwrapped.variation_space['agent']['color'].value.tolist()) != (65,105,225):
            raise ValueError('original renderer required')
        original_step = env.step
        def counted_step(command):
            if phase['value']=='selected':
                return original_step(command)
            cost['setup_attempts'] += 1
            journal.emit(dict(kind='setup_step_attempt', command=np.asarray(command).tolist(), cost=deepcopy(cost)))
            result = original_step(command)
            cost['setup_returns'] += 1
            journal.emit(dict(kind='setup_step_returned', cost=deepcopy(cost)))
            return result
        env.step = counted_step
        roots = base.collect_nonterminal_boundary_contact_roots(env, count=4, seed=p['root_seed'], oversample_factor=4)
        if [base.root_fingerprint(r) for r in roots] != p['original_root_order'] or base.root_fingerprint(roots[1]) != ROOT:
            raise ValueError('original root order changed; no replacement')
        with PredictInstrumentation(model) as counter:
            backend = CudaBackend(model, device, base, counter, journal.emit, cost)
            for arm in ARMS:
                verify_prepared(output)
                def reset():
                    phase['value'] = 'setup'
                    frames, history, goal = base.physical._reset_and_warmup(env, roots[1], history_len=3)
                    if np.asarray(roots[1]['goal']).tolist() != p['physical_goal']:
                        raise ValueError('original fixed physical goal changed')
                    identity = verify_rgb_context(frames, goal, refs[ROOT])
                    between = np.repeat(np.asarray(history)[1:,None,:], 5, axis=1)
                    if not np.array_equal(between, p['between_raw']):
                        raise ValueError('original incoming action history changed')
                    journal.emit(dict(kind='reset_identity', arm=arm, identity=identity,
                        goal=np.asarray(roots[1]['goal']).tolist(), history=np.asarray(history).tolist()))
                    phase['value'] = 'selected'
                    return frames, between
                rows.append(arm_loop(p, arm, env=env, reset=reset, state_of=base.p0.packets.state_of,
                    backend=backend, emit=journal.emit, slots=slots, cost=cost))
        torch.cuda.synchronize(device)
        after = checked.tensor_state_hash(model)
        if before != after or any(m.training for m in model.modules()) or any(x.requires_grad for x in model.parameters()):
            raise ValueError('frozen eval model changed')
        verify_prepared(output)
    except (Exception, KeyboardInterrupt) as exc:
        failures.append(durable.failure(exc, 'runtime'))
        journal.emit(dict(kind='runtime_failure', failures=failures, cost=deepcopy(cost)))
    finally:
        for h in handles:
            h.remove()
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                failures.append(durable.failure(exc, 'close'))
        for arm in ARMS:
            if arm not in {r['arm'] for r in rows}:
                rows.append(dict(arm=arm, root_sha256=ROOT, status='blocked', completed_packets=0))
        census(slots)
        predicts = [] if counter is None else counter.records
        samples = sum(r['predict_samples'] for r in predicts)
        full = (not failures and all(r['status']=='completed' for r in rows) and
            all(s['status']=='completed' for s in slots) and samples==80 and cost['selected_returns']==40 and
            cost['selected_attempts']==40 and all(r['completed'] and r['failed_predict_calls']==0 and
                r['horizon_tokens']==1 and r['context_lengths']=={3:r['predict_calls']} for r in predicts) and
            cost['jacobian_completed']==6 and cost['bank_completed']==6 and cost['encode_completed']==6 and
            cost['jacobian_attempts']==6 and cost['bank_attempts']==6 and cost['encode_attempts']==6 and
            cost['encode_frames_attempted']==18 and cost['visual_encoder_frames_attempted']==18 and
            cost['visual_encoder_frames_completed']==18 and cost['visual_encoder_completed']==6 and
            cost['visual_encoder_attempts']==6 and cost['action_encoder_completed']==80 and
            cost['action_encoder_attempts']==80 and cost['action_encoder_input_tokens']==240 and
            before is not None and before==after)
        journal.emit(dict(kind='final_census', rows=rows, slots=slots, cost=cost, predict=predicts))
        journal.close()
        write(output/'report.json', dict(schema=TAG, status='calibration_recorded' if full else 'partial_or_integrity_fail',
            rows=rows, slots=slots, failures=failures, cost=cost, predict=predicts, actual_predict_samples=samples,
            model_unchanged=before is not None and before==after, protocol_sha256=prepared['protocol_sha256'],
            source_seals=prepared['source_seals'], journal_sha256=sha(output/'runtime.events.jsonl'),
            scalar_vjp_directions_completed=192*cost['jacobian_completed'], svd_calls_completed=2*cost['bank_completed'],
            partial_internal_work_unknown=not full, P1_admission=False, P4_admission=False, efficacy_verified=False))
    if not full:
        raise RuntimeError('runtime partial/integrity failure; all slots and partial work retained')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare','preflight','run'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command=='prepare':
        print(json.dumps(prepare(args.output)))
    elif args.command=='preflight':
        p, r = verify_prepared(args.output)
        print(json.dumps(dict(status='preflight_passed_not_run', census=census(p['slots']), receipt=r)))
    else:
        run(args.output)
