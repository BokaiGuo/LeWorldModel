"""Independent P3 joint-H2/H3 metric x scheduler experiment; MAIN CUDA only.

Prepare/verify/export are arrays and source checks only. No P1 conditional
compiler, physical evaluator, training, outcome input or CPU neural fallback.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import os
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import p3p4_goal_patch_replay_20261027_v1 as evidence
from experiments.spectral_reachability import pusht_full_packet_compiler_20261027 as kernel
from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal

old, metric = evidence.metric.old, evidence.metric
io, require, write, digest = old.io, old.require, old.write, old.digest
NAME = 'p3_joint_horizon_goal_metric_scheduler_20261027_v1'
PROJECT = Path(__file__).resolve().parents[2]
METRICS = ('old_Q', 'goal_patch_Q')
ARMS = tuple(q + '__' + s for q in METRICS for s in ('uniform', 'dynamic'))
META = dict(layout='states_with_between_state_actions_v1', context_length=3, latent_dim=192, action_dim=10)


def contract():
    return dict(roots=4, arms=list(ARMS), horizons=[2, 3],
        coordinates='joint time-major 10H logits; tanh -> Hx5x2 raw; no detach',
        objective='||Q@(native_terminal-goal0)||^2/192; no rank normalization',
        proposals='index0 zero; index1 own-Q bounded ridge; k>=2 old pool[k]',
        damping=.002, radius='sqrt(5H)*.75 joint logit ball',
        seed='20261027+100*sealed_root_seed_index+H', duplicate_policy='retain and pay',
        scheduler=old.contract()['dynamic'], uniform='5 H2 blocks +5 H3 blocks',
        shared_predict_cap=10, arm_predict_cap=66, search_predicts_per_arm=60,
        root_predict_cap=274, four_root_predict_cap=1096,
        action_encoder_cap='same independent caps as predict; full3 tokens per call',
        total_module_entry_cap=2192, fresh_jacobians_per_root=2,
        jacobian_shapes=[[192, 20], [192, 30]], mathematical_vjp_directions_per_root=384,
        jacobian_api_calls_per_root=2, backward_engine_api_calls=None,
        execution_svd_calls_per_root=8, visual_encoder_calls=0,
        winner='stable argmin(active_Q_cost,H,index)', verification='whole+segmented exact full context; 2H/arm',
        backend='torch SDPA MATH for both AD/search and replay',
        inherited_metric='sealed goal vectors/Q; 80 encoder+80 projector+8 SVD, not free',
        source_validation='16 saved-array SVD per validate_metric call; no neural',
        total_resource_matched=False, no_refunds=True, no_padding=True,
        P1_admission=False, P2_admission=False, P3_admission=False, P4_admission=False,
        new_physical_steps=0, scope='known four roots; NOT conditional fixedH2 P1')


def array(value, shape):
    a = np.asarray(value)
    require(a.dtype.kind in 'fi' and a.shape == shape and np.isfinite(a).all(), 'finite aligned array')
    return a.astype(np.float64)


def projection(value):
    q = array(value, (192, 192))
    require(np.allclose(q, q.T, atol=1e-8, rtol=0) and
            np.allclose(q @ q, q, atol=1e-8, rtol=0), 'symmetric idempotent Q')
    require(np.trace(q) > .5, 'nonzero metric rank; no fallback')
    return q


def array_id(value):
    a = np.ascontiguousarray(np.asarray(value, dtype='<f8'))
    require(np.isfinite(a).all(), 'finite array binding')
    return dict(shape=list(a.shape), dtype='<f8', bytes_sha256=io.sha(a.tobytes()))


def active_id(packet, label, q, provenance):
    require(label in METRICS, 'known metric')
    return digest(dict(root_sha256=packet['root_sha256'], source_input_sha256=packet['input_sha256'],
        label=label, projection=array_id(q), goal=array_id(packet['goal']), provenance=provenance))


def context_arrays(context):
    require(all(context.get(k) == v and type(context.get(k)) is type(v) for k, v in META.items()),
        'context layout/dimension metadata')
    return array(context['latent_history'], (3, 192)), array(context['action_history'], (2, 10))


def normalizer_binding(root, sources):
    """AST literal read only; never imports a neural runtime or H2 compiler."""
    path = Path(root)/'experiments/run_pusht_counterfactual.py'
    raw = path.read_bytes()
    require(io.sha(raw) == sources[str(path)], 'sealed normalizer source bytes')
    found = {}
    for node in ast.parse(raw).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            key = node.targets[0].id
            if key in ('PUBLIC_ACTION_MEAN', 'PUBLIC_ACTION_STD'):
                require(key not in found and isinstance(node.value, ast.Call) and
                    isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'asarray' and
                    isinstance(node.value.func.value, ast.Name) and node.value.func.value.id == 'np' and
                    len(node.value.args) == 1, 'unique explicit NumPy literal statistics')
                found[key] = np.asarray(ast.literal_eval(node.value.args[0]), dtype=np.float32)
    require(set(found) == {'PUBLIC_ACTION_MEAN', 'PUBLIC_ACTION_STD'}, 'both normalization constants')
    mean, std = (found[k] for k in ('PUBLIC_ACTION_MEAN', 'PUBLIC_ACTION_STD'))
    require(mean.shape == std.shape == (2,) and np.isfinite(mean).all() and
        np.isfinite(std).all() and np.all(std > 0), 'finite positive per-command statistics')
    return dict(source=str(path), source_sha256=io.sha(raw), mean=mean.astype(float).tolist(),
        std=std.astype(float).tolist(), training_provenance_verified=False)


def check_history(before, after, actions, normalization):
    """Pure float32 packet arithmetic for 1/2/3 rolling transitions, no model."""
    latent, history = context_arrays(before)
    _, actual = context_arrays(after)
    require({k: v for k, v in before.items() if k not in ('latent_history', 'action_history')} ==
        {k: v for k, v in after.items() if k not in ('latent_history', 'action_history')}, 'unchanged context metadata')
    raw = np.asarray(actions)
    require(raw.dtype.kind in 'fi' and raw.shape in ((5, 2), (10, 2), (15, 2)) and
        np.isfinite(raw).all() and np.all(np.abs(raw) <= 1), 'bounded H1/H2/H3 raw controls')
    mean, std = (np.asarray(normalization[k], dtype=np.float32) for k in ('mean', 'std'))
    require(mean.shape == std.shape == (2,) and np.isfinite(mean).all() and
        np.isfinite(std).all() and np.all(std > 0), 'finite float32 normalizer')
    # Separate float32 subtract/divide: no float64 arithmetic, /100, FMA or interleave.
    tokens = ((raw.astype(np.float32)-mean)/std).reshape(-1, 10)
    expected = history.astype(np.float32)
    for token in tokens:
        expected = np.concatenate((expected[1:], token[None]), axis=0)
    require(np.array_equal(actual, expected.astype(np.float64)), 'normalized action history chronology/content')
    h = len(tokens)
    if h < 3:
        require(np.array_equal(np.asarray(after['latent_history'])[:3-h],
            latent.astype(np.float32)[h:].astype(np.float64)), 'retained latent chronology')
    return array_id(expected)


def validate_packet(packet):
    require(packet['input_sha256'] == digest({k: packet[k] for k in
            ('root_sha256', 'root_context', 'goal', 'projection')}), 'source packet hash')
    context_arrays(packet['root_context'])
    array(packet['goal'], (192,))
    projection(packet['projection'])


def source_material(root):
    """Full source metric validation, not just favourable selected winners."""
    plans, p, bindings, inherited = evidence.validate_metric(Path(root))
    by_root = io.index(plans, ('root_sha256', 'arm'))
    roots = [x['root_sha256'] for x in p['packets']]
    require(len(roots) == len(set(roots)) == 4, 'exact four roots')
    require(set(by_root) == {(r, q) for r in roots for q in METRICS}, 'exact source metric census')
    records = []
    for seed_index, packet in enumerate(p['packets']):
        validate_packet(packet)
        new = by_root[packet['root_sha256'], 'goal_patch_Q']
        provenance = dict(metric_report_sha256=new['metric_report_sha256'],
            metric_sha256=new['metric_sha256'], model_bindings=p['model_bindings'])
        qs = dict(old_Q=packet['projection'], goal_patch_Q=new['projection'])
        records.append(dict(packet=deepcopy(packet), seed_index=seed_index,
            metrics={label: dict(projection=projection(q).tolist(),
                active_metric_id=active_id(packet, label, q, provenance)) for label, q in qs.items()},
            provenance=provenance))
    return records, bindings, inherited, p['model_bindings']


def prepare(root, output):
    root, output = Path(root).resolve(), Path(output).absolute()
    require(not output.exists(), 'exclusive NEW prepare path')
    records, bindings, inherited, models = source_material(root)
    sources = old.prior.physical.source_closure(root, [Path(__file__).resolve(),
        root/'tests'/('test_'+NAME+'.py'), root/'innerdocs'/(NAME+'.md')])
    source_hashes = {str(s): io.sha(s.read_bytes()) for s in sources}
    body = dict(schema=NAME+'.protocol', contract=contract(), project_root=str(root),
        records=records, bindings=bindings, inherited_resources=inherited, model_bindings=models,
        arm_order=[list(ARMS[i:]+ARMS[:i]) for i in range(4)],
        sources_sha256=source_hashes, normalization=normalizer_binding(root, source_hashes))
    output.mkdir(exist_ok=False)
    write(output/'protocol.json', body)
    seals = [create_seal(files=sources[i:i+40], args_json=io.canonical(dict(protocol_id=digest(body))),
        output_dir=output/f'source_seal_{i//40}') for i in range(0, len(sources), 40)]
    result = dict(status='prepared_not_executed', protocol_id=digest(body),
        protocol_sha256=io.sha((output/'protocol.json').read_bytes()), source_seals=seals,
        new_neural_calls=0, new_physical_steps=0, P1_admission=False, P3_admission=False)
    write(output/'prepare.json', result)
    return result


def validate_prepared(prepared, expected_sha):
    prepared = Path(prepared).absolute()
    require(io.sha((prepared/'prepare.json').read_bytes()) == expected_sha, 'prepare SHA')
    receipt = io.decode((prepared/'prepare.json').read_bytes())
    require(io.sha((prepared/'protocol.json').read_bytes()) == receipt['protocol_sha256'], 'protocol SHA')
    p = io.decode((prepared/'protocol.json').read_bytes())
    require(p['schema'] == NAME+'.protocol' and p['contract'] == contract() and
            digest(p) == receipt['protocol_id'], 'frozen contract and identity')
    for path, h in {**p['sources_sha256'], **p['bindings']}.items():
        require(io.sha(Path(path).read_bytes()) == h, 'source/input drift: '+path)
    for seal in receipt['source_seals']:
        verify_seal(output_dir=seal['output_dir'], expected_manifest_sha256=seal['manifest_sha256'])
    records, bindings, inherited, models = source_material(p['project_root'])
    require(records == p['records'] and bindings == p['bindings'] and inherited == p['inherited_resources']
            and models == p['model_bindings'], 'source reconstruction')
    require(p['arm_order'] == [list(ARMS[i:]+ARMS[:i]) for i in range(4)], 'sealed balanced arm order')
    require(p['normalization'] == normalizer_binding(p['project_root'], p['sources_sha256']), 'normalizer binding')
    return p


def response(packet, q, j, nominal, h, target, journal):
    """Two separately journaled SVD calls; partial signal survives solve failure."""
    q, j, nominal = projection(q), array(j, (192, 10*h)), array(nominal, (192,))
    goal = array(packet['goal'], (192,))
    target.update(response=(q @ j).tolist(), residual=(q @ (goal-nominal)).tolist())
    journal(dict(kind='response_arrays', horizon=h, arrays=deepcopy(target)))
    for role in ('signal', 'ridge'):
        event = dict(kind='svd', role=role, horizon=h, completed=False)
        journal(deepcopy(event))
        if role == 'signal':
            target['signal'] = old.residual_signal(j, nominal, goal, q, h)
        else:
            target['solve'] = kernel.bounded_ridge(q @ j, q @ (goal-nominal),
                damping=.002, cap=float(np.sqrt(5*h)*.75))
            target['proposal_logits'] = target['solve']['params']
            target['raw_proposal'] = np.tanh(target['proposal_logits']).reshape(5*h, 2).tolist()
        event['completed'] = True
        journal(dict(kind='response_arrays', horizon=h, arrays=deepcopy(target)))
        journal(deepcopy(event))
    return target


def candidate_bank(seed_index, h, count, compiled):
    require(h in (2, 3) and type(count) is int and count >= 2, 'two reserved slots')
    logits = array(compiled['proposal_logits'], (10*h,))
    require(np.linalg.norm(logits) <= np.sqrt(5*h)*.75+1e-12, 'joint ridge logit radius')
    bank = old.pool(seed_index, h, count)
    bank[1] = np.tanh(logits).reshape(5*h, 2)
    return bank


class NeuralCap(old.NeuralCap):
    """Same durable no-refund instrumentation, independent four-arm caps."""
    def __init__(self, model, journal):
        super().__init__(model, journal)
        self.cleanup_errors = []
        self.journal_errors = []

    def __exit__(self, exc_type, exc, traceback):
        for obj, key, callback in ((self.model, 'predict', self.original[0]),
                (self.model.action_encoder, 'forward', self.original[1]),
                (self.model.encoder, 'forward', self.original[2])):
            try:
                setattr(obj, key, callback)
            except Exception as error:
                self.cleanup_errors.append(dict(attribute=key, type=type(error).__name__, message=str(error)))
        if self.cleanup_errors and exc_type is None:
            raise RuntimeError('hook restoration failed; details retained in receipt')
        return False

    def receipt(self):
        return dict(super().receipt(), cleanup_errors=deepcopy(self.cleanup_errors), journal_errors=deepcopy(self.journal_errors))

    def dispatch(self, kind, callback, args, kwargs):
        require(self.role in ('shared', *ARMS), 'explicit neural role required')
        x = args[0]
        require(getattr(getattr(x, 'device', None), 'type', None) == 'cuda' and
            tuple(x.shape) == ((1, 3, 192) if kind == 'predict' else (1, 3, 10)), 'CUDA full context')
        counts = lambda role: sum(e['kind'] == kind and (role is None or e['role'] == role) for e in self.events)
        if counts(None) >= 274 or counts(self.role) >= (10 if self.role == 'shared' else 66):
            self.blocked += 1
            self.journal(dict(kind='blocked', entry=kind, role=self.role))
            raise ValueError('nonrefundable cap before dispatch')
        event = dict(kind=kind, role=self.role, index=len(self.events), context_tokens=3, completed=False)
        self.events.append(event)
        self.journal(deepcopy(event))
        failure = None
        try:
            result = callback(*args, **kwargs)
            event['completed'] = True
            return result
        except BaseException as exc:
            failure = exc
            event['error'] = type(exc).__name__+': '+str(exc)
            raise
        finally:
            try:
                self.journal(deepcopy(event))
            except (Exception, KeyboardInterrupt) as exc:
                self.journal_errors.append(dict(type=type(exc).__name__, message=str(exc)))
                if failure is None:
                    raise


def search(record, arm, compiled, replay, target, journal):
    label, schedule = arm.split('__')
    require(arm in ARMS, 'known arm')
    p, q = record['packet'], record['metrics'][label]['projection']
    own = compiled['metrics'][label]
    allocation = old.allocate(schedule, {h: own[h]['signal'] for h in ('2', '3')})
    target.update(arm=arm, active_metric_id=record['metrics'][label]['active_metric_id'],
        source_input_sha256=p['input_sha256'], allocation=allocation, candidates=[], status='incomplete')
    for h in (2, 3):
        bank = candidate_bank(record['seed_index'], h, allocation['candidates'][str(h)], own[str(h)])
        for k, actions in enumerate(bank):
            c = dict(horizon=h, index=k, actions=actions.tolist(),
                logits=(own[str(h)]['proposal_logits'] if k == 1 else np.arctanh(actions).reshape(-1).tolist()),
                completed=False)
            target['candidates'].append(c)
            journal(dict(kind='candidate', **deepcopy(c)))
            c['context'] = replay(deepcopy(p['root_context']), actions)
            journal(dict(kind='candidate_context_arrays', horizon=h, index=k, context=deepcopy(c['context'])))
            terminal = array(c['context']['latent_history'], (3, 192))[-1]
            c['cost'] = metric.native_cost(terminal, p['goal'], q)
            require(np.isfinite(c['cost']), 'finite candidate cost')
            c['completed'] = True
            journal(dict(kind='candidate', **deepcopy(c)))
    winner = min(target['candidates'], key=lambda c: (c['cost'], c['horizon'], c['index']))
    target['winner'] = deepcopy(winner)
    target['whole_replay_context'] = replay(deepcopy(p['root_context']), winner['actions'])
    journal(dict(kind='whole_replay_context', arm=arm, context=deepcopy(target['whole_replay_context'])))
    target['segments'] = []
    context = deepcopy(p['root_context'])
    for packet in np.asarray(winner['actions']).reshape(-1, 5, 2):
        context = replay(context, packet)
        target['segments'].append(deepcopy(context))
        journal(dict(kind='segmented_replay_context', arm=arm, segment=len(target['segments']), context=deepcopy(context)))
    target['segmented_replay_context'] = context
    require(target['whole_replay_context'] == context == winner['context'], 'exact complete-context replay')
    target.update(verification_predicts=2*winner['horizon'], whole_and_segmented_exact=True, status='completed')
    return target


def compile_shared(model, device, record, target, journal):
    import torch
    p = record['packet']
    transition = kernel.make_transition(model, device)
    emb = torch.tensor(p['root_context']['latent_history'], dtype=torch.float32, device=device)[None]
    hist = torch.tensor(p['root_context']['action_history'], dtype=torch.float32, device=device)[None]
    target.update(raw={}, metrics={q: {} for q in METRICS})
    for h in (2, 3):
        c = target['raw'][str(h)] = dict(nominal_segments=[], jacobian_forward_segments=[])
        def terminal(u, phase='jacobian_forward'):
            state, between = emb, hist
            for raw in torch.tanh(u).reshape(h, 5, 2):
                state, between = transition(state, between, raw)
                context = dict(p['root_context'], latent_history=state.detach()[0].cpu().tolist(),
                    action_history=between.detach()[0].cpu().tolist())
                c[phase+'_segments'].append(context)
                journal(dict(kind='compile_boundary_arrays', phase=phase, horizon=h,
                    segment=len(c[phase+'_segments']), context=deepcopy(context)))
            return state[0, -1]
        zero = emb.new_zeros(10*h)
        with torch.no_grad():
            c['nominal'] = terminal(zero, phase='nominal').cpu().tolist()
        journal(dict(kind='nominal_array', horizon=h, nominal=c['nominal']))
        event = dict(kind='jacobian', horizon=h, vjp_directions=192, completed=False)
        journal(deepcopy(event))
        jac = torch.autograd.functional.jacobian(terminal, zero.requires_grad_(True), vectorize=True)
        c['jacobian'] = jac.detach().cpu().tolist()
        journal(dict(kind='jacobian_array', horizon=h, jacobian=c['jacobian']))
        array(c['jacobian'], (192, 10*h))
        event['completed'] = True
        journal(deepcopy(event))
        for label in METRICS:
            own = target['metrics'][label][str(h)] = {}
            response(p, record['metrics'][label]['projection'], c['jacobian'], c['nominal'], h, own,
                lambda e, label=label: journal(dict(metric=label, **e)))


def accounting_ok(receipt, arms):
    if receipt['blocked'] or receipt.get('cleanup_errors') or receipt.get('journal_errors') or set(arms) != set(ARMS) or any(a.get('status') != 'completed' for a in arms.values()):
        return False
    events = receipt['events']
    expected = dict(shared=10, **{a: 60+arms[a]['verification_predicts'] for a in ARMS})
    return (all(e['completed'] and e['context_tokens'] == 3 and e['role'] in expected for e in events)
        and all(sum(e['kind'] == k and e['role'] == r for e in events) == n
            for k in ('predict', 'action_encoder') for r, n in expected.items()))


def execute_cuda(prepared, expected_sha, output):
    """Explicit main-only entry. No execution from prepare, import or tests."""
    p = validate_prepared(prepared, expected_sha)
    output = Path(output).absolute()
    require(not output.exists(), 'exclusive NEW execution output')
    from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
    from experiments.spectral_reachability.pusht_composition_cuda_replay_v2 import CudaReplay
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    output.mkdir(exist_ok=False)
    write(output/'execution_protocol.json', dict(prepared=str(Path(prepared).absolute()), prepare_sha256=expected_sha))
    result = dict(schema=NAME+'.report', protocol_id=digest(p), contract=contract(), roots=[],
        status='incomplete', errors=[], model_unchanged=False, P1_admission=False, P2_admission=False,
        P3_admission=False, P4_admission=False, new_physical_steps=0)
    # Initial validation above completed one 16-SVD source reconstruction.
    result['source_validation_calls'] = dict(attempted=1, completed=1)
    model = None
    before = None
    execution_recorded = False
    primary_execution = None
    def record_error(stage, exc, **extra):
        result['errors'].append(dict(stage=stage, type=type(exc).__name__, message=str(exc), **extra))
    try:
        for path, h in p['model_bindings'].items():
            require(checked.digest(path) == h, 'model bytes drift')
        result['host'] = require_cuda_allocation('cuda')
        device = torch.device('cuda:0')
        policy = Path(next(x for x in p['model_bindings'] if Path(x).name == 'weights.pt')).parent
        model = load_compatible_pretrained(str(policy), device=device).eval().requires_grad_(False)
        before = checked.tensor_state_hash(model)
        replay = CudaReplay(model, device)
        with sdpa_kernel(SDPBackend.MATH):
            for index, record in enumerate(p['records']):
                row = dict(root_sha256=record['packet']['root_sha256'], compilation={}, arms={}, status='incomplete')
                result['roots'].append(row)
                path = output/f'root_{index}.events.jsonl'
                with path.open('x') as stream:
                    def journal(event):
                        stream.write(io.canonical(event)+'\n')
                        stream.flush()
                        os.fsync(stream.fileno())
                    cap = NeuralCap(model, journal)
                    try:
                        journal(dict(kind='root_context_arrays', root_sha256=row['root_sha256'],
                            context=deepcopy(record['packet']['root_context'])))
                        with cap:
                            with cap.scope('shared'):
                                compile_shared(model, device, record, row['compilation'], journal)
                            for arm in p['arm_order'][index]:
                                target = row['arms'][arm] = {}
                                with cap.scope(arm):
                                    search(record, arm, row['compilation'], replay, target, journal)
                        torch.cuda.synchronize(device)
                        require(accounting_ok(cap.receipt(), row['arms']), 'exact actual accounting')
                        row['status'] = 'completed'
                    except (Exception, KeyboardInterrupt) as exc:
                        record_error('execution', exc, root_sha256=row['root_sha256'])
                        execution_recorded = True
                        primary_execution = exc
                        raise
                    finally:
                        row['neural_receipt'] = cap.receipt()
                        if cap.cleanup_errors:
                            result['errors'].append(dict(stage='hook_cleanup', root_sha256=row['root_sha256'],
                                details=deepcopy(cap.cleanup_errors)))
                        if cap.journal_errors:
                            result['errors'].append(dict(stage='journal_cleanup', root_sha256=row['root_sha256'],
                                details=deepcopy(cap.journal_errors)))
                        row['journal_path'] = str(path)
                        # Each persistence operation is independent. Neither can
                        # replace an exception already propagating from execution.
                        try:
                            row['journal_sha256'] = checked.digest(path)
                        except (Exception, KeyboardInterrupt) as exc:
                            record_error('root_journal_hash', exc, root_sha256=row['root_sha256'])
                        try:
                            write(output/f'root_{index}.json', row)
                        except (Exception, KeyboardInterrupt) as exc:
                            record_error('root_write', exc, root_sha256=row['root_sha256'])
                require(not result['errors'], 'root finalization failed; subsequent roots blocked')
    except (Exception, KeyboardInterrupt) as exc:
        if not execution_recorded:
            record_error('execution', exc)
        elif exc is not primary_execution:
            record_error('execution_unwind', exc)
    # Final checks are independent and never overwrite the primary failure.
    for stage, callback in (
        ('model_after', lambda: checked.tensor_state_hash(model) if model is not None else None),
        ('source_after', lambda: validate_prepared(prepared, expected_sha))):
        try:
            if stage == 'source_after':
                result['source_validation_calls']['attempted'] += 1
            value = callback()
            if stage == 'source_after':
                result['source_validation_calls']['completed'] += 1
            if stage == 'model_after':
                result['model_unchanged'] = before is not None and value == before
                require(result['model_unchanged'], 'model not established unchanged')
        except (Exception, KeyboardInterrupt) as exc:
            result['errors'].append(dict(stage=stage, type=type(exc).__name__, message=str(exc)))
    seen = {r['root_sha256'] for r in result['roots']}
    for record in p['records']:
        if record['packet']['root_sha256'] not in seen:
            result['roots'].append(dict(root_sha256=record['packet']['root_sha256'], status='blocked', arms={}))
    for r in result['roots']:
        for arm in ARMS:
            r['arms'].setdefault(arm, dict(status='blocked'))
    result['status'] = 'cuda_completed' if not result['errors'] and all(r['status'] == 'completed' for r in result['roots']) else 'incomplete'
    result['resources'] = dict(
        attempted={k: sum(r.get('neural_receipt', {}).get('attempted', {}).get(k, 0) for r in result['roots'])
            for k in ('predict', 'action_encoder')},
        incomplete_counts_are_lower_bound=result['status'] != 'cuda_completed',
        inherited=p['inherited_resources'], backward_engine_api_calls=None,
        completed_source_validation_svds=16*result['source_validation_calls']['completed'],
        validation_saved_array_svds_per_source_material_call=16,
        operation_authority='per-root fsync journal including attempted AD/SVD; no failed operation charged zero')
    write(output/'report.json', result)
    require(result['status'] == 'cuda_completed', 'incomplete; preserve partials, no automatic retry')
    return result


def selected_actions(output):
    """Model-free rebuild of metric responses, banks, costs, census and witnesses."""
    output = Path(output).absolute()
    binding = io.decode((output/'execution_protocol.json').read_bytes())
    p = validate_prepared(binding['prepared'], binding['prepare_sha256'])
    raw = (output/'report.json').read_bytes()
    r = io.decode(raw)
    require(r['schema'] == NAME+'.report' and r['protocol_id'] == digest(p) and r['contract'] == contract(), 'report binding')
    require(r['status'] == 'cuda_completed' and r['errors'] == [] and r['model_unchanged'] is True and
        r['new_physical_steps'] == 0 and
        all(r[k] is False for k in ('P1_admission', 'P2_admission', 'P3_admission', 'P4_admission')), 'complete unchanged model only')
    roots = io.index(r['roots'], ('root_sha256',))
    require(set(roots) == {(x['packet']['root_sha256'],) for x in p['records']}, 'exact root census')
    plans = []
    for record in p['records']:
        row = roots[record['packet']['root_sha256'],]
        require(row['status'] == 'completed' and accounting_ok(row['neural_receipt'], row['arms']), 'actual per-arm counts')
        require(io.sha(Path(row['journal_path']).read_bytes()) == row['journal_sha256'], 'journal binding')
        events = [io.decode(line.encode()) for line in Path(row['journal_path']).read_text().splitlines()]
        entries = [e for e in events if e['kind'] in ('predict', 'action_encoder')]
        require(len(entries) == 2*len(row['neural_receipt']['events']) and
            [e for e in entries if e['completed']] == row['neural_receipt']['events'], 'complete entry journal')
        require(sum(e['kind'] == 'svd' and e['completed'] for e in events) == 8 and
            sum(e['kind'] == 'jacobian' and e['completed'] for e in events) == 2, 'AD/SVD completion census')
        require(set(row['compilation']['raw']) == {'2', '3'} and set(row['compilation']['metrics']) == set(METRICS), 'raw/Q census')
        for h in (2, 3):
            c = row['compilation']['raw'][str(h)]
            for phase in ('nominal', 'jacobian_forward'):
                require(len(c[phase+'_segments']) == h, 'complete nominal/AD-forward prefix census')
                previous = record['packet']['root_context']
                for context in c[phase+'_segments']:
                    check_history(previous, context, np.zeros((5, 2)), p['normalization'])
                    previous = context
            require(c['nominal'] == c['nominal_segments'][-1]['latent_history'][-1], 'nominal terminal array binding')
        rebuilt = dict(raw=row['compilation']['raw'], metrics={label: {} for label in METRICS})
        for label in METRICS:
            for h in (2, 3):
                c = row['compilation']['raw'][str(h)]
                rebuilt['metrics'][label][str(h)] = response(record['packet'], record['metrics'][label]['projection'],
                    c['jacobian'], c['nominal'], h, {}, lambda e: None)
        require(rebuilt == row['compilation'], 'own-Q response/solve reconstruction')
        for arm in ARMS:
            source = row['arms'][arm]
            contexts = iter([c['context'] for c in source['candidates']] +
                [source['whole_replay_context']] + source['segments'])
            def checked_context_replay(before, actions):
                after = deepcopy(next(contexts))
                check_history(before, after, actions, p['normalization'])
                return after
            rebuilt_search = search(record, arm, rebuilt, checked_context_replay, {}, lambda e: None)
            require(rebuilt_search == source, 'complete bank, costs, selection and double replay reconstruction')
            require(next(contexts, None) is None, 'no extra replay contexts')
            label = arm.split('__')[0]
            plan = dict(root_sha256=row['root_sha256'], seed_index=record['seed_index'], arm=arm,
                active_metric_id=source['active_metric_id'], source_input_sha256=source['source_input_sha256'],
                root_context=record['packet']['root_context'], goal=record['packet']['goal'],
                projection=record['metrics'][label]['projection'], winner=source['winner'],
                report_sha256=io.sha(raw), protocol_id=digest(p),
                P1_admission=False, P3_admission=False, physical_ready=False,
                export_response_svd_calls=32, new_neural_calls=0)
            plan['selection_sha256'] = digest(plan)
            plans.append(plan)
    return plans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare', 'verify', 'execute-cuda', 'export'):
        action.add_argument('--'+name, action='store_true')
    parser.add_argument('--project-root', type=Path, default=PROJECT)
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--prepare-sha256')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.prepare:
        require(args.output is not None, 'prepare output required')
        result = prepare(args.project_root, args.output)
    elif args.export:
        require(args.output is not None, 'existing execution output required')
        result = dict(plans=selected_actions(args.output))
    else:
        require(args.prepared is not None and args.prepare_sha256 is not None, 'reviewed prepare + SHA required')
        if args.verify:
            validate_prepared(args.prepared, args.prepare_sha256)
            result = dict(status='verified_no_neural')
        else:
            require(args.output is not None, 'new execution output required')
            result = execute_cuda(args.prepared, args.prepare_sha256, args.output)
    print(io.canonical({k: v for k, v in result.items() if k not in ('roots', 'source_seals')}))


if __name__ == '__main__':
    main()
