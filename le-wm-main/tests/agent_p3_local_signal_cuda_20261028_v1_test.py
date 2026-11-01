"""Synthetic RAM/saved-buffer regression only; no model, CUDA or physics."""
from copy import deepcopy
import builtins
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p3_local_signal_cuda_20261028_v1 as m


def task(slot):
    c = dict(layout='states_with_between_state_actions_v1', context_length=3, latent_dim=192,
        action_dim=10, latent_history=np.zeros((3, 192), np.float32).tolist(),
        action_history=np.full((2, 10), np.float32(slot/8)).tolist())
    t = dict(root_context=c, goal=np.ones(192).tolist(), projection=np.eye(192).tolist(),
        action_mean=np.array([.125, -.25], np.float32).tolist(), action_std=[2., 4.])
    t['binding'] = {k: m.digest([slot, k]) for k in m.core.BINDINGS}
    t['binding'].update(goal_sha256=m.digest(t['goal']), projection_sha256=m.digest(t['projection']),
        normalizer_sha256=m.digest([t['action_mean'], t['action_std']]), context_sha256=m.digest(c),
        action_domain_sha256=m.digest(m.core.action_domain()))
    return m.core.validate_task(t)


class Sink:
    def __init__(self): self.events = []
    def event(self, v): self.events.append(deepcopy(v))


class RamSignal:
    """Paid array arithmetic fixture, NOT a model or a CUDA emulation claim."""
    def __init__(self, t, sink, fault=False): self.task, self.sink, self.fault = t, sink, fault

    def signal(self, req, ledger):
        t = self.task; h = req['horizon']; phase = step = None
        def emit(kind, operation_index=None, **arrays):
            self.sink.event(dict(kind=kind, scope=deepcopy(ledger.scope), phase=phase, step=step,
                operation_index=operation_index, arrays={k: m.blob(np.asarray(v, np.float32)) for k, v in arrays.items()}))
        def call(resource, fn, inputs, shape=None, directions=None):
            index = len(ledger.events)
            def actual():
                emit('io_input', operation_index=index, **{resource+':'+k: v for k, v in inputs.items()})
                out = fn(); emit('io_return', operation_index=index, **{resource+':output': out}); return out
            return ledger.call(resource, actual, shape=shape, directions=directions)
        last = np.zeros(192, np.float32)
        for phase in ('nominal', 'AD'):
            latent = np.asarray(t['root_context']['latent_history'], np.float32)[None]
            hist = np.asarray(t['root_context']['action_history'], np.float32)[None]
            mean, std = (np.asarray(t[k], np.float32) for k in ('action_mean', 'action_std'))
            emit('origin', latent=latent, history=hist, raw_flat=np.zeros(10*h, np.float32), mean=mean, std=std)
            # Nest AD forward inside the Jacobian charge, as the real adapter does.
            def forward():
                nonlocal latent, hist, step
                for step in range(h):
                    raw = np.zeros((5, 2), np.float32); token = ((raw-mean)/std).reshape(10)
                    outgoing = np.concatenate((hist, token[None, None]), axis=1)
                    emit('packet', raw=raw, token=token, outgoing=outgoing)
                    ae = call('action_encoder', lambda: np.zeros((1, 3, 192), np.float32), {'0': outgoing}, [1, 3, 10])
                    def predict():
                        pred = call('predictor', lambda: np.zeros((1, 3, 192), np.float32), {'0': latent, '1': ae}, [1, 3, 192])
                        proj = call('pred_proj', lambda: pred.reshape(3, 192), {'0': pred.reshape(3, 192)}, [3, 192])
                        return proj.reshape(1, 3, 192)
                    predicted = call('predict', predict, {'latent': latent, 'act_emb': ae}, [1, 3, 192])
                    latent = np.concatenate((latent[:, 1:], predicted[:, -1:]), axis=1); hist = outgoing[:, 1:]
                    emit('boundary', latent=latent, history=hist)
                return latent[0, -1]
            if phase == 'nominal':
                last = forward(); emit('nominal', value=last)
            else:
                def ad():
                    nonlocal step
                    forward(); step = None; emit('AD_terminal', value=last)
                    if self.fault: raise RuntimeError('paid AD forward RAM fault')
                    rows = []
                    for direction in range(192):
                        self.sink.event(dict(kind='direction', scope=deepcopy(ledger.scope), direction=direction))
                        raw_grad = np.zeros(10*h, np.float32)
                        def grad():
                            def derivative():
                                def raw_hook():
                                    emit('raw_backward', operation_index=ledger.events[-1]['index'], value=raw_grad)
                                    return raw_grad
                                return ledger.call('raw_backward', raw_hook)
                            return call('grad', derivative, {'output': last, 'raw': np.zeros(10*h, np.float32),
                                'basis': np.eye(192, dtype=np.float32)[direction]})
                        rows.append(ledger.call('vjp', grad, directions=1))
                    return np.stack(rows)
                jac = ledger.call('jacobian', ad); emit('jacobian', value=jac)
        return dict(binding_sha256=req['binding_sha256'], parent_context_sha256=req['parent_context_sha256'],
            horizon=h, nominal=last.tolist(), jacobian=jac.tolist(), producer_sha256=m.sha(m.__file__))


@pytest.fixture(scope='module')
def saved():
    p = dict(tasks=[task(0), task(1)], profile=m.profile(), expected_model_hash=m.digest('RAM'), sources={m.__file__: m.sha(m.__file__)})
    p.update(runtime_rule=m.runtime_rule(p['tasks']), contract_revision=m.REVISION)
    sink = Sink(); ledger = m.core.Ledger(p['profile']['resource_caps'], sink=sink.event)
    rows = m.census(); sink.event(dict(kind='denominator', rows=deepcopy(rows)))
    decisions = m.decision_census(); sink.event(dict(kind='decision_denominator', rows=deepcopy(decisions)))
    source = m.digest(p['sources']); sink.event(dict(kind='source_before', manifest_sha256=source))
    model = dict(original_state_dict=p['expected_model_hash'], all_buffers=m.digest('RAM buffers'))
    sink.event(dict(kind='model_before', hashes=model))
    diagnostics = m.execute_panel(p, lambda s: RamSignal(p['tasks'][s], sink), ledger, sink, rows, decisions)
    sink.event(dict(kind='model_after', hashes=model)); sink.event(dict(kind='source_after', manifest_sha256=source))
    r = dict(rows=rows, diagnostics=diagnostics, model_before=model, model_after=model,
        decision_rows=decisions, runtime_rule=p['runtime_rule'], contract_revision=m.REVISION,
        source_manifest_before=source, source_manifest_after=source,
        resource_receipt=ledger.receipt(), wall_seconds=1., CUDA_peak_bytes=[0, 0],
        host=dict(device='cuda:0', index=0, name='RAM_FIXTURE_NOT_EVIDENCE', allocation_sum=4.,
            torch_version='RAM_ONLY', torch_cuda='RAM_ONLY', visible_nvidia_nodes=['RAM_ONLY']))
    return p, r, sink.events


def test_full_saved_native_buffer_contract_and_exact_paid_denominator(saved):
    p, r, events = saved
    m.validate_saved(p, r, events); m.core.audit_ledger(r['resource_receipt'])
    assert r['resource_receipt']['attempted'] == r['resource_receipt']['completed'] == m.profile()['resource_caps']
    assert len(r['rows']) == 4 and r['diagnostics'][0]['signed_gain'] == 0


@pytest.mark.parametrize('fault', ['denominator', 'outcome', 'request', 'model', 'attempt', 'direction', 'precision', 'raw', 'normalizer', 'projection', 'source', 'chi', 'nominal', 'J', 'wall', 'memory', 'operation_link', 'module_shape', 'basis', 'raw_gradient'])
def test_saved_mutations_fail_closed(saved, fault):
    p, original, original_events = saved
    r = dict(original); events = list(original_events); p = dict(p)
    def mutate(kind):
        i = next(i for i, e in enumerate(events) if e['kind'] == kind)
        events[i] = deepcopy(events[i]); return events[i]
    if fault == 'denominator': r['rows'] = r['rows'][:-1]
    elif fault == 'outcome': events.append(dict(kind='native_success', value=True))
    elif fault == 'request': mutate('signal_request')['row']['slot'] = 1
    elif fault == 'model': r['model_before'] = dict(original_state_dict=m.digest('wrong'), all_buffers='wrong')
    elif fault == 'attempt': events.pop(next(i for i, e in enumerate(events) if e['kind'] == 'resource_attempt'))
    elif fault == 'direction': mutate('direction')['direction'] = 191
    elif fault == 'precision':
        e = mutate('origin'); e['arrays']['latent'] = m.blob(m.unblob(e['arrays']['latent']).astype(np.float64))
    elif fault == 'raw':
        e = mutate('packet'); e['arrays']['raw'] = m.blob(np.ones((5, 2), np.float32))
    elif fault == 'normalizer':
        e = mutate('origin'); e['arrays']['std'] = m.blob(np.ones(2, np.float32))
    elif fault == 'projection':
        p['tasks'] = deepcopy(p['tasks']); p['tasks'][0]['projection'][0][0] = 2
    elif fault == 'source':
        r['rows'] = deepcopy(r['rows']); r['rows'][0]['response']['producer_sha256'] = m.digest('future fake SHA')
    elif fault == 'chi':
        r['diagnostics'] = deepcopy(r['diagnostics']); r['diagnostics'][0]['signed_gain'] = .75
    elif fault == 'nominal':
        e = mutate('nominal'); e['arrays']['value'] = m.blob(np.ones(192, np.float32))
    elif fault == 'J':
        e = mutate('jacobian'); e['arrays']['value'] = m.blob(np.ones((192, 10), np.float32))
    elif fault == 'wall': r['wall_seconds'] = 901.
    elif fault == 'memory': r['CUDA_peak_bytes'] = [2**40, 0]
    elif fault == 'operation_link': mutate('io_input')['operation_index'] = 0
    elif fault in ('basis', 'raw_gradient', 'module_shape'):
        key = {'basis': 'grad:basis', 'raw_gradient': 'value', 'module_shape': 'predictor:0'}[fault]
        kind = 'raw_backward' if fault == 'raw_gradient' else 'io_input'
        i = next(i for i, e in enumerate(events) if e['kind'] == kind and key in e['arrays'])
        events[i] = deepcopy(events[i]); a = m.unblob(events[i]['arrays'][key]).copy()
        if fault == 'module_shape': a = a[:, :2]
        else: a.reshape(-1)[0] = 2
        events[i]['arrays'][key] = m.blob(a)
    with pytest.raises((ValueError, IndexError, KeyError)): m.validate_saved(p, r, events)


def test_failures_keep_four_slots_and_no_refund_or_retry():
    p = dict(tasks=[task(0), task(1)], profile=m.profile()); rows = m.census(); sink = Sink()
    ledger = m.core.Ledger(m.profile()['resource_caps'], sink=sink.event)
    with pytest.raises(RuntimeError): m.execute_panel(p, lambda s: RamSignal(p['tasks'][s], sink, fault=True), ledger, sink, rows)
    assert len(rows) == 4 and rows[0]['status'] == 'failed' and all(r['response'] is None for r in rows)
    assert ledger.attempted['jacobian_API'] == 1 and ledger.completed['jacobian_API'] == 0
    assert ledger.attempted['predict_API'] == ledger.completed['predict_API'] == 2
    assert ledger.attempted['signal_compile_calls'] == 1


def test_cap_blocks_before_dispatch_no_refund():
    caps = m.profile()['resource_caps']; caps['grad_API'] = 0; ledger = m.core.Ledger(caps)
    ledger.scope = {'slot': 0, 'horizon': 1}; ran = []
    with pytest.raises(m.core.BudgetExceeded): ledger.call('grad', lambda: ran.append(True))
    assert not ran and ledger.events[0]['status'] == 'blocked_before_dispatch'


def test_same_reference_and_own_nominal_differ_signed_loss_preserved():
    t = task(0); j1 = np.zeros((192, 10)); j1[0, 0] = 1
    j3 = np.zeros((192, 30)); nominal = np.zeros(192); nominal[1:] = 1
    rs = [dict(horizon=1, nominal=nominal.tolist(), jacobian=j1.tolist()),
          dict(horizon=3, nominal=np.ones(192).tolist(), jacobian=j3.tolist())]
    out = m.spectra(t, rs)
    assert out['signed_gain'] == -1 and out['scheduling_gain'] == 0
    assert out['spectra'][1]['own_nominal']['chi'] == 0
    assert out['spectra'][1]['short_nominal_reference']['chi'] == 1
    assert out['spectra'][1]['condition_number'] is None


@pytest.mark.parametrize('fault', ['shape', 'bytes', 'hash', 'nan', 'bool_shape'])
def test_lossless_blob_rejects_corruption(fault):
    b = m.blob(np.arange(10, dtype=np.float32))
    if fault == 'shape': b['shape'] = [9]
    elif fault == 'bytes': b['data'] = b['data'][:-4]
    elif fault == 'hash': b['sha256'] = '0'*64
    elif fault == 'nan':
        with pytest.raises(ValueError): m.blob(np.full(2, np.nan))
        return
    else: b['shape'] = [True]
    with pytest.raises(ValueError): m.unblob(b)


def test_store_exclusive_chain_and_torn_or_changed_line(tmp_path):
    store = m.Store(tmp_path/'RAM_fixture', m.profile())
    store.event(dict(kind='RAM_only', raw=m.blob(np.zeros((5, 2), np.float32))))
    store.event(dict(kind='RAM_end')); store.close()
    events, tail = m.journal(store.output/'events.jsonl')
    assert len(events) == 2 and tail == store.tail
    with pytest.raises(FileExistsError): m.Store(store.output, m.profile())


def test_strict_consume_actual_saved_ram_files_not_a_cuda_report(saved, tmp_path, monkeypatch):
    p, partial, events = saved; p = dict(p, sources={m.__file__: m.sha(m.__file__)})
    store = m.Store(tmp_path/'RAM_ONLY_saved_fixture', m.profile())
    binding = dict(prepared=str(tmp_path/'RAM_prepare_not_real.json'), prepare_sha256=m.digest('RAM prepare'))
    store.put('binding.json', binding)
    for e in events: store.event(e)
    store.close()
    report = dict(partial, schema=m.NAME+'.report', status='signals_saved_not_admitted', **binding,
        errors=[], sources_verified=True, cleanup_ok=True,
        journal_tail=store.tail, artifact_sha256={n: m.sha(store.output/n) for n in ('binding.json', 'events.jsonl')},
        total_pipeline_cost_claim=False, **m.FLAGS)
    store.put('report.json', report)
    # Replace ONLY this new module's input verifier; never touch old globals/model.
    seen = []
    def verify(path, expected): seen.append((path, expected)); return p
    monkeypatch.setattr(m, 'verify', verify)
    receipt = m.consume(store.output, m.sha(store.output/'report.json'))
    assert receipt['denominator'] == 4 and receipt['resources']['VJP_directions'] == 768
    assert seen == [(binding['prepared'], binding['prepare_sha256'])]
    assert all(receipt[k] is False for k in m.FLAGS)


@pytest.mark.parametrize('mutation', ['extra_outcome', 'horizon_bool', 'horizon2', 'other_parent', 'model_binding'])
def test_native_request_rejected_before_torch_import(mutation, monkeypatch):
    original_import = builtins.__import__; attempted = []
    def guarded_import(name, *args, **kwargs):
        if name == 'torch' or name.startswith('torch.'):
            attempted.append(name); raise AssertionError('invalid request tried to import torch')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded_import)
    native = object.__new__(m.NativeSignal); native.task = task(0)
    req = m.core.request(native.task, dict(context=native.task['root_context']), operation='compile_finite_horizon_J', horizon=1)
    if mutation == 'extra_outcome': req['native_success'] = True
    elif mutation == 'horizon_bool': req['horizon'] = True
    elif mutation == 'horizon2': req['horizon'] = 2
    elif mutation == 'other_parent': req['parent_context']['latent_history'][-1][0] = 1.
    else: req['binding']['model_sha256'] = m.digest('other frozen model')
    with pytest.raises(ValueError): native.signal(req, None)
    assert attempted == []  # Existing torch in another test cannot mask a new import.


def test_failed_journal_does_not_dispatch_delegate():
    def bad(e): raise OSError('RAM journal failure')
    ledger = m.core.Ledger(m.profile()['resource_caps'], sink=bad); ledger.scope = {'slot': 0, 'horizon': 1}; ran = []
    with pytest.raises(OSError): ledger.call('signal_compile', lambda: ran.append(1))
    assert not ran and ledger.attempted['signal_compile_calls'] == 1 and ledger.completed['signal_compile_calls'] == 0


def test_wrapper_cleanup_restores_instance_without_global_monkeypatch():
    class Module:
        def forward(self, *args): return args
    model = SimpleNamespace(**{k: Module() for k in ('action_encoder', 'predictor', 'pred_proj', 'encoder', 'projector')})
    native = object.__new__(m.NativeSignal); native.model = model
    originals = {k: v.forward for k, v in vars(model).items()}
    with pytest.raises(RuntimeError):
        with native.observe():
            assert model.action_encoder.forward != originals['action_encoder']; raise RuntimeError('RAM')
    assert all(v.forward == originals[k] and 'forward' not in vars(v) for k, v in vars(model).items())


def test_authorization_and_cuda_guard_before_model_import(tmp_path):
    with pytest.raises(ValueError, match='MAIN'): m.run('no', 'no', tmp_path/'no', 'worker')
    with pytest.raises(ValueError, match='CUDA'): m.NativeSignal(None, SimpleNamespace(type='cpu'), task(0), Sink())


def test_no_runtime_imports_or_generate_and_science_closed():
    script = '''
import sys
assert 'torch' not in sys.modules and 'gymnasium' not in sys.modules
from experiments.spectral_reachability import agent_p3_local_signal_cuda_20261028_v1
assert 'torch' not in sys.modules and 'gymnasium' not in sys.modules
'''
    child = subprocess.run([sys.executable, '-B', '-c', script], cwd=m.ROOT,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1'),
        capture_output=True, text=True, timeout=30)
    assert child.returncode == 0, child.stdout+child.stderr
    assert not hasattr(m.NativeSignal, 'generate') and not hasattr(m.NativeSignal, 'replay')
    assert all(v is False for v in m.FLAGS.values())
    assert m.profile()['resource_caps']['VJP_directions'] == 4*192
    assert m.profile()['resource_caps']['encoder_calls'] == 0


def test_actual_prepare_not_called_external_sha_rejected(tmp_path):
    # No actual prepare or checkpoint access in RAM tests.
    p = tmp_path/'metadata.json'
    with p.open('xb') as f: f.write(b'{}')
    with pytest.raises(ValueError, match='external prepare'): m.verify(p, '0'*64)


@pytest.mark.parametrize('geometry,kind,horizon,depth,count', [
    ('progress', 'progress', 1, 1, 4), ('reconfiguration', 'reconfiguration', 1, 3, 8),
    ('long', 'reconfiguration', 3, 3, 8), ('low_chi_gain', 'progress', 3, 3, 4)])
def test_saved_local_arrays_drive_original_same_task_type_horizon_depth(geometry, kind, horizon, depth, count):
    t = task(0); goal = np.zeros(192); goal[0] = 1
    if geometry == 'low_chi_gain': goal[1] = .5
    t['goal'] = goal.tolist(); t['binding']['goal_sha256'] = m.digest(t['goal'])
    responses = []
    for h in (1, 3):
        j = np.zeros((192, 10*h), np.float32)
        if geometry in ('progress', 'low_chi_gain') or (geometry == 'long' and h == 3): j[0, 0] = 1
        else: j[1, 0] = 1
        if geometry == 'low_chi_gain' and h == 3: j[1, 1] = 1
        responses.append(dict(binding_sha256=m.digest(t['binding']), parent_context_sha256=m.digest(t['root_context']),
            horizon=h, nominal=np.zeros(192).tolist(), jacobian=j.tolist(), producer_sha256=m.sha(m.__file__)))
    diagnostic = m.spectra(t, responses)
    node, signals, receipt = m.runtime_signal(t, responses, diagnostic)
    q = m.decide_root(t, 'adaptive', node, signals, receipt); d = q['decision']
    assert (d['program_type'], d['horizon'], d['program_depth_limit'], d['candidates']) == (kind, horizon, depth, count)
    assert d == m.core.choose(t, node, signals, 'adaptive')
    assert len(q['requests']) == count and q['program_plan'] is None
    assert all(r['signal_sha256'] == m.digest(receipt) and r['local_response'] == responses and
        r['prefix_edges'] == r['prefix_actions'] == [] and r['parent_context_sha256'] == receipt['parent_context_sha256']
        for r in q['requests'])
    fixed = m.decide_root(t, 'fixed', node, signals, receipt)['decision']
    assert (fixed['program_type'], fixed['horizon'], fixed['program_depth_limit'], fixed['candidates']) == ('progress', 1, 3, 4)
    assert q['execution_authorized'] is False and q['physical_release'] is False


def test_live_native_protocol_scheduling_signal_matches_sealed_runtime_math(saved):
    p, report, _ = saved; t = p['tasks'][0]; sink = Sink()
    ledger = m.core.Ledger(m.profile()['resource_caps'], sink=sink.event); ledger.scope = dict(slot=0, horizon=1)
    node = dict(context=t['root_context'], depth=0, indices=[], actions=[], path=[], edges=[])
    old_signal, old_receipt = m.core.local_signal(t, node, RamSignal(t, sink), ledger, m.core.policy(t, node))
    diagnostic = {k: v for k, v in report['diagnostics'][0].items() if k != 'slot'}
    _, new_signal, receipt = m.runtime_signal(t, old_receipt['responses'], diagnostic)
    assert m.asdict(old_signal) == m.asdict(new_signal) == receipt['scheduling_signal']
    assert old_receipt['signed_horizon_coverage_gain'] == receipt['signed_gain']
    assert ledger.completed['predict_API'] == 8 and ledger.completed['VJP_directions'] == 384


@pytest.mark.parametrize('fault', ['missing', 'order', 'type', 'horizon', 'depth', 'candidates', 'threshold',
    'cross_root', 'signal_seal', 'prefix', 'release', 'plan', 'journal', 'unpaid_schedule'])
def test_decision_saved_mutations_rejected_without_outcome_inputs(saved, fault):
    p, original, original_events = saved; r = dict(original); p = dict(p); events = list(original_events)
    r['decision_rows'] = deepcopy(original['decision_rows'])
    b = r['decision_rows'][1]
    if fault == 'missing': r['decision_rows'].pop()
    elif fault == 'order': r['decision_rows'].reverse()
    elif fault == 'type': b['queue']['decision']['program_type'] = 'composition'
    elif fault == 'horizon': b['queue']['decision']['horizon'] = 3
    elif fault == 'depth': b['queue']['decision']['program_depth_limit'] = 1
    elif fault == 'candidates': b['queue']['requests'].pop()
    elif fault == 'threshold':
        r['runtime_rule'] = deepcopy(r['runtime_rule']); r['runtime_rule']['policies'][0]['chi_threshold'] = .4
    elif fault == 'cross_root': b['signal']['binding_sha256'] = m.digest(p['tasks'][1]['binding'])
    elif fault == 'signal_seal': b['signal']['scheduling_signal']['source_seal'] = m.digest('other parent')
    elif fault == 'prefix': b['queue']['requests'][0]['prefix_actions'] = [[0., 0.]]*5
    elif fault == 'release': b['queue']['execution_authorized'] = True
    elif fault == 'plan': b['model_plan'] = {'commands': [[0., 0.]]*5}
    elif fault == 'journal': events = [e for e in events if e['kind'] != 'decision_request']
    else:
        r['resource_receipt'] = deepcopy(r['resource_receipt'])
        r['resource_receipt']['events'] = [e for e in r['resource_receipt']['events'] if e['resource'] != 'schedule']
    with pytest.raises((ValueError, IndexError)): m.validate_decisions(p, r, events)


def test_schedule_cap_failure_keeps_four_decision_slots_and_paid_signal():
    p = dict(tasks=[task(0), task(1)]); sink = Sink(); rows = m.census(); branches = m.decision_census()
    caps = m.profile()['resource_caps']; caps['schedule_decisions'] = 1
    ledger = m.core.Ledger(caps, sink=sink.event)
    with pytest.raises(m.core.BudgetExceeded):
        m.execute_panel(p, lambda s: RamSignal(p['tasks'][s], sink), ledger, sink, rows, branches)
    assert len(branches) == 4 and branches[0]['status'] == 'decision_prepared_not_executable'
    assert branches[1]['status'] == 'failed' and all(b['status'] == 'not_started' for b in branches[2:])
    assert ledger.attempted['predict_API'] == ledger.completed['predict_API'] == 8
    assert ledger.attempted['jacobian_API'] == ledger.completed['jacobian_API'] == 2
    assert ledger.completed['schedule_decisions'] == 1 and all(b['model_plan'] is None for b in branches)
