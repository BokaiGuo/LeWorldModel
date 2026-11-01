"""Pure NumPy/mock direct optimization: NO neural forward/reset/physics."""
from contextlib import contextmanager
from copy import deepcopy
import inspect
import json
from pathlib import Path
import numpy as np
import pytest

from experiments.spectral_reachability import p1_conditional_direct_screen_20261027_v1 as m
from experiments.spectral_reachability import p1_conditional_procedural_source_20261027_v1 as src


def task():
    t = dict(environment='array_only_not_real_environment', task_contract_sha256='f'*64,
             root_raw_action_history=np.zeros((10, 2)).tolist(), goal=[.1]*20+[0.]*172,
             projection=np.eye(192).tolist(), model_sha256='a'*64, model_config_sha256='b'*64,
             metric_source_sha256='c'*64, goal_source_sha256='d'*64, representation='native_projected_cls',
             latent_dim=192, horizon=2, packet_steps=5, control_dim=2,
             action_mean=[-.0087, .0068], action_std=[.2019, .2002], packet_logit_cap=float(.75*np.sqrt(5)),
             feedback='open_loop_model_only')
    t['root_context'] = dict(layout='states_with_between_state_actions_v1', context_length=3,
        latent_dim=192, action_dim=10, latent_history=np.zeros((3, 192)).tolist(),
        action_history=m.core.normalized_tokens(t, np.zeros((10, 2))).tolist())
    return t


def config(method='full_H2_GN'):
    return dict(method=method, B1=1, B2=3, initial_logits=[np.zeros((2, 10)).tolist()],
                step_scale=1., damping=.002, extra_line_search_trials=0)


class ArrayBackend:
    """Analytic autoregression depending on BOTH packets; not a neural mock."""
    def __init__(self, fail=None):
        self.fail = fail; self.calls = []
        self.b = np.zeros((192, 10)); self.b[:10] = np.eye(10)

    @contextmanager
    def guard(self, meter):
        yield

    def path(self, t, raw, meter, mode):
        old = deepcopy(t['root_context']); result = []
        for k, token in enumerate(m.core.normalized_tokens(t, raw)):
            meter.call('action_encoder', lambda: None)
            def predict():
                self.calls.append((mode, k))
                if self.fail == (mode, k):
                    raise RuntimeError('predict_primary')
                return .7*np.asarray(old['latent_history'][-1])+self.b@token
            z = meter.call('predict', predict)
            old = dict(old, latent_history=old['latent_history'][1:]+[z.tolist()],
                       action_history=old['action_history'][1:]+[token.tolist()])
            result.append(deepcopy(old))
            meter.emit('mock_packet', mode=mode, packet=k, context=old)
        return result

    def rollout(self, t, raw, meter, mode):
        return self.path(t, raw, meter, mode)

    def jacobian(self, t, inc, meter):
        path = self.path(t, np.asarray(inc['commands']), meter, 'jacobian')
        z = path[-1]['latent_history'][-1]
        meter.emit('jacobian_forward_complete', terminal=z)
        if self.fail == 'after_jforward':
            raise RuntimeError('AD_failure_after_saved_forward')
        std = np.tile(np.asarray(t['action_std'], np.float32), 5).astype(float)
        jraw = np.concatenate((.7*self.b/std, self.b/std), axis=1)
        for i in range(192):
            meter.call('vjp_directions', lambda: None)
        return z, jraw*(1-np.asarray(inc['commands']).reshape(20)**2)


def execute(c=None, backend=None, sink=None):
    events = []
    report = m.run_direct(task(), config() if c is None else c,
                          ArrayBackend() if backend is None else backend, events.append if sink is None else sink)
    return report, events


@pytest.mark.parametrize('method', ['full_H2_GN', 'full_H2_gradient'])
def test_genuine_continuous_fullH2_optimization_and_exact_budget(method):
    c = config(method); r, events = execute(c)
    assert r['error'] is None and r['status'] == 'direct_component_completed_not_admitted'
    run = r['direct_runs'][0]
    assert len(run) == 4 and run[-1]['incumbent']['cost'] < run[0]['incumbent']['cost']
    assert all(step['accepted'] for step in run[1:])
    for i, step in enumerate(run):
        m.core.check_prediction(task(), step['incumbent'])
        if i:
            m.core.check_derivative(task(), step['derivative'], run[i-1]['incumbent'])
    assert r['budget']['predict'] == 30 and r['budget']['jacobian'] == 3
    assert r['budget']['vjp_directions'] == 576
    for key, cap in r['budget'].items():
        rows = [e for e in r['extended_journal'] if e['resource'] == key]
        assert len(rows) == cap and all(e['completed'] for e in rows)
    assert len([e for e in events if e['kind'] == 'trajectory_step']) == 4
    assert not r['P1_admission'] and not r['model_task_eligible']
    assert 'independent_threshold_authority' in r['missing_components']


def test_receipts_fit_old_direct_group_contract_without_fake_witness():
    r, _ = execute(); c = config()
    for i in range(4):
        events = [e for e in r['journal'] if e['evaluation_id'] == f'run_0_iter_{i}']
        assert {key: sum(e['samples'] for e in events if e['resource'] == key) for key in m.core.RESOURCES} == dict(predict=6, action_encoder=6, jacobian=0, svd=0)
    for i in range(1, 4):
        events = [e for e in r['journal'] if e['evaluation_id'] == f'opt_0_iter_{i}']
        assert {key: sum(e['samples'] for e in events if e['resource'] == key) for key in m.core.RESOURCES} == dict(predict=2, action_encoder=2, jacobian=1, svd=1)
    p = dict(budgets=dict(direct={k: r['budget'][k] for k in m.core.RESOURCES},
                         producer={k: 0 for k in m.core.RESOURCES}, witness={k: 0 for k in m.core.RESOURCES}))
    assert m.core.ledger(p, r['journal'], False)['direct']['predict']['completed'] == 30


def test_rejected_proposal_preserves_incumbent_and_derivative_condition():
    c = config(); c['step_scale'] = 10000
    r, _ = execute(c)
    assert r['error'] is None
    for step in r['direct_runs'][0][1:]:
        assert not step['accepted']
        assert step['incumbent'] == r['direct_runs'][0][0]['incumbent']
        assert step['derivative']['at_logits'] == r['direct_runs'][0][0]['incumbent']['logits']


def test_multiple_restarts_are_predeclared_and_fully_paid():
    c = config(); c['initial_logits'].append(np.full((2, 10), np.float32(.02)).astype(float).tolist())
    r, _ = execute(c)
    assert r['error'] is None and len(r['direct_runs']) == 2 and r['budget']['predict'] == 60


@pytest.mark.parametrize('defect', ['tau', 'candidate_bank', 'extra_line_search_trials', 'B1', 'damping', 'initial_logits', 'future_state'])
def test_reject_undeclared_or_incompatible_work_before_backend(defect):
    c = config(); backend = ArrayBackend()
    if defect == 'extra_line_search_trials': c[defect] = 1
    elif defect == 'B1': c[defect] = 3
    elif defect == 'damping': c[defect] = 0
    elif defect == 'initial_logits': c[defect][0][0][0] = .1  # not exactly float32
    else: c[defect] = object()
    with pytest.raises(ValueError): execute(c, backend)
    assert backend.calls == []


@pytest.mark.parametrize('defect', ['projection', 'goal', 'action_std', 'feedback'])
def test_native_backend_task_pin_rejects_Q_goal_scale_feedback_drift_without_cuda(defect):
    # _start checks task identity before touching torch. No GPU object needed.
    backend = object.__new__(m.NativeCuda); backend.bound_id = m.core.task_id(task())
    t = task()
    if defect == 'projection': t[defect][0][0] = 0
    elif defect == 'goal': t[defect][0] += 1
    elif defect == 'action_std': t[defect][0] *= 2; t['root_context']['action_history'] = m.core.normalized_tokens(t, np.zeros((10, 2))).tolist()
    else: t[defect] = 'closed_loop'
    with pytest.raises(ValueError): backend._start(t)


def test_full_joint_chain_both_packets_with_nontrivial_Q_and_finite_difference():
    t = task(); t['projection'][2][2] = 0
    x = np.full((2, 10), np.float32(.1)).astype(float)
    meter = m.Meter(dict(predict=100, action_encoder=100, vjp_directions=192), lambda e: None)
    b = ArrayBackend(); inc = m.evaluate(t, x, b, meter)
    z, j = b.jacobian(t, inc, meter)
    g = 2*(np.asarray(t['projection'])@j).T@(np.asarray(t['projection'])@(np.asarray(z)-t['goal']))/192
    for col in (0, 9, 10, 19):
        values = []
        for sign in (-1, 1):
            trial = x.copy(); trial.reshape(20)[col] += sign*.001
            p = b.path(t, np.tanh(trial.astype(np.float32)).reshape(10, 2), meter, 'fd')
            values.append(m.core.model_cost(t, p[-1]['latent_history'][-1]))
        assert g[col] == pytest.approx((values[1]-values[0])/.002, rel=3e-4, abs=1e-7)
    assert np.any(j[:, :10]) and np.any(j[:, 10:])


def test_nonrefundable_predict_failure_partial_trajectory_and_journal():
    r, events = execute(backend=ArrayBackend(('jacobian', 1)))
    assert r['status'] == 'failed' and r['error']['message'] == 'predict_primary'
    assert len(r['direct_runs'][0]) == 1
    assert any(e['resource'] == 'predict' and not e['completed'] for e in r['journal'])
    assert any(e['resource'] == 'jacobian' and not e['completed'] for e in r['journal'])
    assert any(e['kind'] == 'mock_packet' and e['mode'] == 'jacobian' for e in events)


def test_forward_arrays_saved_before_reverse_failure():
    r, events = execute(backend=ArrayBackend('after_jforward'))
    assert r['status'] == 'failed'
    assert any(e['kind'] == 'jacobian_forward_complete' for e in events)
    assert sum(e['resource'] == 'vjp_directions' for e in r['extended_journal']) == 0


def test_primary_failure_not_replaced_by_final_sink_failure():
    def sink(e):
        if e['kind'] == 'direct_finished': raise OSError('final_sink_secondary')
    r, _ = execute(backend=ArrayBackend(('jacobian', 1)), sink=sink)
    assert r['error']['message'] == 'predict_primary'
    assert r['secondary_errors'][0]['message'] == 'final_sink_secondary'


def test_attempt_journal_failure_stops_before_forward_and_is_not_refunded():
    b = ArrayBackend()
    def sink(e):
        if e['kind'] == 'operation_attempted': raise OSError('disk_full_mock')
    r, _ = execute(backend=b, sink=sink)
    assert r['status'] == 'failed' and b.calls == []
    assert len(r['journal']) == 1 and not r['journal'][0]['completed']


def test_predispatch_cap_is_real_not_posthoc_accounting():
    meter = m.Meter(dict(predict=1), lambda e: None); called = []
    meter.call('predict', lambda: called.append(1))
    with pytest.raises(ValueError): meter.call('predict', lambda: called.append(2))
    assert called == [1]


def test_small_durable_journal_exclusive_and_reconstructable(tmp_path):
    sink = m.DiskJournal(tmp_path/'small', byte_limit=8*1024**2)
    c = config(); c['B2'] = 2
    r, _ = execute(c, sink=sink)
    assert r['error'] is None
    rows = [json.loads(line) for line in sink.path.read_text().splitlines()]
    assert rows[-1]['report']['status'] == 'direct_component_completed_not_admitted'
    assert len([e for e in rows if e['kind'] == 'trajectory_step']) == 3
    with pytest.raises(ValueError): m.DiskJournal(tmp_path/'small', byte_limit=1)


def source_fixture(tmp_path, monkeypatch):
    # Isolate the bridge, NOT a fake global freshness certificate. Real raw
    # writer/reader and SHA validation are used; source plan readiness is mocked.
    p = dict(purpose='calibration_exploration', slots=[dict(candidate_id=0, root_seed=100, reset_seed_span=[200, 201], split='calibration')],
             environment=task()['environment'], generator_sha256='e'*64, environment_sha256='a'*64,
             proposal_limit_per_seed=1, reset_attempts_per_proposal=1, placement_reset_seed=0,
             recipe={}, domain={}, history={}, rgb_shape=[4, 4, 3])
    monkeypatch.setattr(src, 'validate_plan', lambda p, e: dict(ready=True))
    path = tmp_path/'source.npz'; frame = np.full((4, 4, 3), 100, np.uint8); goal = frame+1
    src.write_raw_source(path, p, p['slots'][0], 0, 0, np.zeros(7), np.ones(7), frame, goal)
    template = {k: v for k, v in task().items() if k not in ('root_context', 'goal', 'root_raw_action_history')}
    template['goal_source_sha256'] = m.core.sha(goal.tobytes())
    return p, path, template


def test_source_to_encoding_to_true_task_identity_no_state_in_callback(tmp_path, monkeypatch):
    p, path, template = source_fixture(tmp_path, monkeypatch); seen = []
    def encode(rgb):
        assert rgb.dtype == np.uint8 and rgb.shape == (4, 4, 3)
        seen.append(rgb.copy()); return np.full(192, float(rgb[0, 0, 0]))
    receipt = m.encode_source(path, p, {}, src.file_sha(path), template, encode)
    assert len(seen) == 4 and receipt['task_id'] == m.core.task_id(receipt['task'])
    assert receipt['history_semantics'] == 'causal_padding_not_observed_transitions'
    assert not receipt['census_validated'] and not receipt['P1_admission']
    assert receipt['source_purpose'] == 'calibration_exploration' and not receipt['freshness_certified']
    assert receipt['native_image_overhead'] == dict(encoder=4, projector=4)
    assert receipt['task']['root_context']['latent_history'][0] == [100.]*192
    assert receipt['task']['goal'] == [101.]*192


@pytest.mark.parametrize('defect', ['hash', 'not_ready', 'physical_state', 'goal_hash', 'action_std', 'transplant'])
def test_source_failclosed_before_native_encoding(tmp_path, monkeypatch, defect):
    p, path, template = source_fixture(tmp_path, monkeypatch); calls = []; h = src.file_sha(path)
    if defect == 'hash': h = '0'*64
    elif defect == 'not_ready': monkeypatch.setattr(src, 'validate_plan', lambda p, e: dict(ready=False))
    elif defect == 'physical_state': template[defect] = object()
    elif defect == 'goal_hash': template['goal_source_sha256'] = '0'*64
    elif defect == 'action_std': template[defect][0] = 0
    else: template['root_context'] = task()['root_context']
    with pytest.raises(ValueError): m.encode_source(path, p, {}, h, template, lambda x: calls.append(x))
    assert calls == []


def test_cuda_factories_reject_cpu_before_import_or_forward():
    class Device: type = 'cpu'
    with pytest.raises(ValueError): m.NativeCuda(None, Device(), task())
    with pytest.raises(ValueError): m.native_image_encoder(None, Device(), None, None)


def test_native_joint_graph_not_candidate_library_or_prefix_detach():
    text = inspect.getsource(m.NativeCuda.jacobian)
    assert 'raw.requires_grad' not in text  # explicit tensor leaf construction below
    assert 'requires_grad_(True)' in text and 'range(2)' in text and 'range(192)' in text
    assert 'self._step(task, z, h' in text
    assert 'bounded_ridge' not in text and 'spectral_bank' not in text


class HookOnlyModule:
    """No tensors/computation: exercise real backend guard with fake entries."""
    def __init__(self): self.pre = []; self.post = []; self.fail_remove = False
    def _register(self, target, fn):
        target.append(fn); owner = self
        class Handle:
            def remove(self):
                target.remove(fn)
                if owner.fail_remove: raise OSError('cleanup_secondary')
        return Handle()
    def register_forward_pre_hook(self, fn): return self._register(self.pre, fn)
    def register_forward_hook(self, fn, always_call):
        assert always_call
        return self._register(self.post, fn)
    def entry(self, fail=False):
        for fn in self.pre: fn(self, ())
        for fn in self.post: fn(self, (), None if fail else object())


def guarded_backend():
    class Model: pass
    model = Model()
    for key in ('predictor', 'action_encoder', 'encoder', 'projector'):
        setattr(model, key, HookOnlyModule())
    backend = object.__new__(m.NativeCuda); backend.model = model
    return backend


def test_native_actual_module_entry_charges_and_forbids_image_calls():
    backend = guarded_backend(); meter = m.Meter(dict(predict=1, action_encoder=1), lambda e: None)
    with backend.guard(meter):
        backend.model.action_encoder.entry(); backend.model.predictor.entry()
        with pytest.raises(ValueError): backend.model.encoder.entry()
        with pytest.raises(ValueError): backend.model.projector.entry()
        with pytest.raises(ValueError): backend.model.predictor.entry()
    assert [e['resource'] for e in meter.events] == ['action_encoder', 'predict']
    assert all(e['completed'] for e in meter.events)
    assert not hasattr(backend, '_active_meter')
    assert all(not getattr(backend.model, k).pre for k in ('encoder', 'projector', 'predictor', 'action_encoder'))


def test_native_failed_module_attempt_not_refunded_and_cleanup_primary_kept():
    backend = guarded_backend(); meter = m.Meter(dict(predict=1, action_encoder=1), lambda e: None)
    backend.model.predictor.fail_remove = True
    with pytest.raises(RuntimeError, match='primary_forward') as caught:
        with backend.guard(meter):
            backend.model.predictor.entry(fail=True)
            raise RuntimeError('primary_forward')
    assert not meter.events[0]['completed']
    assert 'cleanup_secondary' in str(caught.value.__notes__)
    assert not hasattr(backend, '_active_meter')
