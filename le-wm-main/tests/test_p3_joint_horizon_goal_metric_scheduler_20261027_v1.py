"""Pure NumPy and constant-interface mocks only; no torch/model/physics runs."""
from copy import deepcopy
from contextlib import nullcontext
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import p3_joint_horizon_goal_metric_scheduler_20261027_v1 as m


def packet():
    p = dict(root_sha256='fixed-root', root_context=dict(m.META,
        latent_history=np.zeros((3, 192)).tolist(), action_history=np.zeros((2, 10)).tolist()),
        goal=np.eye(1, 192)[0].tolist(), projection=np.eye(192).tolist())
    return dict(p, input_sha256=m.digest(p))


def record():
    p = packet()
    q = np.zeros((192, 192)); q[1, 1] = 1
    qs = dict(old_Q=p['projection'], goal_patch_Q=q.tolist())
    return dict(packet=p, seed_index=0, provenance={}, metrics={label:
        dict(projection=projection, active_metric_id=m.active_id(p, label, projection, {}))
        for label, projection in qs.items()})


def compilation(r):
    out = dict(raw={}, metrics={label: {} for label in m.METRICS})
    for h in (2, 3):
        j = np.zeros((192, 10*h)); j[0, 0] = 1
        n = np.zeros(192)
        out['raw'][str(h)] = dict(jacobian=j.tolist(), nominal=n.tolist())
        for label in m.METRICS:
            out['metrics'][label][str(h)] = m.response(r['packet'], r['metrics'][label]['projection'],
                j, n, h, {}, lambda e: None)
    return out


def model():
    f = lambda x: x
    return SimpleNamespace(predict=f, action_encoder=SimpleNamespace(forward=f), encoder=SimpleNamespace(forward=f))


def tensor(kind='predict', device='cuda'):
    return SimpleNamespace(shape=(1, 3, 192 if kind == 'predict' else 10), device=SimpleNamespace(type=device))


@pytest.mark.parametrize('h', [2, 3])
def test_own_q_response_and_ridge_not_score_only(h):
    r = record(); c = compilation(r)
    a, b = (c['metrics'][label][str(h)] for label in m.METRICS)
    assert a['proposal_logits'][0] == pytest.approx(1/1.002)
    assert not np.any(b['proposal_logits'])
    assert a['signal']['rank'] == 1 and b['signal']['rank'] == 0
    assert a['signal']['chi'] == b['signal']['chi'] == 0
    assert a['solve']['params'][0] != pytest.approx(1/(1+.002*192))
    assert a['solve']['linear_terminal_cost'] == pytest.approx((1-a['proposal_logits'][0])**2/192)


@pytest.mark.parametrize('h', [2, 3])
def test_bank_tail_same_q_prefix_seed_and_logit_ball(h):
    r = record(); c = compilation(r)
    a, b = [m.candidate_bank(0, h, 20, c['metrics'][q][str(h)]) for q in m.METRICS]
    original = m.old.pool(0, h, 20)
    assert all(np.array_equal(a[k], b[k]) and np.array_equal(a[k], original[k]) for k in range(2, 20))
    assert np.array_equal(a[0], original[0]) and not np.array_equal(a[1], b[1])
    assert all(np.array_equal(x, y) for x, y in zip(a, m.candidate_bank(0, h, 30, c['metrics']['old_Q'][str(h)])))
    assert not np.array_equal(a[2], m.candidate_bank(1, h, 20, c['metrics']['old_Q'][str(h)])[2])
    assert all(np.linalg.norm(np.arctanh(x)) <= .75*np.sqrt(5*h)+1e-12 for x in a)
    assert a[1].reshape(-1)[0] == np.tanh(c['metrics']['old_Q'][str(h)]['proposal_logits'][0])


def test_ridge_ball_and_zero_rank_nonzero_residual():
    p = packet(); p['goal'][0] = 100
    j = np.zeros((192, 20)); j[0, 0] = 1
    log = []
    c = m.response(p, np.eye(192), j, np.zeros(192), 2, {}, log.append)
    assert np.linalg.norm(c['proposal_logits']) == pytest.approx(.75*np.sqrt(10))
    assert c['solve']['ball_multiplier'] > 0
    z = m.response(p, np.eye(192), j*0, np.zeros(192), 2, {}, log.append)
    assert z['signal']['chi'] == 1 and not np.any(z['proposal_logits'])
    operations = [e for e in log if e['kind'] == 'svd']
    assert len(operations) == 8 and [e['completed'] for e in operations] == [False, True]*4
    assert any(e['kind'] == 'response_arrays' and 'solve' in e['arrays'] for e in log)


@pytest.mark.parametrize('value', [np.zeros((192, 192)), np.eye(192)*2, np.eye(3), np.full((192, 192), np.nan)])
def test_bad_q_strict_refusal(value):
    with pytest.raises(ValueError):
        m.projection(value)


def test_original_input_and_active_q_goal_binding():
    p = packet(); m.validate_packet(p)
    a = m.active_id(p, 'old_Q', p['projection'], {'weights': 'a'})
    q = deepcopy(p['projection']); q[0][0] = 0
    assert a != m.active_id(p, 'old_Q', q, {'weights': 'a'})
    assert a != m.active_id(p, 'old_Q', p['projection'], {'weights': 'b'})
    p['goal'][0] = 2
    assert a != m.active_id(p, 'old_Q', p['projection'], {'weights': 'a'})
    with pytest.raises(ValueError):
        m.validate_packet(p)


@pytest.mark.parametrize('a,b,blocks', [(0,0,3), (1,0,8), (1,1,3), (.8,.4,5), (.2,.8,3)])
def test_dynamic_budget_extremes(a, b, blocks):
    allocation = m.old.allocate('dynamic', {'2': dict(chi=a), '3': dict(chi=b)})
    assert allocation['blocks']['3'] == blocks
    assert 2*allocation['candidates']['2']+3*allocation['candidates']['3'] == 60
    assert min(allocation['candidates'].values()) >= 2
    assert m.contract()['root_predict_cap'] == 10+4*66 == 274
    assert m.contract()['four_root_predict_cap'] == 1096


@pytest.mark.parametrize('kind', ['predict', 'action_encoder'])
def test_all_four_arms_and_independent_caps(kind):
    f = model(); log = []
    original = f.predict
    with m.NeuralCap(f, log.append) as cap:
        for role in ('shared', *m.ARMS):
            with cap.scope(role):
                fn = f.predict if kind == 'predict' else f.action_encoder.forward
                for _ in range(10 if role == 'shared' else 66): fn(tensor(kind))
                with pytest.raises(ValueError): fn(tensor(kind))
        with pytest.raises(ValueError): f.encoder.forward(tensor())
    assert cap.receipt()['attempted'][kind] == 274 and cap.blocked == 5
    assert f.predict is original


def test_failure_attempt_completed_and_restoration():
    f = model(); log = []
    def fail(x): raise RuntimeError('attempt failed')
    f.predict = fail
    with m.NeuralCap(f, log.append) as cap:
        with cap.scope(m.ARMS[0]):
            with pytest.raises(RuntimeError): f.predict(tensor())
            with pytest.raises(ValueError): f.predict(tensor(device='cpu'))
        with pytest.raises(ValueError): f.predict(tensor())
    assert cap.receipt()['attempted']['predict'] == 1
    assert cap.receipt()['completed']['predict'] == 0
    assert log[0]['completed'] is False and 'error' in log[-1]
    assert f.predict is fail


def test_search_all_candidates_duplicate_retention_and_stable_tie():
    r = record(); c = compilation(r); out = {}; calls = []
    def replay(context, actions):
        calls.append(len(actions)//5)
        return deepcopy(context)
    m.search(r, 'goal_patch_Q__uniform', c, replay, out, lambda e: None)
    assert len(out['candidates']) == 25
    assert out['candidates'][0]['actions'] == out['candidates'][1]['actions']
    assert out['winner']['horizon'] == 2 and out['winner']['index'] == 0
    assert sum(calls) == 64 and out['verification_predicts'] == 4
    assert out['active_metric_id'] == r['metrics']['goal_patch_Q']['active_metric_id']


def test_search_uses_own_q_signal_and_scoring():
    r = record(); c = compilation(r)
    c['metrics']['old_Q']['2']['signal']['chi'] = 1
    c['metrics']['old_Q']['3']['signal']['chi'] = 0
    results = {}
    for label in m.METRICS:
        out = {}
        def replay(context, actions):
            context['latent_history'][-1][0] = float(np.asarray(actions).reshape(-1)[0])
            return context
        # Whole/segmented mock need not agree for arbitrary sequence; zero wins for new Q.
        if label == 'goal_patch_Q':
            m.search(r, label+'__dynamic', c, replay, out, lambda e: None)
            assert all(x['cost'] == 0 for x in out['candidates'])
        else:
            with pytest.raises(ValueError, match='exact complete-context'):
                m.search(r, label+'__dynamic', c, replay, out, lambda e: None)
            assert min(x['cost'] for x in out['candidates']) < out['candidates'][0]['cost']
        results[label] = out
    assert results['old_Q']['allocation']['blocks']['3'] == 8
    assert results['goal_patch_Q']['allocation']['blocks']['3'] == 3


def test_partial_candidate_arrays_preserved_after_second_failure():
    r = record(); c = compilation(r); target = {}; calls = []
    def replay(context, actions):
        calls.append(1)
        if len(calls) == 2: raise RuntimeError('second candidate')
        return deepcopy(context)
    with pytest.raises(RuntimeError):
        m.search(r, m.ARMS[0], c, replay, target, lambda e: None)
    assert len(target['candidates']) == 2
    assert target['candidates'][0]['completed'] and 'context' in target['candidates'][0]
    assert not target['candidates'][1]['completed'] and 'actions' in target['candidates'][1]
    assert target['status'] == 'incomplete'


def test_signal_survives_ridge_exception(monkeypatch):
    target = {}
    def fail(*a, **k): raise RuntimeError('ridge failed')
    monkeypatch.setattr(m.kernel, 'bounded_ridge', fail)
    with pytest.raises(RuntimeError):
        m.response(packet(), np.eye(192), np.zeros((192, 20)), np.zeros(192), 2, target, lambda e: None)
    assert 'signal' in target and 'response' in target and 'residual' in target


def test_accounting_exact_and_failed_or_missing_arm_refused():
    f = model(); arms = {arm: dict(status='completed', verification_predicts=4) for arm in m.ARMS}
    with m.NeuralCap(f, lambda e: None) as cap:
        for role in ('shared', *m.ARMS):
            with cap.scope(role):
                for _ in range(10 if role == 'shared' else 64):
                    f.action_encoder.forward(tensor('action_encoder')); f.predict(tensor())
    receipt = cap.receipt()
    assert m.accounting_ok(receipt, arms)
    receipt['events'][0]['completed'] = False
    assert not m.accounting_ok(receipt, arms)
    assert not m.accounting_ok(cap.receipt(), {m.ARMS[0]: arms[m.ARMS[0]]})


def test_prepare_and_validation_are_model_free_with_source_reconstruction(tmp_path, monkeypatch):
    # Only constant protocol fixtures + real small source sealing in tmp_path.
    r = record(); records = [dict(deepcopy(r), seed_index=i) for i in range(4)]
    for i, v in enumerate(records): v['packet']['root_sha256'] = f'fixture-{i}'
    source = tmp_path/'source.txt'; source.write_text('constant fixture')
    monkeypatch.setattr(m, 'source_material', lambda root: (records, {}, {'known': 'fixture'}, {}))
    monkeypatch.setattr(m, 'normalizer_binding', lambda root, sources: dict(mean=[0., 0.], std=[1., 1.]))
    monkeypatch.setattr(m.old.prior.physical, 'source_closure', lambda root, paths: [source])
    out = tmp_path/'prepare'
    receipt = m.prepare(tmp_path, out)
    sha = m.io.sha((out/'prepare.json').read_bytes())
    assert receipt['new_neural_calls'] == 0
    assert m.validate_prepared(out, sha)['records'] == records
    with pytest.raises(ValueError): m.prepare(tmp_path, out)
    with pytest.raises(ValueError): m.validate_prepared(out, 'bad-sha')
    source.write_text('drift')
    with pytest.raises(ValueError, match='drift'): m.validate_prepared(out, sha)


def test_hook_cleanup_preserves_primary_error_and_restores_other_hooks():
    class Endpoint:
        def __init__(self):
            self.forward = lambda x: x
            self.fail_restore = False
        def __setattr__(self, name, value):
            if name == 'forward' and getattr(self, 'fail_restore', False):
                raise RuntimeError('cleanup fixture')
            object.__setattr__(self, name, value)
    f = model(); f.action_encoder = Endpoint()
    original_predict, original_encoder = f.predict, f.encoder.forward
    with pytest.raises(RuntimeError, match='PRIMARY'):
        with m.NeuralCap(f, lambda e: None) as cap:
            f.action_encoder.fail_restore = True
            raise RuntimeError('PRIMARY')
    assert cap.receipt()['cleanup_errors'][0]['message'] == 'cleanup fixture'
    assert f.predict is original_predict and f.encoder.forward is original_encoder


@pytest.mark.parametrize('persistence_failure', ['none', 'journal_hash', 'root_write', 'both'])
def test_runner_failure_partials_survive_source_and_model_final_errors(tmp_path, monkeypatch, persistence_failure):
    # Entire CUDA runtime/loader/preflight replaced with constant interfaces.
    # No torch tensor, model module, device allocation or autograd is executed.
    from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
    from experiments.spectral_reachability import pusht_composition_cuda_replay_v2 as adapter
    fake_torch = ModuleType('torch')
    fake_torch.device = lambda x: SimpleNamespace(type='cuda')
    fake_torch.cuda = SimpleNamespace(synchronize=lambda x: None)
    attention = ModuleType('torch.nn.attention')
    attention.SDPBackend = SimpleNamespace(MATH='mock-MATH')
    attention.sdpa_kernel = lambda backend: nullcontext()
    for name, value in [('torch', fake_torch), ('torch.nn', ModuleType('torch.nn')),
            ('torch.nn.attention', attention)]: monkeypatch.setitem(sys.modules, name, value)
    f = model(); attempts = []
    def predict(x):
        attempts.append(1)
        if len(attempts) == 2: raise RuntimeError('PRIMARY second predictor failure')
        return 'constant first result'
    f.predict = predict
    f.eval = lambda: f
    f.requires_grad_ = lambda flag: f
    loader = ModuleType('experiments.checkpoint_compat')
    loader.load_compatible_pretrained = lambda *a, **k: f
    preflight = ModuleType('experiments.gpu_preflight')
    preflight.require_cuda_allocation = lambda x: {'mock_runtime': True}
    monkeypatch.setitem(sys.modules, 'experiments.checkpoint_compat', loader)
    monkeypatch.setitem(sys.modules, 'experiments.gpu_preflight', preflight)
    monkeypatch.setattr(adapter, 'CudaReplay', lambda *a: None)
    def file_digest(path):
        if str(path).endswith('.events.jsonl') and persistence_failure in ('journal_hash', 'both'):
            raise OSError('journal digest failed')
        return m.io.sha(Path(path).read_bytes()) if Path(path).exists() else 'weightsha'
    monkeypatch.setattr(checked, 'digest', file_digest)
    original_write = m.write
    def faulty_write(path, value):
        if Path(path).name == 'root_0.json' and persistence_failure in ('root_write', 'both'):
            raise OSError('root write failed')
        original_write(path, value)
    monkeypatch.setattr(m, 'write', faulty_write)
    hashes = []
    def state_hash(value):
        hashes.append(1)
        if len(hashes) == 2: raise RuntimeError('final hash failed')
        return 'frozen'
    monkeypatch.setattr(checked, 'tensor_state_hash', state_hash)
    records = [record() for _ in range(4)]
    for i, r in enumerate(records): r['packet']['root_sha256'] = f'mock-{i}'
    p = dict(records=records, contract=m.contract(), model_bindings={'/fixture/weights.pt': 'weightsha'},
        arm_order=[list(m.ARMS)]*4, inherited_resources={})
    validations = []
    def validate(*a):
        validations.append(1)
        if len(validations) == 2: raise RuntimeError('source disappeared')
        return p
    monkeypatch.setattr(m, 'validate_prepared', validate)
    def partial_compile(model, device, record, target, journal):
        target['nominal_array_fixture'] = [1., 2., 3.]
        target['first_result'] = model.predict(tensor())
        model.predict(tensor())
    monkeypatch.setattr(m, 'compile_shared', partial_compile)
    out = tmp_path/'run'
    with pytest.raises(ValueError, match='incomplete'):
        m.execute_cuda(tmp_path/'unused-prepare', 'mock-sha', out)
    result = json.loads((out/'report.json').read_text())
    persistence_stages = []
    if persistence_failure in ('journal_hash', 'both'): persistence_stages.append('root_journal_hash')
    if persistence_failure in ('root_write', 'both'): persistence_stages.append('root_write')
    assert [e['stage'] for e in result['errors']] == ['execution', *persistence_stages, 'model_after', 'source_after']
    assert 'PRIMARY' in result['errors'][0]['message']
    assert result['roots'][0]['compilation']['nominal_array_fixture'] == [1., 2., 3.]
    receipt = result['roots'][0]['neural_receipt']
    assert receipt['attempted']['predict'] == 2 and receipt['completed']['predict'] == 1
    assert all(r['status'] == 'blocked' for r in result['roots'][1:])
    assert all(set(r['arms']) == set(m.ARMS) for r in result['roots'])
    assert result['resources']['incomplete_counts_are_lower_bound'] is True


NORMALIZATION = dict(mean=np.asarray([-.0087, .0068], np.float32).astype(float).tolist(),
    std=np.asarray([.2019, .2002], np.float32).astype(float).tolist())


def array_rollout(before, actions, normalization=NORMALIZATION):
    """Constant latent arrays, actual pure float32 action token arithmetic."""
    out = deepcopy(before)
    tokens = ((np.asarray(actions, np.float32)-np.asarray(normalization['mean'], np.float32)) /
        np.asarray(normalization['std'], np.float32)).reshape(-1, 10)
    for token in tokens:
        out['latent_history'] = out['latent_history'][1:]+[np.zeros(192).tolist()]
        out['action_history'] = out['action_history'][1:]+[token.astype(float).tolist()]
    return out


@pytest.mark.parametrize('h', [1, 2, 3])
def test_float32_history_content_order_metadata_and_segments(h):
    before = packet()['root_context']
    actions = np.linspace(-.73, .82, 10*h).reshape(5*h, 2)
    after = array_rollout(before, actions)
    m.check_history(before, after, actions, NORMALIZATION)
    context = deepcopy(before)
    for raw in actions.reshape(h, 5, 2):
        successor = array_rollout(context, raw)
        m.check_history(context, successor, raw, NORMALIZATION)
        context = successor
    assert context == after
    wrong = deepcopy(after); wrong['action_history'].reverse()
    with pytest.raises(ValueError, match='history'): m.check_history(before, wrong, actions, NORMALIZATION)
    wrong = deepcopy(after); wrong['action_dim'] = 2
    with pytest.raises(ValueError, match='metadata'): m.check_history(before, wrong, actions, NORMALIZATION)
    wrong = deepcopy(after); wrong['unbound_metadata'] = True
    with pytest.raises(ValueError, match='metadata'): m.check_history(before, wrong, actions, NORMALIZATION)
    wrong = deepcopy(after); wrong['action_history'][1][3] += 1e-12
    with pytest.raises(ValueError, match='history'): m.check_history(before, wrong, actions, NORMALIZATION)
    with pytest.raises(ValueError): m.check_history(before, after, actions/100, NORMALIZATION)
    double_tokens = ((actions-np.array(NORMALIZATION['mean']))/np.array(NORMALIZATION['std'])).reshape(h, 10)
    wrong = deepcopy(after); wrong['action_history'][-1] = double_tokens[-1].tolist()
    with pytest.raises(ValueError): m.check_history(before, wrong, actions, NORMALIZATION)


def test_normalizer_ast_exact_source_hash(tmp_path):
    path = tmp_path/'experiments'/'run_pusht_counterfactual.py'
    path.parent.mkdir()
    path.write_text('PUBLIC_ACTION_MEAN=np.asarray([-0.0087,0.0068],dtype=np.float32)\n'
        'PUBLIC_ACTION_STD=np.asarray([0.2019,0.2002],dtype=np.float32)\n')
    sources = {str(path): m.io.sha(path.read_bytes())}
    n = m.normalizer_binding(tmp_path, sources)
    assert n['mean'] == NORMALIZATION['mean'] and n['std'] == NORMALIZATION['std']
    path.write_text(path.read_text()+'# drift\n')
    with pytest.raises(ValueError, match='source bytes'): m.normalizer_binding(tmp_path, sources)


def test_predict_primary_survives_terminal_journal_failure():
    f = model(); logs = []
    def fail(x): raise RuntimeError('PRIMARY predictor')
    f.predict = fail
    def journal(e):
        logs.append(deepcopy(e))
        if 'error' in e: raise OSError('SECONDARY journal')
    with pytest.raises(RuntimeError, match='PRIMARY predictor'):
        with m.NeuralCap(f, journal) as cap:
            with cap.scope('shared'): f.predict(tensor())
    assert cap.receipt()['attempted']['predict'] == 1
    assert cap.receipt()['journal_errors'][0]['message'] == 'SECONDARY journal'


@pytest.mark.parametrize('fail_at', [None, 2, 4])
def test_compile_prefix_nominal_jacobian_arrays_journal_immediate(monkeypatch, fail_at):
    # NumPy-backed constant interface only; no actual torch tensors or AD.
    class T:
        def __init__(self, value): self.value = np.asarray(value, np.float32)
        def __getitem__(self, key): return T(self.value[key])
        def reshape(self, *shape): return T(self.value.reshape(*shape))
        def detach(self): return self
        def cpu(self): return self
        def tolist(self): return self.value.tolist()
        def new_zeros(self, size): return T(np.zeros(size))
        def requires_grad_(self, flag): return self
    torch = ModuleType('torch')
    torch.float32 = 'constant-interface'
    torch.tensor = lambda value, **kw: T(value)
    torch.tanh = lambda value: T(np.tanh(value.value))
    torch.no_grad = nullcontext
    def jacobian(fn, zero, **kw):
        fn(zero)
        return T(np.zeros((192, len(zero.value))))
    torch.autograd = SimpleNamespace(functional=SimpleNamespace(jacobian=jacobian))
    monkeypatch.setitem(sys.modules, 'torch', torch)
    transitions = []
    def transition(state, between, raw):
        transitions.append(1)
        if len(transitions) == fail_at: raise RuntimeError('prefix failed')
        return state, between
    monkeypatch.setattr(m.kernel, 'make_transition', lambda *args: transition)
    target, log = {}, []
    if fail_at is None:
        m.compile_shared(None, None, record(), target, log.append)
        assert len(transitions) == 10
        assert len([e for e in log if e['kind'] == 'jacobian_array']) == 2
        assert len([e for e in log if e['kind'] == 'nominal_array']) == 2
        assert len([e for e in log if e['kind'] == 'compile_boundary_arrays']) == 10
    else:
        with pytest.raises(RuntimeError): m.compile_shared(None, None, record(), target, log.append)
        phase = 'nominal' if fail_at == 2 else 'jacobian_forward'
        saved = [e for e in log if e['kind'] == 'compile_boundary_arrays' and e['phase'] == phase]
        assert len(saved) == 1 and saved[0]['context'] == target['raw']['2'][phase+'_segments'][0]
        if fail_at == 4: assert any(e['kind'] == 'nominal_array' for e in log)


def test_export_rejects_consistently_duplicated_wrong_history(tmp_path, monkeypatch):
    # Isolate saved-context content verification; entry accounting has separate tests.
    r = record(); compiled = compilation(r)
    for h in (2, 3):
        context = r['packet']['root_context']; segments = []
        for _ in range(h):
            context = array_rollout(context, np.zeros((5, 2)))
            segments.append(context)
        compiled['raw'][str(h)].update(nominal_segments=deepcopy(segments), jacobian_forward_segments=deepcopy(segments))
    arms = {arm: m.search(r, arm, compiled, array_rollout, {}, lambda e: None) for arm in m.ARMS}
    journal = tmp_path/'root.events.jsonl'
    journal.write_text(''.join(m.io.canonical(dict(kind=k, completed=True))+'\n'
        for k in ['svd']*8+['jacobian']*2))
    row = dict(root_sha256=r['packet']['root_sha256'], compilation=compiled, arms=arms, status='completed',
        neural_receipt=dict(events=[]), journal_path=str(journal), journal_sha256=m.io.sha(journal.read_bytes()))
    p = dict(records=[r], normalization=NORMALIZATION)
    report = dict(schema=m.NAME+'.report', protocol_id=m.digest(p), contract=m.contract(), roots=[row],
        status='cuda_completed', errors=[], model_unchanged=True, new_physical_steps=0,
        P1_admission=False, P2_admission=False, P3_admission=False, P4_admission=False)
    m.write(tmp_path/'execution_protocol.json', dict(prepared='fixture', prepare_sha256='fixture'))
    m.write(tmp_path/'report.json', report)
    monkeypatch.setattr(m, 'validate_prepared', lambda *a: p)
    monkeypatch.setattr(m, 'accounting_ok', lambda *a: True)
    assert len(m.selected_actions(tmp_path)) == 4
    bad = report['roots'][0]['arms'][m.ARMS[0]]
    # All repeated receipts get the same fake tokens. Equality alone would pass.
    for c in bad['candidates']: c['context']['action_history'] = np.zeros((2, 10)).tolist()
    bad['winner']['context']['action_history'] = np.zeros((2, 10)).tolist()
    bad['whole_replay_context']['action_history'] = np.zeros((2, 10)).tolist()
    bad['segmented_replay_context']['action_history'] = np.zeros((2, 10)).tolist()
    for c in bad['segments']: c['action_history'] = np.zeros((2, 10)).tolist()
    # Existing write is exclusive; only rewrite the temporary test fixture.
    (tmp_path/'report.json').write_text(m.io.canonical(report))
    with pytest.raises(ValueError, match='history'): m.selected_actions(tmp_path)
