"""Pure NumPy/mock contracts; no torch, CUDA, gym or simulator construction."""
from copy import deepcopy
import json
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.spectral_reachability import p4_feedback_live_cuda_20261027_v1 as m


def rgb(value, visible=True):
    a = np.full((224, 224, 3), value, np.uint8)
    if visible: a[10:15, 10:15] = [65, 105, 225]
    return a


def protocol():
    return dict(root_sha256=m.ROOT, initial_context=dict(layout='states_with_between_state_actions_v1',
        context_length=3, latent_dim=192, action_dim=10,
        latent_history=np.repeat(np.arange(1, 4, dtype=np.float32)[:, None], 192, axis=1).tolist(),
        action_history=np.zeros((2, 10), np.float32).tolist()),
        prefix_commands=np.full((5, 2), .1).tolist(), goal=np.full(192, 20.).tolist(), q=np.eye(192).tolist(),
        normalization=dict(mean=[0., 0.], std=[1., 1.], source='metadata-kept', source_sha256='a'*64,
                           training_provenance_verified=False), limits=m.caps(600, 3221225472, 4294967296))


class Backend:
    def __init__(self, p, ledger): self.p, self.ledger = p, ledger; self.bound = []; self.encodes = []
    def bind(self, request):
        self.context = deepcopy(request['context']); self.bound.append((self.ledger.arm, self.ledger.boundary, deepcopy(self.context)))
    def forward(self, commands):
        def action(): return None
        self.ledger.call('action_encoder', action, shape=(1, 3, 10))
        def predict():
            c = deepcopy(self.context); states = np.asarray(c['latent_history'], np.float32)
            last = states[-1] + np.float32(.125); last[:10] += np.asarray(commands, np.float32).reshape(10)
            c['latent_history'] = np.concatenate((states[1:], last[None])).tolist()
            norm = self.p['normalization']; token = ((np.asarray(commands, np.float32) - np.asarray(norm['mean'], np.float32)) / np.asarray(norm['std'], np.float32)).reshape(10)
            c['action_history'] = [c['action_history'][-1], token.tolist()]; return c
        return self.ledger.call('predict', predict, shape=(1, 3, 192))
    def nominal(self): return self.forward(np.zeros((5, 2)))
    def candidate(self, commands): return self.forward(commands)
    def replay(self, commands): return self.forward(commands)
    def jacobian(self):
        def ad():
            forward = self.nominal()
            for _ in range(192):
                self.ledger.call('grad_api', lambda: self.ledger.call('engine_api', lambda: None))
            j = np.zeros((192, 10), np.float32); j[:10] = np.eye(10, dtype=np.float32)
            return j, forward
        return self.ledger.call('jacobian_api', ad)
    def encode(self, frames):
        self.encodes.append((self.ledger.arm, self.ledger.boundary, [int(f[0, 0, 0]) for f in frames]))
        self.ledger.call('encoder', lambda: None, shape=(len(frames), 3, 224, 224))
        return self.ledger.call('projector', lambda: np.stack([np.full(192, f[0, 0, 0], np.float32) for f in frames]), shape=(len(frames), 192))


class MockPhysical:
    def __init__(self, ledger, store):
        self.ledger, self.store = ledger, store; self.executions = []; self.eval_count = 0
        for _ in range(32): ledger.physical('root_resets', lambda: None)
    def start(self, arm):
        self.ledger.physical('branch_resets', lambda: None)
        for _ in range(15): self.ledger.physical('warmup_steps', lambda: None)
        self.value = 3; return [rgb(i) for i in (1, 2, 3)]
    def packet(self, arm, boundary, commands, sealed):
        assert m.io.sha(self.store.path / sealed['file']) == sealed['sha256']
        selection = m.io.read(self.store.path / sealed['file'])
        assert m.io.same(selection['commands'] if boundary == 0 else selection['winner']['commands'], commands)
        self.executions.append((arm, boundary, sealed))
        for _ in range(5): self.ledger.physical('selected_steps', lambda: None)
        self.value += 1; return rgb(self.value)
    def evaluate(self): self.eval_count += 1; return dict(mock=True)


@pytest.fixture
def setup(tmp_path):
    p = protocol(); store = m.Store(tmp_path / 'output'); events = []
    ledger = m.Ledger(p['limits'], events.append); backend = Backend(p, ledger)
    return p, store, ledger, backend, events


def execute_mock(setup):
    p, store, ledger, backend, events = setup
    physical = MockPhysical(ledger, store); rows = []
    m.workflow(p, backend, physical, ledger, store, rows)
    return physical, rows


def test_full_mock_exact_budget_causal_own_frames_and_three_rounds(setup):
    physical, rows = execute_mock(setup); p, store, ledger, backend, events = setup
    assert ledger.counts() == {k: dict(attempted=v, completed=v) for k, v in m.KINDS.items()}
    assert sum(ledger.physics[k] for k in ('warmup_steps', 'selected_steps')) == 70
    assert ledger.physics['root_resets'] + ledger.physics['branch_resets'] == 34
    assert [(r['arm'], r['boundary']) for r in ledger.rounds] == [(a, b) for a in m.ARMS for b in m.BOUNDARIES]
    assert physical.eval_count == 2 and all(r['status'] == 'completed20' for r in rows)
    assert backend.encodes == [(a, b, f) for a in m.ARMS for b, f in ((5, [2, 3, 4]), (10, [3, 4, 5]), (15, [4, 5, 6]), (20, [7]))]
    for arm, boundary, c in backend.bound:
        if boundary == 0: continue
        if arm == 'actual': assert c['latent_history'][-1] == [boundary//5+3]*192
        else: assert c['latent_history'][-1] != [boundary//5+3]*192
    assert [x[:2] for x in physical.executions] == [(a, b) for a in m.ARMS for b in (0, 5, 10, 15)]
    for arm in m.ARMS:
        evaluation = m.io.read(store.path / (arm + '.result.json'))['evaluation']
        assert evaluation['consumed_for_planning'] is False
        assert np.load(store.path / evaluation['latent_file'], allow_pickle=False).shape == (192,)
    assert sum(e['event'] == 'selection_sealed' for e in events) == 6


def test_no_admission_in_caps_and_no_hidden_work():
    c = protocol()['limits']
    assert c['encoded_frames'] == 20 and c['processed_context_tokens'] == 240
    assert c['mathematical_VJP_directions'] == 1152 and c['neural']['backward_api'] == 0
    assert c['FD'] == 0 and c['training'] is c['retries'] is False


@pytest.mark.parametrize('values', [(None, 1, 1), (0, 1, 1), (float('inf'), 1, 1), (1, 2, 1), (1, True, 2), (1, 0, 2)])
def test_caps_require_explicit_finite_engineering_values(values):
    with pytest.raises(ValueError): m.caps(*values)


@pytest.mark.parametrize('kind,phase,shape', [('predict', 'terminal', (1, 3, 192)), ('encoder', 'candidate_0', (3, 3, 224, 224)),
    ('projector', 'observe', (3, 192)), ('grad_api', 'nominal', None), ('engine_api', 'jacobian', None), ('backward_api', 'jacobian', None)])
def test_illegal_work_fails_sticky(setup, kind, phase, shape):
    _, _, ledger, _, _ = setup; ledger.slot('actual', 5, phase, frames=3)
    called = []
    with pytest.raises(ValueError): ledger.call(kind, lambda: called.append(1), shape=shape)
    assert not called and ledger.failed
    with pytest.raises(ValueError): ledger.check()


def test_cpu_neural_refused(setup):
    _, _, ledger, _, _ = setup; ledger.slot('actual', 5, 'nominal')
    with pytest.raises(ValueError): ledger.call('action_encoder', lambda: None, device='cpu', shape=(1, 3, 10))


def test_failed_callback_keeps_attempt_and_no_refund(setup):
    _, _, ledger, _, _ = setup; ledger.slot('imagined', 5, 'nominal')
    def broken(): raise RuntimeError('mock exception')
    with pytest.raises(RuntimeError): ledger.call('action_encoder', broken, shape=(1, 3, 10))
    assert ledger.counts()['action_encoder'] == dict(attempted=1, completed=0)


def test_physical_throw_keeps_attempt(setup):
    _, _, ledger, _, _ = setup
    def broken(): raise RuntimeError('step failed')
    with pytest.raises(RuntimeError): ledger.physical('selected_steps', broken)
    assert ledger.physics['selected_steps'] == 1 and ledger.physics_returns['selected_steps'] == 0


@pytest.mark.parametrize('kind,cap', [('root_resets', 32), ('branch_resets', 2), ('warmup_steps', 30), ('selected_steps', 40)])
def test_physics_cap_attempts_not_refundable(setup, kind, cap):
    _, _, ledger, _, _ = setup
    for _ in range(cap): ledger.physical(kind, lambda: None)
    with pytest.raises(ValueError): ledger.physical(kind, lambda: None)
    assert ledger.physics[kind] == cap


def test_journal_failure_before_selection_prevents_dispatch(setup, monkeypatch):
    p, store, ledger, backend, _ = setup; physical = MockPhysical(ledger, store); rows = []
    original = store.write
    def broken(name, value, **kwargs):
        if name == 'imagined.step5.selection.json': raise OSError('mock fsync failure')
        return original(name, value, **kwargs)
    monkeypatch.setattr(store, 'write', broken)
    with pytest.raises(OSError): m.workflow(p, backend, physical, ledger, store, rows)
    assert [(a, b) for a, b, _ in physical.executions] == [('imagined', 0)]
    assert rows[0]['status'] == 'started'


def test_overlap_failure_stops_without_compile(setup):
    p, store, ledger, backend, _ = setup; physical = MockPhysical(ledger, store); rows = []
    original = backend.encode
    def corrupt(frames):
        value = original(frames); value[0, 0] += 1; return value
    backend.encode = corrupt
    with pytest.raises(ValueError, match='overlapping'): m.workflow(p, backend, physical, ledger, store, rows)
    assert ledger.counts()['jacobian_api']['attempted'] == 0


def test_replay_mismatch_never_dispatches_candidate(setup):
    p, store, ledger, backend, _ = setup; physical = MockPhysical(ledger, store); rows = []
    original = backend.replay
    def corrupt(commands):
        c = original(commands)
        if ledger.boundary: c['latent_history'][-1][0] += 1
        return c
    backend.replay = corrupt
    with pytest.raises(ValueError, match='replay exact'): m.workflow(p, backend, physical, ledger, store, rows)
    assert len(physical.executions) == 1


def test_six_selection_artifacts_scored_with_original_Q(setup):
    execute_mock(setup); p, store, _, _, _ = setup
    for file in store.path.glob('*.step*.selection.json'):
        r = m.io.read(file); assert len(r['candidates']) == 10
        assert r['request']['goal'] == p['goal'] and r['request']['q_sha256'] == m.io.digest(p['q'])
        assert r['winner']['index'] == min(r['candidates'], key=lambda c: (c['cost'], c['index']))['index']
        assert m.io.sha(store.path / r['jacobian_file']) == r['jacobian_sha256']
        for c in r['candidates']:
            d = np.asarray(c['context']['latent_history'])[-1] - p['goal']
            assert c['cost'] == float(np.mean((np.asarray(p['q']) @ d)**2))


def test_wall_and_VRAM_caps(setup):
    _, _, ledger, _, _ = setup; ledger.sample = lambda: (3221225473, 4294967296)
    with pytest.raises(ValueError, match='VRAM'): ledger.check()
    ledger.sample = lambda: (0, 0); ledger.clock = lambda: ledger.started + 601
    with pytest.raises(ValueError, match='wall'): ledger.check()


def test_store_exclusive_output_and_disk_floor(setup, monkeypatch):
    _, store, _, _, _ = setup
    store.write('once.json', {})
    with pytest.raises(FileExistsError): store.write('once.json', {})
    with pytest.raises(ValueError): store.write('../bad.json', {})
    monkeypatch.setattr(m.shutil, 'disk_usage', lambda p: SimpleNamespace(free=m.RESERVE-1))
    with pytest.raises(ValueError, match='disk floor'): store.write('no.json', {})


def test_store_small_cap(setup):
    _, store, _, _, _ = setup; store.cap = 1024**2+10
    with pytest.raises(ValueError, match='output cap'): store.blob('too.bin', b'0'*11)


class Env:
    def __init__(self, disappear=None): self.steps = 0; self.disappear = disappear
    def render(self): return rgb(4, self.steps != self.disappear)
    def step(self, action): self.steps += 1; return None, 0., False, False, {}


def physical_adapter(setup, disappear=None):
    p, store, ledger, _, _ = setup; physical = object.__new__(m.Physical)
    physical.p, physical.store, physical.ledger = p, store, ledger
    env = Env(disappear); physical.env = m.Meter(env, ledger); physical.env.phase = 'selected_steps'
    physical.arm = 'actual'; physical.next_boundary = 0; physical.used = set()
    return physical, env


@pytest.mark.parametrize('disappear,expected', [(0, 0), (1, 1), (3, 3), (5, 5)])
def test_actual_shared_guard_every_step_and_abort_receipt(setup, disappear, expected):
    physical, env = physical_adapter(setup, disappear)
    with pytest.raises(ValueError, match='incomplete'): physical.guarded(np.zeros((5, 2)), 'guard')
    assert env.steps == expected
    r = m.io.read(physical.store.path / 'guard.physics.json')
    assert r['guard']['abstained'] and r['completed_commands'] == expected
    assert len(r['guard']['observations']) == expected+1


def test_meter_no_hidden_collection_step(setup):
    _, _, ledger, _, _ = setup; env = m.Meter(Env(), ledger)
    with pytest.raises(ValueError, match='collector step'): env.step([0., 0.])


def test_physical_once_only_and_selection_tamper(setup):
    p, store, ledger, _, _ = setup; physical, env = physical_adapter(setup)
    name = 'prefix.json'; h = store.write(name, dict(root=p['root_sha256'], commands=p['prefix_commands']))
    sealed = dict(file=name, sha256=h)
    with pytest.raises(ValueError): physical.packet('imagined', 0, p['prefix_commands'], sealed)
    with pytest.raises(ValueError): physical.packet('actual', 0, p['prefix_commands'], dict(file=name, sha256='f'*64))
    physical.packet('actual', 0, p['prefix_commands'], sealed)
    with pytest.raises(ValueError): physical.packet('actual', 0, p['prefix_commands'], sealed)
    assert env.steps == 5


def test_missing_real_source_fails_before_prepare_directory(tmp_path, monkeypatch):
    def missing(*args): raise ValueError('missing real source')
    monkeypatch.setattr(m.response.source, 'verify', missing)
    output = tmp_path / 'no-capability'
    with pytest.raises(ValueError, match='missing real source'): m.prepare(output, protocol()['limits'])
    assert not output.exists()


def test_verify_external_SHA_first(tmp_path):
    folder = tmp_path / 'prepared'; folder.mkdir(); (folder / 'prepare.json').write_text('{}')
    with pytest.raises(ValueError, match='external prepare'): m.verify(folder, '0'*64)


@pytest.mark.parametrize('path', ['transition', 'replay'])
def test_native_normalization_metadata_is_not_kwargs(path):
    from contextlib import nullcontext
    class Tensor(np.ndarray):
        def detach(self): return self
        def cpu(self): return self
    tensor = lambda a, **kw: np.asarray(a, np.float32).view(Tensor)
    t = SimpleNamespace(tensor=tensor, float32='MOCK_float32', cat=lambda xs, dim: np.concatenate(xs, axis=dim).view(Tensor),
        isfinite=np.isfinite, no_grad=nullcontext)
    model = SimpleNamespace(action_encoder=lambda a: a, predict=lambda emb, act: emb + np.float32(.01))
    p = protocol(); native = m.Native(t, model, SimpleNamespace(type='cuda'), p)
    native.bind(dict(context=p['initial_context'])); seen = []
    def normalizer(raw, *, mean, std):
        seen.append(dict(mean=mean, std=std))
        return ((raw-np.asarray(mean, np.float32))/np.asarray(std, np.float32)).reshape(10).view(Tensor)
    native.normalize = normalizer; raw = tensor(np.zeros((5, 2)))
    result = native.export(*native.transition(raw)) if path == 'transition' else native.replay(raw.tolist())
    assert seen == [dict(mean=p['normalization']['mean'], std=p['normalization']['std'])]
    m.io.history(p['initial_context'], result, raw, p['normalization'])


def test_native_encode_real_keyword_only_helper_contract_mocked(monkeypatch):
    from contextlib import nullcontext
    import sys
    seen = []; frames = [rgb(4)]
    def frame_tensor(values, *, transform, device):
        assert values is frames and transform == 'original-transform' and device == 'MOCK_cuda'
        seen.append('frame_tensor'); return 'MOCK_pixels'
    monkeypatch.setitem(sys.modules, 'experiments.spectral_reachability.preflight_pusht_d2p_p0', SimpleNamespace(_frames_tensor=frame_tensor))
    monkeypatch.setitem(sys.modules, 'experiments.run_pusht_counterfactual', SimpleNamespace(image_transform=lambda: 'original-transform'))
    class Tensor:
        dtype = 'MOCK_float32'; device = 'MOCK_cuda'
        def __getitem__(self, index): return self
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return np.zeros((1, 192), np.float32)
    def encode(info):
        assert info == dict(pixels='MOCK_pixels'); seen.append('encode'); return dict(emb=Tensor())
    native = object.__new__(m.Native); native.t = SimpleNamespace(no_grad=nullcontext, float32='MOCK_float32')
    native.model = SimpleNamespace(encode=encode); native.device = 'MOCK_cuda'
    assert native.encode(frames).shape == (1, 192) and seen == ['frame_tensor', 'encode']


def test_hooks_nested_serial_AD_visual_slots_and_restore(setup):
    _, _, ledger, _, _ = setup
    tensor = lambda shape: SimpleNamespace(shape=shape, device=SimpleNamespace(type='cuda'))
    model = SimpleNamespace(predict=lambda *args: 'state', encode=lambda info: (
        model.encoder.forward(tensor((3, 3, 224, 224)), interpolate_pos_encoding=True),
        model.projector.forward(tensor((3, 192)))),
        action_encoder=SimpleNamespace(forward=lambda *args: 'action'),
        encoder=SimpleNamespace(forward=lambda *args, **kw: 'pixels'), projector=SimpleNamespace(forward=lambda *args: 'emb'))
    engine = lambda *args: 'gradient'
    t = SimpleNamespace(autograd=SimpleNamespace(backward=lambda *args: None, _engine_run_backward=engine,
        graph=SimpleNamespace(_engine_run_backward=engine), functional=SimpleNamespace()))
    t.autograd.grad = lambda: t.autograd._engine_run_backward()
    def jacobian(*args, **kwargs):
        model.action_encoder.forward(tensor((1, 3, 10))); model.predict(tensor((1, 3, 192)), None)
        for _ in range(192): t.autograd.grad()
        return 'J'
    t.autograd.functional.jacobian = jacobian; hooks = m.io.Hooks(); original = model.predict
    m.install_hooks(t, model, ledger, hooks)
    try:
        ledger.slot('actual', 5, 'observe', frames=3); model.encode({})
        ledger.slot('actual', 5, 'jacobian'); assert t.autograd.functional.jacobian() == 'J'
        assert ledger.counts()['grad_api']['completed'] == ledger.counts()['engine_api']['completed'] == 192
        assert ledger.counts()['encoder']['completed'] == ledger.counts()['projector']['completed'] == 1
    finally:
        errors = []; hooks.restore(errors); assert errors == []
    assert model.predict is original and t.autograd.graph._engine_run_backward is t.autograd._engine_run_backward is engine


@pytest.mark.parametrize('phase,boundary,kind,shape', [('observe', 20, 'encoder', (3, 3, 224, 224)),
    ('terminal', 15, 'encoder', (1, 3, 224, 224)), ('prefix', 5, 'action_encoder', (1, 3, 10))])
def test_terminal_cannot_become_feedback_slot(setup, phase, boundary, kind, shape):
    _, _, ledger, _, _ = setup; ledger.slot('actual', boundary, phase, frames=1 if phase == 'terminal' else 3)
    with pytest.raises(ValueError): ledger.call(kind, lambda: None, shape=shape)


def test_real_physical_start_mock_original_warmup_identity(setup):
    p, store, ledger, _, _ = setup
    from experiments.pusht_protocol_contracts import root_fingerprint
    class ResetEnv(Env):
        def reset(self, *args, **kwargs): self.steps = 0; return None, dict(goal=rgb(9))
        def render(self): return rgb(self.steps//5+1)
    env = ResetEnv(); roots = [dict(state=np.full(7, float(i)), goal=np.full(7, float(i+4))) for i in range(4)]
    p['root_sha256'] = root_fingerprint(roots[0]); p['references'] = {root_fingerprint(r): dict(root_sha256=root_fingerprint(r)) for r in roots}
    p['references'][p['root_sha256']].update(history_rgb_sha256=m.pixel_sha(np.stack([rgb(i) for i in (2, 3, 4)])), goal_rgb_sha256=m.pixel_sha(rgb(9)))
    p['root_seed'] = 202753010
    def collect(meter, **kwargs):
        for _ in range(32): meter.reset()
        return roots
    base = SimpleNamespace(collect_nonterminal_boundary_contact_roots=collect, root_fingerprint=root_fingerprint)
    physical = m.Physical(p, env, ledger, store, base); ledger.slot('imagined', 0, 'reset')
    frames = physical.start('imagined')
    assert [f[0, 0, 0] for f in frames] == [2, 3, 4]
    assert ledger.physics == dict(root_resets=32, branch_resets=1, warmup_steps=15, selected_steps=0)
    assert len(list(store.path.glob('imagined.warmup*.step*.png'))) == 15


def test_public_backward_never_dispatched(setup):
    _, _, ledger, _, _ = setup
    with pytest.raises(ValueError): ledger.reject('backward_api', 'public backward forbidden')
    assert ledger.failed and ledger.counts()['backward_api']['attempted'] == 0


def test_saved_import_has_no_run_side_effects():
    import ast
    tree = ast.parse(m.Path(m.__file__).read_text())
    top_imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert all(not any(x in ast.unparse(n) for x in ('import torch', 'import gym')) for n in top_imports)
