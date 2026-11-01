"""RAM state/color/call bookkeeping only, no pymunk import/physics/render."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
from experiments.spectral_reachability import agent_p5_renderer_factorial_20261028_v1 as m


@pytest.fixture(scope='module')
def actual():
    root = m.ROOT/'results'
    return m.prepare(root/'agent_p5_terminal_goal_attribution_20261028_v5_prepare_MAIN01.json',
                     root/'agent_p5_terminal_goal_attribution_20261028_v5_cuda_MAIN01/report.json',
                     root/'agent_p5_terminal_goal_attribution_20261028_v5_consume_MAIN01.json')


def test_actual_source_and_all_twenty_fixed_cells(actual):
    assert actual['source_contract']['target_color_supported'] is True
    assert actual['source_contract']['agent_hide_supported'] is False
    assert len(actual['design']) == 20
    assert [(x['state_id'], x['agent'], x['target']) for x in actual['design']] == [
        (s, a, t) for s in m.STATES for a, t in m.CONDITIONS]
    assert actual['protocol']['terminal_cells'] == 16 and actual['protocol']['goal_cells'] == 4
    assert actual['physical_release'] is False
    m.verify(actual)


def test_actual_blocked_runner_no_native_import(actual, monkeypatch, tmp_path):
    def forbidden(*args, **kwargs): raise AssertionError('no live provider or native import allowed')
    monkeypatch.setattr(m.importlib, 'import_module', forbidden)
    r = m.run(actual, True)
    assert len(r['rows']) == 20 and all(x['status'] == 'not_run' for x in r['rows'])
    assert r['attempted'] == r['returned'] == m.ZERO_CALLS and r['assets'] == []
    path = tmp_path/'blocked.json'; m.write_new(path, r)
    assert m.consume(actual, path, m.sha(path))['rendered'] == 0
    with pytest.raises(ValueError, match='authorization'): m.run(actual)


@pytest.mark.parametrize('mutation', ['ready', 'calls', 'rows', 'asset', 'science'])
def test_cannot_promote_rejection_to_evidence(actual, tmp_path, mutation):
    r = m.blocked_result(actual)
    if mutation == 'ready': r['status'] = 'all20_rendered'
    elif mutation == 'calls': r['attempted']['native_render'] = 20
    elif mutation == 'rows': r['rows'].pop()
    elif mutation == 'asset': r['assets'] = ['old_rgb.npy']
    else: r['scientific_admission'] = True
    path = tmp_path/'bad.json'; m.write_new(path, r)
    with pytest.raises(ValueError, match='promote'): m.consume(actual, path, m.sha(path))


def test_actual_cli_prepare_verify_and_no_overwrite(actual, tmp_path):
    path = tmp_path/'plan.json'
    authority = actual['authority']
    assert m.main(['prepare', '--manifest', str(path), '--authority',
                   authority['manifest'], authority['report'], authority['receipt']]) == 0
    assert m.main(['verify', '--manifest', str(path), '--manifest-sha256', m.sha(path)]) == 0
    assert m.read_json(path) == actual
    with pytest.raises(FileExistsError):
        m.main(['prepare', '--manifest', str(path), '--authority', *authority.values()])


class FakeEnv:
    """Record-only RAM seam, not an environment or image-based physics model."""
    def __init__(self):
        self.goal_color = ('original',)
        self.render_action = False; self.window = self.clock = None
        self.space = SimpleNamespace(shapes=[object()], bodies=[object()])
        self.agent = object(); self.block = object()
        self.state = dict(position=[1., 2.], velocity=[3., 4.], rng={'counter': 0})
        self.mutation = None; self.calls = []
    def _render_frame(self, mode):
        self.calls.append((mode, self.goal_color))
        if self.mutation == 'raise': raise RuntimeError('RAM renderer exception')
        if self.mutation == 'state': self.state['rng']['counter'] += 1
        if self.mutation == 'topology': self.agent = object()
        return np.full((224, 224, 3), 1 if self.mutation == 'rgb' else 0, np.uint8)


class FakeGuard:
    def __init__(self, env):
        self.attempted = self.returned = dict(native_render=1, debug_draw=1, goal_body=1, body_construct=1)
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def check(self): return None


def pair(env):
    return m._target_pair(env, deepcopy(env.state), np.zeros((224, 224, 3), np.uint8),
                          lambda e: deepcopy(e.state), lambda name: (name,), FakeGuard)


def test_supported_target_pair_original_restore_and_all_returns():
    env = FakeEnv(); original = env.goal_color
    result = pair(env)
    assert [x['target'] for x in result] == ['baseline', 'visible', 'white']
    assert env.goal_color is original and len(env.calls) == 3
    assert all(x['before'] == x['after'] == env.state for x in result)
    assert env.calls[1][1] == ('LightGreen',) and env.calls[2][1] == ('White',)


@pytest.mark.parametrize('fault', ['raise', 'state', 'topology', 'rgb'])
def test_render_failures_restore_color_do_not_claim_invariance(fault):
    env = FakeEnv(); original = env.goal_color; env.mutation = fault
    with pytest.raises((ValueError, RuntimeError)): pair(env)
    assert env.goal_color is original
    assert len(env.calls) == 1


def test_reject_nonlive_source_before_draw():
    env = FakeEnv()
    with pytest.raises(ValueError, match='already-live'):
        m._target_pair(env, {'fake': True}, np.zeros((224, 224, 3), np.uint8),
                       lambda e: deepcopy(e.state), tuple, FakeGuard)
    assert env.calls == []


def test_public_target_kernel_no_fake_native_provenance():
    env = FakeEnv()
    with pytest.raises(ValueError, match='original native renderer'):
        m.render_live_target_pair(env, env.state, np.zeros((224, 224, 3), np.uint8), True)


def test_native_guard_forbidden_before_body_and_profile_cleanup():
    class Dummy:
        def reset(self): raise AssertionError('must not enter forbidden body')
    dummy = Dummy()
    guard = object.__new__(m.RenderGuard)
    guard.codes = {Dummy.reset.__code__: 'reset'}
    from collections import Counter
    guard.attempted = Counter(); guard.returned = Counter(); guard.pending = {}; guard.errors = []
    with pytest.raises(ValueError, match='forbidden native entry'):
        with guard: dummy.reset()
    assert m.sys.getprofile() is None
    assert guard.attempted['reset'] == 1 and guard.returned['reset'] == 0


def test_binding_mutation(actual):
    v = deepcopy(actual); v['bindings'][str(m.ROOT/m.ENV)] = '0'*64
    with pytest.raises(ValueError, match='binding drift'): m.verify(v)
    v = deepcopy(actual); v['source_contract']['agent_hide_supported'] = True
    with pytest.raises(ValueError, match='recomputation'): m.verify(v)


class RAMBody:
    def __init__(self): self.created = True


class RAMSpace:
    def __init__(self):
        self.shapes = [object()]; self.bodies = [object()]; self.draws = 0; self.steps = 0
    def debug_draw(self): self.draws += 1
    def step(self): self.steps += 1


class RAMNative(FakeEnv):
    """Distinct Python entry codes exercise real profiler, never physics."""
    def __init__(self):
        super().__init__(); self.space = RAMSpace(); self.block = RAMBody()
    def _render_frame(self, mode):
        self._get_goal_pose_body()
        if self.mutation == 'step': self.space.step()
        self.space.debug_draw()
        return np.zeros((224, 224, 3), np.uint8)
    def _get_goal_pose_body(self): return RAMBody()
    def reset(self): self.calls.append('reset')
    def _set_state(self): self.calls.append('set_state')
    def step(self): self.calls.append('step')
    def _setup(self): self.calls.append('setup')
    def seed(self): self.calls.append('seed')
    def close(self): self.calls.append('close')


def test_real_profiler_ram_nested_call_return_and_no_leak():
    env = RAMNative()
    rows = m._target_pair(env, deepcopy(env.state), np.zeros((224, 224, 3), np.uint8),
                          lambda e: deepcopy(e.state), lambda name: (name,), m.RenderGuard)
    assert env.space.draws == 3 and env.space.steps == 0
    assert m.sys.getprofile() is None
    assert all(r['attempted'] == r['returned'] == dict(native_render=1, goal_body=1, body_construct=1, debug_draw=1)
               for r in rows)


def test_real_profiler_ram_space_step_blocked_before_body():
    env = RAMNative(); env.mutation = 'step'; original = env.goal_color
    with pytest.raises(ValueError, match='forbidden native entry: space_step'):
        m._target_pair(env, deepcopy(env.state), np.zeros((224, 224, 3), np.uint8),
                       lambda e: deepcopy(e.state), lambda name: (name,), m.RenderGuard)
    assert env.space.steps == env.space.draws == 0
    assert env.goal_color is original and m.sys.getprofile() is None
