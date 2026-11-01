import copy
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p3_c3_k17_native_episode_20261029_v1 as m


@pytest.fixture
def fixture(monkeypatch, tmp_path):
    warm = np.zeros((10, 2), dtype=np.float32)
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    obs = {'state': np.zeros(7)}
    body = SimpleNamespace(eval_state=lambda *args: (False,))
    env = SimpleNamespace(unwrapped=body)
    root = {'packets': {'off': {'raw_action_history': warm, 'rgb_history': np.array([img]*3),
                                 'goal_rgb': img}}, 'warm_trace': [], 'goal': np.zeros(7),
            'slot': {'seed': 1}, 'root_sha256': 'root'}
    row = {'prefix51': {'frames': []}}
    events, commands = [], []
    store = SimpleNamespace(path=tmp_path, emit=lambda kind, **kw: events.append((kind, kw)))
    monkeypatch.setattr(m.h3.moving, 'warmup_commands', lambda: warm)
    monkeypatch.setattr(m.h3.moving, 'exact', np.array_equal)
    monkeypatch.setattr(m.h3, 'place', lambda *args: (obs, np.zeros(7)))
    monkeypatch.setattr(m.h3.source, 'vector', np.asarray)
    monkeypatch.setattr(m.h3.source, 'root_hash', lambda *args: 'root')
    monkeypatch.setattr(m.h3, 'bind_goal', lambda *args: None)
    monkeypatch.setattr(m.h3, 'invariant', lambda *args: {'fixed': True})
    monkeypatch.setattr(m.h3.paired, 'check_return', lambda *args: None)
    def observe(env, obs, phase, step, row, meter, store, trace):
        row['prefix51']['frames'].extend([{'phase': phase, 'step': step}] * 3)
        return img
    monkeypatch.setattr(m.h3, 'observe', observe)
    def dispatch(env, command, phase, step, row, meter, store, goal):
        if phase == 'plan': commands.append((step, command.copy()))
        return obs, {}, False, False, 0., command.copy()
    monkeypatch.setattr(m.h3, 'dispatch', dispatch)
    monkeypatch.setattr(m.gate, 'capture_check', lambda *args: {'physical_dispatch_ready': False,
        'history': {'accepted': True}, 'action': {'actual_replay_tokens_exact': True}})
    item = {'first5_raw_f32_hex': np.zeros((5, 2), dtype='<f4').tobytes().hex()}
    return env, root, item, row, store, events, commands


def test_guard_precedes_five_literal_dispatches(fixture):
    env, root, item, row, store, events, commands = fixture
    result = m.episode(env, root, {}, item, {}, {}, row, None, store)
    assert result == {'status': 'horizon_completed', 'returned': 5, 'missing_steps': []}
    assert events[0][0] == 'k17_guard' and events[0][1]['accepted']
    assert [x[0] for x in commands] == [1, 2, 3, 4, 5]


def test_guard_rejection_never_dispatches(fixture, monkeypatch):
    env, root, item, row, store, events, commands = fixture
    def reject(*args): raise ValueError('tampered replay')
    monkeypatch.setattr(m.gate, 'capture_check', reject)
    with pytest.raises(ValueError, match='tampered'):
        m.episode(env, root, {}, item, {}, {}, row, None, store)
    assert not commands and events[-1][1]['accepted'] is False


def test_early_stop_does_not_pad(fixture, monkeypatch):
    env, root, item, row, store, events, commands = fixture
    original = m.h3.dispatch
    def stop(env, command, phase, step, row, meter, store, goal):
        out = list(original(env, command, phase, step, row, meter, store, goal))
        if phase == 'plan' and step == 2: out[2] = True
        return tuple(out)
    monkeypatch.setattr(m.h3, 'dispatch', stop)
    result = m.episode(env, root, {}, item, {}, {}, row, None, store)
    assert result == {'status': 'native_stopped', 'returned': 2, 'missing_steps': [3, 4, 5]}
    assert len(commands) == 2
