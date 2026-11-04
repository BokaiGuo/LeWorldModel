from contextlib import nullcontext
import hashlib

import numpy as np
import pytest

from experiments.spectral_reachability import p2_cp5_native_session_20261102_codexexclusive as mod


class Store:
    def __init__(self):
        self.events = []
        self.prefix_events = {0: []}

    def emit(self, kind, **fields):
        self.events.append((kind, fields))


class FakeNative:
    class moving:
        @staticmethod
        def exact(a, b):
            return np.array_equal(a, b)

        @staticmethod
        def warmup_commands():
            return np.zeros((10, 2), dtype=np.float32)

    class source:
        @staticmethod
        def vector(x):
            return np.asarray(x)

        @staticmethod
        def root_hash(state, goal):
            return hashlib.sha256(state.tobytes() + goal.tobytes()).hexdigest()

    class paired:
        @staticmethod
        def check_return(env, command, actual, info, goal, reward, terminated):
            assert np.array_equal(command, actual)
            return False

    class native:
        class allmode:
            class NativeStop(RuntimeError):
                pass

        @staticmethod
        def metrics(goal, state):
            return {"native_success": False, "native7_reward": 0., "native": {}}

    def __init__(self, stop=None):
        self.stop = stop

    @staticmethod
    def place(env, plan, slot, meter, store):
        return {"state": np.array([0.])}, np.array([1.])

    @staticmethod
    def bind_goal(*args):
        pass

    @staticmethod
    def invariant(body):
        return {"goal": "fixed", "snapshot": {"step": body.step}}

    @staticmethod
    def observe(env, obs, phase, step, row, meter, store, trace):
        return np.full((224, 224, 3), step if phase == "plan" else 0, np.uint8)

    def dispatch(self, env, command, phase, step, row, meter, store, goal):
        env.step += 1
        store.emit("dispatch", phase=phase, step=step)
        return ({"state": np.array([0.])}, {"goal": np.zeros((224, 224, 3), np.uint8)},
                False, (phase, step) == self.stop, 0., command.copy())


class Body:
    step = 0

    @staticmethod
    def eval_state(goal, state):
        return False, None


class Env:
    def __init__(self):
        self.unwrapped = Body()
        self.step = 0


def make_session(stop=None, root_override=None):
    fake, store, env = FakeNative(stop), Store(), Env()
    root_hash = fake.source.root_hash(np.array([0.]), np.array([1.]))
    root = {"slot": 0, "goal": [1.], "root_sha256": root_override or root_hash,
            "warm_trace": [], "packets": {"off": {
                "raw_action_history": np.zeros((10, 2), np.float32),
                "rgb_history": np.zeros((3, 224, 224, 3), np.uint8),
                "goal_rgb": np.zeros((224, 224, 3), np.uint8)}}}
    plan = {"task_id": "task", "commands": np.zeros((25, 2), np.float32).tolist()}
    row = {"index": 0}
    meter = type("Meter", (), {"where": {"branch": 0}})()
    session = mod.NativeSession(fake, env, root, {}, plan, row, meter, store,
                                {"prefix_reference": []}, nullcontext())
    return session, store, env


def test_warmup_shared_prefix_and_cp5_are_recorded():
    session, store, env = make_session()
    root = session.reconstruct()
    assert root["root_sha256"] == session.root["root_sha256"]
    assert root["rgb"].shape == (2, 224, 224, 3)
    for _ in range(5):
        result = session.step(np.zeros(2, np.float32))
    assert result["rgb"][0, 0, 0] == 20
    assert env.step == 30
    kinds = [kind for kind, _ in store.events]
    assert kinds.count("segment_step_returned") == 30
    assert "root_guard_accepted" in kinds and "shared_prefix_guard_accepted" in kinds
    assert "cp5_capture" in kinds


def test_root_mismatch_stops_before_shared_prefix():
    session, store, env = make_session(root_override="f" * 64)
    with pytest.raises(ValueError, match="root"):
        session.reconstruct()
    assert env.step == 10
    assert not any(k == "shared_prefix_guard_accepted" for k, _ in store.events)
    assert any(k == "reconstruction_failed" for k, _ in store.events)


def test_native_stop_recorded_without_padding():
    session, store, env = make_session(stop=("plan", 18))
    session.reconstruct()
    session.step(np.zeros(2, np.float32))
    session.step(np.zeros(2, np.float32))
    with pytest.raises(RuntimeError, match="stop"):
        session.step(np.zeros(2, np.float32))
    assert env.step == 28
    assert any(k == "segment_native_stop" for k, _ in store.events)
    with pytest.raises(ValueError):
        session.step(np.zeros(2, np.float32))


def test_changed_shared_prefix_blocks_candidate():
    session, store, env = make_session()
    session.prepared["prefix_reference"] = [{"expected": "original event"}]
    with pytest.raises(ValueError, match="event identity"):
        session.reconstruct()
    assert env.step == 25
    assert not any(k == "shared_prefix_guard_accepted" for k, _ in store.events)
    with pytest.raises(ValueError, match="not ready"):
        session.step(np.zeros(2, np.float32))


def test_cp5_cannot_dispatch_an_unverified_winner_suffix():
    session, _, env = make_session()
    session.reconstruct()
    for _ in range(5):
        session.step(np.zeros(2, np.float32))
    with pytest.raises(ValueError, match="cp5 reached"):
        session.step(np.zeros(2, np.float32))
    assert env.step == 30
