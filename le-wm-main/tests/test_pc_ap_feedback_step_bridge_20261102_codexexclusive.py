import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import pc_ap_feedback_step_bridge_20261102_codexexclusive as bridge


class FakeEnv:
    action_space = SimpleNamespace(low=np.array([-1, -1]), high=np.array([1, 1]))

    def __init__(self, stop=None, error=None):
        self.calls = []
        self.stop, self.error = stop, error
        self.renders = 0

    def step(self, action):
        self.calls.append(action.copy())
        n = len(self.calls)
        if n == self.error:
            raise RuntimeError("injected")
        return None, 0.0, n == self.stop, False, {}

    def render(self):
        self.renders += 1
        return np.full((3, 4, 3), 9, np.uint8)


def run(env, actions=None, emit=None):
    root = np.zeros((3, 4, 3), np.uint8)
    goal = np.ones_like(root)
    events = []
    result = bridge.execute_block(
        env, np.zeros((5, 2), np.float32) if actions is None else actions,
        expected_root=root, goal=goal,
        expected_goal=hashlib.sha256(goal.tobytes()).hexdigest(),
        emit=events.append if emit is None else emit,
    )
    return result, events


def test_full_block_returns_actual_rendered_successor():
    env = FakeEnv()
    (receipt, handoff), events = run(env)
    assert receipt["status"] == "complete" and receipt["attempted"] == receipt["returned"] == 5
    assert env.renders == 1 and np.all(handoff["root_frame"] == 9)
    assert receipt["successor_rgb_sha256"] == handoff["root_rgb_sha256"]
    assert [e["kind"] for e in events] == (["block_open"] + ["step_intent", "step_returned"] * 5
                                             + ["render_intent", "feedback_ready"])


@pytest.mark.parametrize("stop", [1, 3, 5])
def test_termination_never_steps_or_replans_after_stop(stop):
    env = FakeEnv(stop=stop)
    (receipt, handoff), _ = run(env)
    assert receipt["status"] == "native_stopped"
    assert receipt["attempted"] == receipt["returned"] == stop
    assert len(env.calls) == stop and env.renders == 0 and handoff is None


def test_exception_counts_attempt_and_prevents_handoff():
    env = FakeEnv(error=3)
    (receipt, handoff), events = run(env)
    assert (receipt["attempted"], receipt["returned"]) == (3, 2)
    assert receipt["status"] == "step_failed" and handoff is None
    assert events[-1]["kind"] == "step_failed"


def test_invalid_action_or_sink_failure_before_dispatch():
    env = FakeEnv()
    with pytest.raises(ValueError):
        run(env, np.ones((5, 2), np.float32) * 2)
    def fail_on_intent(event):
        if event["kind"] == "step_intent":
            raise OSError("journal unavailable")
    with pytest.raises(OSError):
        run(env, emit=fail_on_intent)
    assert not env.calls


def test_second_block_is_bound_to_first_actual_feedback():
    env = FakeEnv()
    (first, handoff), _ = run(env)
    events = []
    second, _ = bridge.execute_block(
        env, np.full((5, 2), 0.25, np.float32),
        expected_root=handoff["root_frame"], goal=handoff["goal"],
        expected_goal=handoff["goal_rgb_sha256"], emit=events.append,
    )
    assert second["root_rgb_sha256"] == first["successor_rgb_sha256"]
    assert second["attempted"] == second["returned"] == 5
    assert len(env.calls) == 10
