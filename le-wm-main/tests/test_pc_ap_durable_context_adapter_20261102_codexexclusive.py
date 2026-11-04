import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import pc_ap_durable_context_adapter_20261102_codexexclusive as a


class Env:
    action_space = SimpleNamespace(low=np.array([-1., -1.]), high=np.array([1., 1.]))

    def __init__(self, stop=0):
        self.n = 0
        self.stop = stop

    def step(self, action):
        self.n += 1
        return None, 0., self.n == self.stop, False, {}

    def render(self):
        return np.full((2, 2, 3), self.n, np.uint8)


def snapshot(env):
    goal = np.full((2, 2, 3), 8, np.uint8)
    return {"physics_integration_state": [env.n], "integration_state_spec": 1,
            "model_binary_sha256": "model", "target_qpos": [1., 2.],
            "target_marker_pos": [0., 0., 0.], "rendered_goal_rgb": a.runtime.pack_rgb_observation(goal)}


def context(env):
    frames = [np.full((2, 2, 3), i, np.uint8) for i in range(3)]
    root, goal = np.full((2, 2, 3), 3, np.uint8), np.full((2, 2, 3), 8, np.uint8)
    return {"frames": frames, "root_frame": root, "goal": goal,
            "history_action_blocks": [np.full(10, i, np.float32) for i in range(3)],
            "execution_snapshot": snapshot(env), "root_rgb_sha256": a.runtime.rgb_digest(root),
            "goal_rgb_sha256": a.runtime.rgb_digest(goal),
            "history_rgb_sha256": [a.runtime.rgb_digest(x) for x in frames]}


@pytest.fixture
def subject(tmp_path, monkeypatch):
    monkeypatch.setattr(a.runtime, "capture_execution_snapshot", snapshot)
    journal = a.Journal(tmp_path / "events.jsonl")
    yield a.Adapter(journal), tmp_path / "events.jsonl"
    journal.close()


def events(path):
    return [json.loads(line)["payload"] for line in path.read_text().splitlines()]


def test_two_blocks_bind_actual_pixels_and_actions(subject):
    adapter, path = subject
    env = Env()
    original = context(env)
    actions = np.full((5, 2), .25, np.float32)
    receipt, next_context = adapter.advance(env, original, actions)
    assert receipt["returned"] == 5 and env.n == 5
    np.testing.assert_array_equal(next_context["frames"][-1], original["root_frame"])
    np.testing.assert_array_equal(next_context["history_action_blocks"][-1], actions.reshape(-1))
    np.testing.assert_array_equal(next_context["root_frame"], np.full((2, 2, 3), 5, np.uint8))
    adapter.advance(env, next_context, actions)
    saved = events(path)
    assert [x["kind"] for x in saved].count("context_committed") == 2
    for first, second in zip([x for x in saved if x["kind"] == "context_committed"],
                             [x for x in saved if x["kind"] == "context_bound"][1:]):
        assert first["context_sha256"] == second["context_sha256"]


def test_snapshot_mismatch_blocks_before_action(subject):
    adapter, path = subject
    env = Env()
    bad = context(env)
    bad["execution_snapshot"]["target_marker_pos"][0] = 2.
    with pytest.raises(RuntimeError, match="stale"):
        adapter.advance(env, bad, np.zeros((5, 2), np.float32))
    assert env.n == 0 and events(path) == []
    with pytest.raises(RuntimeError, match="poisoned"):
        adapter.advance(env, context(env), np.zeros((5, 2), np.float32))
    with pytest.raises(ValueError, match="unclaimed"):
        a.Adapter(adapter.journal)


def test_history_hash_and_goal_mismatch_fail_closed(subject):
    adapter, path = subject
    env = Env()
    bad = context(env)
    bad["frames"][0][0, 0, 0] = 99
    with pytest.raises(ValueError, match="RGB hashes"):
        adapter.advance(env, bad, np.zeros((5, 2), np.float32))
    assert env.n == 0 and not events(path)


def test_early_stop_has_no_committed_handoff(subject):
    adapter, path = subject
    env = Env(stop=3)
    receipt, output = adapter.advance(env, context(env), np.zeros((5, 2), np.float32))
    assert output is None and receipt["status"] == "native_stopped" and env.n == 3
    assert events(path)[-1]["kind"] == "block_incomplete"
    with pytest.raises(RuntimeError, match="poisoned"):
        adapter.advance(env, context(env), np.zeros((5, 2), np.float32))


def test_sink_failure_after_intent_never_retries(subject, monkeypatch):
    adapter, path = subject
    env = Env()
    emit = adapter.journal.emit
    def failing(payload):
        if payload["kind"] == "bridge_event" and payload["event"]["kind"] == "step_returned":
            raise OSError("injected disk failure")
        emit(payload)
    monkeypatch.setattr(adapter.journal, "emit", failing)
    with pytest.raises(OSError):
        adapter.advance(env, context(env), np.zeros((5, 2), np.float32))
    assert env.n == 1
    saved = events(path)
    assert saved[-1]["event"]["kind"] == "step_intent"
    with pytest.raises(RuntimeError, match="poisoned"):
        adapter.advance(env, context(env), np.zeros((5, 2), np.float32))


def test_reused_old_context_is_rejected(subject):
    adapter, _ = subject
    env = Env()
    old = context(env)
    _, new = adapter.advance(env, old, np.zeros((5, 2), np.float32))
    stale = deepcopy(new)
    stale["history_action_blocks"][0][0] = 123.
    with pytest.raises(ValueError, match="previous durable handoff"):
        adapter.advance(env, stale, np.zeros((5, 2), np.float32))
    assert env.n == 5


def test_journal_exclusive_and_hash_chain(tmp_path):
    path = tmp_path / "events.jsonl"
    journal = a.Journal(path)
    journal.emit({"kind": "one"})
    journal.emit({"kind": "two"})
    journal.close()
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    assert rows[1]["previous_sha256"] == rows[0]["event_sha256"]
    assert all(row["event_sha256"] == a._digest({k: v for k, v in row.items() if k != "event_sha256"})
               for row in rows)
    with pytest.raises(FileExistsError):
        a.Journal(path)
