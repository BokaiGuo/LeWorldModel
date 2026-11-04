import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import pc_ap_durable_development_runner_20261102_codexexclusive as r
from experiments.spectral_reachability import pc_ap_durable_context_adapter_20261102_codexexclusive as d


class Env:
    action_space = SimpleNamespace(low=np.array([-1., -1.]), high=np.array([1., 1.]))

    def __init__(self, stop=0):
        self.count, self.stop = 0, stop
        self._goal = np.full((2, 2, 3), 8, np.uint8)

    @property
    def unwrapped(self):
        return self

    def step(self, action):
        self.count += 1
        return None, 0., self.count == self.stop, False, {}

    def render(self):
        return np.full((2, 2, 3), self.count, np.uint8)


def snapshot(env):
    env = env.unwrapped
    return {"physics_integration_state": [env.count], "integration_state_spec": 1,
            "model_binary_sha256": "fixture", "target_qpos": [1., 2.],
            "target_marker_pos": [0., 0., 0.],
            "rendered_goal_rgb": d.runtime.pack_rgb_observation(env._goal)}


def context(env):
    frames = [np.full((2, 2, 3), i, np.uint8) for i in range(3)]
    root = np.full((2, 2, 3), 3, np.uint8)
    return {"frames": frames, "root_frame": root, "goal": env._goal.copy(),
            "history_action_blocks": [np.full(10, i, np.float32) for i in range(3)],
            "execution_snapshot": snapshot(env), "root_rgb_sha256": d.runtime.rgb_digest(root),
            "goal_rgb_sha256": d.runtime.rgb_digest(env._goal),
            "history_rgb_sha256": [d.runtime.rgb_digest(x) for x in frames]}


@pytest.fixture
def harness(tmp_path, monkeypatch):
    monkeypatch.setattr(d.runtime, "capture_execution_snapshot", snapshot)
    env = Env()
    journal = d.Journal(tmp_path / "journal.jsonl")
    yield env, journal, tmp_path / "journal.jsonl"
    journal.close()


def payloads(path):
    return [json.loads(line)["payload"] for line in path.read_text().splitlines()]


def test_full_chain_feedback_and_evaluator_isolation(harness):
    env, journal, path = harness
    views = []
    def planner(view, seed, arm, index):
        views.append(view)
        assert set(view) == {"frames", "root_frame", "goal", "history_action_blocks",
                             "root_rgb_sha256", "goal_rgb_sha256"}
        return np.full((5, 2), .1 * (index + 1), np.float32), {
            "prediction_batches": 3, "prediction_samples": 34,
            "search_seed": seed + 8100 + index}
    result = r.run_arm(env, context(env), planner, journal, seed=20263160,
                       arm="ap_observed_target", distance=lambda env: env.count / 10)
    assert result["status"] == "complete" and env.count == 10
    np.testing.assert_array_equal(views[1]["root_frame"], np.full((2, 2, 3), 5, np.uint8))
    np.testing.assert_array_equal(views[1]["history_action_blocks"][-1], np.full(10, .1, np.float32))
    saved = payloads(path)
    metrics = [x for x in saved if x["kind"] == "evaluator_distance"]
    assert len(metrics) == 10 and [x["qpos_distance"] for x in metrics] == [i / 10 for i in range(1, 11)]
    assert len([x for x in saved if x["kind"] == "context_committed"]) == 2
    assert saved[-1]["kind"] == "arm_completed"


def test_budget_mismatch_blocks_before_physical_steps(harness):
    env, journal, path = harness
    result = r.run_arm(env, context(env), lambda *args: (np.zeros((5, 2), np.float32), {}),
                       journal, seed=20263160, arm="final_goal", distance=lambda _: 1.)
    assert result["status"] == "error" and env.count == 0
    assert payloads(path)[-1]["kind"] == "arm_failed"


def test_termination_does_not_call_second_planner(harness):
    env, journal, path = harness
    env.stop = 3
    calls = []
    def planner(view, seed, arm, index):
        calls.append(index)
        return np.zeros((5, 2), np.float32), {
            "prediction_batches": 3, "prediction_samples": 34, "search_seed": seed + 8100 + index}
    result = r.run_arm(env, context(env), planner, journal, seed=20263160,
                       arm="final_goal", distance=lambda _: 1.)
    assert result["status"] == "native_stopped" and calls == [0] and env.count == 3
    assert len([x for x in payloads(path) if x["kind"] == "evaluator_distance"]) == 3


def test_stale_snapshot_stops_before_first_step(harness):
    env, journal, path = harness
    ctx = context(env)
    ctx["execution_snapshot"]["target_marker_pos"][0] = 1.
    planner = lambda view, seed, arm, index: (np.zeros((5, 2), np.float32),
        {"prediction_batches": 3, "prediction_samples": 34, "search_seed": seed + 8100 + index})
    result = r.run_arm(env, ctx, planner, journal, seed=20263161,
                       arm="final_goal", distance=lambda _: 1.)
    assert result["status"] == "error" and env.count == 0
    assert payloads(path)[-1]["kind"] == "arm_failed"


def test_evaluator_persistence_failure_does_not_repeat_step(harness, monkeypatch):
    env, journal, path = harness
    emit = journal.emit
    def fail_metrics(payload):
        if payload["kind"] == "evaluator_distance":
            raise OSError("fixture disk failure")
        emit(payload)
    monkeypatch.setattr(journal, "emit", fail_metrics)
    calls = []
    def planner(view, seed, arm, index):
        calls.append(index)
        return np.zeros((5, 2), np.float32), {
            "prediction_batches": 3, "prediction_samples": 34, "search_seed": seed + 8100 + index}
    result = r.run_arm(env, context(env), planner, journal, seed=20263160,
                       arm="final_goal", distance=lambda _: 1.)
    assert result["status"] == "step_failed" and env.count == 1 and calls == [0]
    saved = payloads(path)
    assert not any(x["kind"] == "context_committed" for x in saved)
    assert any(x.get("event", {}).get("kind") == "step_intent" for x in saved)
