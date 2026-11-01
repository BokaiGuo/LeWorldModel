"""Non-neural fixtures for physical feedback bindings and accounting."""
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import p1_six_prefix_feedback_20261101_codex as feedback


def fake_runner():
    def normalized(task, raw):
        return np.asarray(raw, np.float32).reshape(2, 10).tolist()
    return SimpleNamespace(rgb=SimpleNamespace(moving=SimpleNamespace(normalized=normalized)),
                           p3=SimpleNamespace(core=SimpleNamespace(validate_task=lambda task: None)))


def test_actual_history_and_previous_plan_binding():
    base = dict(root_context={}, binding=dict(task_id="old"))
    raw = np.arange(20, dtype=np.float32).reshape(10, 2) / 20
    previous = np.zeros((3, 5, 2), np.float32)
    previous[0] = raw[-5:]
    latent = np.arange(576, dtype=np.float32).reshape(3, 192)
    task = feedback.make_task(base, latent, raw, "a" * 64, previous, 10, fake_runner())
    assert task["root_context"]["latent_history"] == latent.tolist()
    assert task["root_context"]["action_history"] == raw.reshape(2, 10).tolist()
    assert base["root_context"] == {}
    previous[0, 0, 0] += .1
    with pytest.raises(ValueError, match="actual feedback inputs"):
        feedback.make_task(base, latent, raw, "a" * 64, previous, 10, fake_runner())


def test_bound_plan_rejects_stale_episode_packet_and_rgb():
    plan = np.zeros((3, 5, 2), np.float32)
    plan[0] = .2
    bound = feedback.carry.BoundPlan.seal("native", 5, plan)
    ok = bound.bind_feedback(episode="native", boundary=10, issued_packet=plan[0], rgb_sha256="f"*64)
    assert ok["tail"].shape == (2, 5, 2)
    with pytest.raises(ValueError):
        bound.bind_feedback(episode="zero", boundary=10, issued_packet=plan[0], rgb_sha256="f"*64)
    with pytest.raises(ValueError):
        bound.bind_feedback(episode="native", boundary=10, issued_packet=np.zeros((5, 2), np.float32),
                            rgb_sha256="f"*64)


def test_budget_and_roster_fixed():
    assert feedback.PREFIXES == ("structured", "native", "random", "immediate_progress", "zero",
                                 "same_bank_greedy")
    for boundary, horizon, forward, replay in ((10, 2, 260, 81), (15, 1, 66, 17)):
        cost = feedback.h4.h4.node_cost("structured", boundary)
        assert feedback.h4.boundary_config(boundary)["planning_horizon_packets"] == horizon
        assert cost["predict_calls"] == cost["action_encode_calls"] == forward
        assert cost["backward_calls"] == cost["vjp_rows"] == 240
        assert cost["replay_calls"] == cost["selection_evaluations"] == replay
        assert cost["encode_calls"] == 0


def test_checkpoint_mismatch_rejected():
    expected = dict(step=10, state=[0.] * 7, pixel_sha256="a"*64)
    feedback.check_checkpoint(dict(expected), expected)
    with pytest.raises(ValueError, match="replayed checkpoint identity"):
        feedback.check_checkpoint(dict(expected, pixel_sha256="b"*64), expected)


def test_packet_stop_records_actual_step_and_causes(monkeypatch):
    class Env:
        unwrapped = None

        def __init__(self):
            self.unwrapped = self
            self.count = 0

        def step(self, command):
            self.count += 1
            self.latest_action = command
            return ({"state": np.array([1., 2., 3., 4., 0., 0., 0.])}, 0,
                    self.count == 2, False, {"goal_state": np.zeros(7)})

    monkeypatch.setattr(feedback.common.metrics.qualification, "native",
                        lambda goal, state: {"initial_success": True})
    row = {"executed_raw": [[0., 0.]] * 10}
    event = feedback.packet(Env(), np.zeros((5, 2), np.float32), row, np.zeros(7))
    assert event == dict(step=12, terminated=True, truncated=False, outside=False,
                         native_success=True, status="native_stopped")
    assert len(row["executed_raw"]) == 12


def test_driver_guard_invalidates_before_dispatch():
    class Driver:
        failed = False
        closed = False
        emitted = []
        dispatched = False

        def _emit(self, event, payload):
            self.emitted.append(event)

        def dispatch(self):
            self.dispatched = True

    driver = Driver()
    def limit():
        raise ValueError("limit")
    feedback.guard_driver(driver, limit)
    with pytest.raises(ValueError, match="limit"):
        driver.dispatch()
    assert driver.failed and driver.emitted == ["invalid"] and not driver.dispatched
