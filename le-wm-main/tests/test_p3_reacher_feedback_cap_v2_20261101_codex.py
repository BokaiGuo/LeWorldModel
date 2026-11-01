from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from stable_worldmodel.envs.dmcontrol.custom_tasks.reacher import ReacherQPosMatchTask
from experiments.spectral_reachability import p3_reacher_feedback_cap_v2_20261101_codex as v2
from experiments.spectral_reachability import p3_reacher_feedback_cap_20261101_codex as v1
from tests.test_p3_reacher_feedback_cap_20261101_codex import root


def make_env(qpos, target, threshold=0.05):
    task = object.__new__(ReacherQPosMatchTask)
    task.target_qpos = np.asarray(target, dtype=np.float64)
    task.qpos_threshold = threshold
    physics = SimpleNamespace(data=SimpleNamespace(qpos=np.asarray(qpos, dtype=np.float64)))
    raw = SimpleNamespace(_task_name="qpos_match", env=SimpleNamespace(task=task, physics=physics))
    return SimpleNamespace(unwrapped=raw)


def test_native_geometry_is_recomputable_and_strict():
    env = make_env([0.01, -0.02], [0.0, 0.0])
    g = v2.physical_geometry(env)
    delta = np.array(g["qpos"]) - np.array(g["target_qpos"])
    np.testing.assert_array_equal(delta, g["qpos_difference"])
    assert g["qpos_l2"] == pytest.approx(np.linalg.norm(delta))
    assert g["native_success"] and g["native_termination"] == 0.0
    g = v2.physical_geometry(make_env([0.05, 0.0], [0.0, 0.0]))
    assert not g["native_success"] and g["native_termination"] is None
    assert not v2.physical_geometry(make_env([2 * np.pi, 0.0], [0.0, 0.0]))["native_success"]


def test_real_registered_hard_task_geometry_without_render_or_model():
    import gymnasium as gym
    import stable_worldmodel  # registers swm environments
    from experiments.run_live_reacher_planning import install_missing_field_compat
    from dm_control.suite.reacher import Reacher

    install_missing_field_compat()
    env = gym.make("swm/ReacherDMControl-v0")
    try:
        env.reset(seed=20261101)
        raw = env.unwrapped
        assert raw._task_name == "hard" and type(raw.env.task) is Reacher
        # The real goal wrapper sets this attribute during its render transaction.
        qpos = np.asarray(raw.env.physics.data.qpos).copy()
        raw.env.task.target_qpos = qpos + np.array([.2, -.3])
        before = v2.physical_geometry(env)
        np.testing.assert_allclose(before["qpos_difference"], [-.2, .3])
        assert before["native_success"] is None and before["native_qpos_threshold"] is None
        _, _, _, _, _ = env.step(np.zeros(2, np.float32))
        after = v2.physical_geometry(env)
        np.testing.assert_allclose(np.asarray(after["qpos"]) - after["target_qpos"],
                                   after["qpos_difference"])
        assert after["native_success"] is None
    finally:
        env.close()


def test_geometry_rejects_unknown_task_and_invalid_state():
    env = make_env([0.0], [0.0])
    env.unwrapped._task_name = "hard"
    with pytest.raises(ValueError, match="unrecognized"):
        v2.physical_geometry(env)
    with pytest.raises(ValueError, match="invalid"):
        v2.physical_geometry(make_env([float("nan")], [0.0]))


def test_full_snapshot_comparison_and_contract_unchanged():
    a, b = root(), root()
    a["execution_snapshot"]["extra"] = [1]
    b["execution_snapshot"]["extra"] = [1]
    v2.audit_roots(a, b)
    b = deepcopy(b)
    b["execution_snapshot"]["extra"] = [2]
    with pytest.raises(ValueError, match="complete saved"):
        v2.audit_roots(a, b)
    assert (v2.CAPS, v2.TOTAL, v2.BLOCK, v2.SEED) == (v1.CAPS, v1.TOTAL, v1.BLOCK, v1.SEED)


def test_v2_budget_entry_fails_closed():
    caps = dict.fromkeys(v2.CAPS, 1)
    meter = v2.Budget(caps)
    meter.charge("predict", 1, "planning")
    with pytest.raises(ValueError, match="before dispatch"):
        meter.charge("predict", 1, "planning")
    assert meter.attempted["predict_calls"] == 1
    assert meter.events[-1]["status"] == "blocked"


def test_returned_action_is_retained_if_geometry_fails(tmp_path, monkeypatch):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Identity()
            self.action_encoder = torch.nn.Identity()

        def predict(self, states, actions):
            return states

    class Env:
        action_space = SimpleNamespace(low=np.array([-1., -1.]), high=np.array([1., 1.]))

        def step(self, action):
            return None, 0., False, False, {}

    calls = []
    def geometry(env):
        calls.append(1)
        if len(calls) == 2:
            raise ValueError("geometry failed")
        return {"root": True}

    fake = SimpleNamespace(
        _prepare=lambda *args: (torch.zeros((1, 3, 2)), torch.zeros((1, 1, 2)), torch.zeros((1, 3, 10))),
        _latent_context=lambda *args: torch.zeros((1, 3, 2)),
        _goal_latent=lambda *args: torch.zeros(2))
    monkeypatch.setattr(v2, "physical_geometry", geometry)
    monkeypatch.setattr(v2.planned, "plan_actions", lambda *a, **k:
                        (np.zeros((15, 2), np.float32), {"fixture": True}))
    def step_once(**kwargs):
        kwargs["env"].step(kwargs["physical_actions"][0])
    monkeypatch.setattr(v2.adapter, "run_chunk", step_once)
    row = v2.run_arm("fixed", Env(), root(), Model(), None, np.zeros(10, np.float32),
                     np.ones(10, np.float32), fake, tmp_path, device="cpu")
    assert row["status"] == "failed"
    assert row["steps"] == row["physical_step_attempts"] == len(row["actions"]) == 1
    assert "geometry" not in row["actions"][0]
    assert row["chunks"][0]["status"] == "started"
    assert row["chunks"][0]["actual_returned"] == 1
    assert row["chunks"][0]["candidate_raw"][0] == row["actions"][0]["action"]
    assert row["error"]["message"] == "geometry failed"
    assert (tmp_path / "fixed.json").exists()
