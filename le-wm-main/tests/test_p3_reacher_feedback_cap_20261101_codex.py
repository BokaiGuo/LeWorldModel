"""Non-neural tests for strict root checks, feedback history and shared caps."""
from copy import deepcopy
import fcntl
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.spectral_reachability import p3_reacher_feedback_cap_20261101_codex as subject


def root():
    frame = np.zeros((2, 2, 3), np.uint8)
    return dict(execution_snapshot=dict(physics_integration_state=[1., 2.], integration_state_spec=1,
                model_binary_sha256="m", target_qpos=[1.], target_marker_pos=[2.],
                rendered_goal_rgb={"sha256": "g"}),
                root_frame=frame.copy(), goal=frame.copy(), frames=[frame.copy() for _ in range(3)],
                history_action_blocks=[np.zeros(10, np.float32) for _ in range(3)],
                warmup_raw_steps=np.zeros((15, 2), np.float32), root_rgb_sha256="r",
                goal_rgb_sha256="g", history_rgb_sha256=["a", "b", "c"], warmup_actions_sha256="w")


def test_root_integration_scene_and_rgb_must_all_match():
    left, right = root(), root()
    assert subject.audit_roots(left, right)["root_rgb_sha256"] == "r"
    changed = deepcopy(right)
    changed["execution_snapshot"]["physics_integration_state"][0] += 1
    with pytest.raises(ValueError, match="integration"):
        subject.audit_roots(left, changed)
    changed = deepcopy(right)
    changed["frames"][0][0, 0, 0] = 1
    with pytest.raises(ValueError, match="root/history"):
        subject.audit_roots(left, changed)
    del right["execution_snapshot"]["model_binary_sha256"]
    with pytest.raises(ValueError, match="unavailable"):
        subject.audit_roots(left, right)


def test_actual_packet_updates_context_causally():
    ctx = root()
    before = np.full((2, 2, 3), 4, np.uint8)
    after = np.full((2, 2, 3), 5, np.uint8)
    packet = np.arange(10, dtype=np.float32).reshape(5, 2)
    updated = subject.update_context(ctx, before, after, packet)
    np.testing.assert_array_equal(updated["frames"][-1], before)
    np.testing.assert_array_equal(updated["root_frame"], after)
    np.testing.assert_array_equal(updated["history_action_blocks"][-1], packet.reshape(-1))
    np.testing.assert_array_equal(ctx["root_frame"], 0)


def test_common_budget_fails_closed_before_overrun():
    caps = dict.fromkeys(subject.CAPS, 1)
    meter = subject.Budget(caps)
    meter.charge("predict", 1, "planning")
    with pytest.raises(ValueError, match="before dispatch"):
        meter.charge("predict", 1, "planning")
    assert meter.attempted["predict_calls"] == 1
    assert meter.events[-1]["status"] == "blocked" and meter.failed
    with pytest.raises(ValueError, match="inactive"):
        meter.charge("encoder", 1, "context")


def test_feedback_runner_shifts_actual_actions_each_packet(tmp_path, monkeypatch):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Identity()
            self.action_encoder = torch.nn.Identity()
        def predict(self, states, actions):
            return states
    class Env:
        action_space = SimpleNamespace(low=np.array([-1., -1.]), high=np.array([1., 1.]))
        def __init__(self):
            self.steps = 0
        def step(self, action):
            self.steps += 1
            return None, 0., False, False, {}
        def render(self):
            return np.full((2, 2, 3), self.steps, np.uint8)
    histories = []
    def prepare(preprocessor, frames, root_frame, goal, actions, device):
        histories.append((np.asarray(actions).copy(), root_frame.copy()))
        return torch.zeros((1, 3, 2)), torch.zeros((1, 1, 2)), torch.zeros((1, 3, 10))
    fake = SimpleNamespace(_prepare=prepare,
        _latent_context=lambda model, pixels, history: torch.zeros((1, 3, 2)),
        _goal_latent=lambda model, pixels: torch.zeros(2))
    monkeypatch.setattr(subject.planned, "plan_actions", lambda *a, **k:
                        (np.full((15, 2), .25, np.float32), {"fixture": True}))
    def fake_chunk(**kwargs):
        for action in kwargs["physical_actions"][:5]:
            kwargs["env"].step(action)
        return dict(chunk_steps=5, terminated=False, truncated=False)
    monkeypatch.setattr(subject.adapter, "run_chunk", fake_chunk)
    row = subject.run_arm("fixed", Env(), root(), Model(), None, np.zeros(10, np.float32),
                          np.ones(10, np.float32), fake, tmp_path, device="cpu")
    assert row["status"] == "completed15" and row["physical_step_attempts"] == 15
    assert len(histories) == 3 and len(row["chunks"]) == 3
    np.testing.assert_array_equal(histories[1][0][-1], np.full(10, .25, np.float32))
    assert np.all(histories[1][1] == 5) and np.all(histories[2][1] == 10)


@pytest.mark.parametrize("fails", [False, True])
def test_cooperative_lock_covers_callback_and_releases_on_exit(tmp_path, fails):
    lock_path = tmp_path / "cooperative.lock"
    lock_path.touch()
    entered = []
    def experiment():
        entered.append(True)
        fd = os.open(lock_path, os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)
        if fails:
            raise RuntimeError("simulated environment failure")
        return "finished"
    if fails:
        with pytest.raises(RuntimeError, match="environment failure"):
            subject.run_locked(experiment, lock_path=lock_path)
    else:
        assert subject.run_locked(experiment, lock_path=lock_path) == "finished"
    assert entered == [True]
    fd = os.open(lock_path, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def test_lock_failure_prevents_experiment_entry(tmp_path):
    path = tmp_path / "lock"
    path.touch()
    fd = os.open(path, os.O_RDWR)
    calls = []
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            subject.run_locked(lambda: calls.append("entered"), lock_path=path)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    assert not calls


def test_cli_enters_experiment_only_while_shared_lock_held(tmp_path, monkeypatch):
    path = tmp_path / "lock"
    path.touch()
    seen = []
    def fake_run(args):
        fd = os.open(path, os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)
        seen.append((args.output, args.authorize_host_cuda_and_physical))
    monkeypatch.setattr(subject, "LOCK", path)
    monkeypatch.setattr(subject, "run_experiment", fake_run)
    monkeypatch.setattr(sys, "argv", ["runner", "--output", "fixture", "--authorize-host-cuda-and-physical"])
    subject.main()
    assert seen == [("fixture", True)]
