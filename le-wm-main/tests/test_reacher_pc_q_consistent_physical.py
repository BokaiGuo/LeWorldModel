"""Paired-root restoration checks; no neural inference or physical claim."""
import gymnasium as gym
import numpy as np
import pytest

from experiments.spectral_reachability import run_reacher_pc_q_consistent_physical as runner


def test_goal_restore_forwards_to_wrapped_environment():
    class GoalEnv(gym.Env):
        def __init__(self):
            self._goal = np.zeros((2, 2), dtype=np.uint8)

    inner = GoalEnv()
    wrapped = runner.RestorableObserveSteps(inner)
    goal = np.ones((2, 2), dtype=np.uint8)
    wrapped._goal = goal
    assert wrapped._goal is goal
    assert inner._goal is goal


def test_each_arm_restores_and_verifies_same_snapshot(monkeypatch):
    calls = []
    context = {"execution_snapshot": object(), "root_frame": object(), "goal": object()}
    restored_frame, restored_goal = object(), object()

    def restore(env, *, seed, snapshot):
        calls.append(("restore", env, seed, snapshot))
        return restored_frame, restored_goal

    def verify(frame, goal, expected_frame, expected_goal):
        calls.append(("verify", frame, goal, expected_frame, expected_goal))
        return {"verified": True}

    monkeypatch.setattr(runner.setup.p0, "restore_execution_snapshot", restore)
    monkeypatch.setattr(runner.setup.p0, "_verify_execution_identity", verify)
    for _ in range(3):
        frame, identity = runner.restore_paired_root("env", 42, context)
        assert frame is restored_frame
        assert identity == {"verified": True}
    assert len(calls) == 6
    assert all(calls[i] == ("restore", "env", 42, context["execution_snapshot"])
               for i in (0, 2, 4))
    assert all(calls[i] == ("verify", restored_frame, restored_goal,
                            context["root_frame"], context["goal"])
               for i in (1, 3, 5))


def test_failed_root_identity_is_not_accepted(monkeypatch):
    monkeypatch.setattr(runner.setup.p0, "restore_execution_snapshot",
                        lambda *args, **kwargs: ("wrong", "goal"))

    def reject(*args):
        raise RuntimeError("identity mismatch")

    monkeypatch.setattr(runner.setup.p0, "_verify_execution_identity", reject)
    with pytest.raises(RuntimeError, match="identity mismatch"):
        runner.restore_paired_root("env", 1, {
            "execution_snapshot": {}, "root_frame": "expected", "goal": "goal"})
