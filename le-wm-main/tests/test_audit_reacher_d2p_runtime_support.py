from types import SimpleNamespace

import numpy as np
import torch

from experiments.spectral_reachability import audit_reacher_d2p_runtime_support as audit


def test_source_state_context_rebinds_goal_to_the_restored_source_root(monkeypatch, tmp_path):
    class Physics:
        def __init__(self):
            self.data = SimpleNamespace(qpos=np.zeros(2), qvel=np.zeros(2))

        def forward(self):
            return None

    class Env:
        def __init__(self):
            self.unwrapped = SimpleNamespace(env=SimpleNamespace(physics=Physics()))
            self.action_space = SimpleNamespace(shape=(2,), low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]))
            self.goal_calls = []
            self.events = []

        def reset(self, *, seed):
            return None, {"goal": np.full((2, 2, 3), 1, dtype=np.uint8)}

        def render(self):
            self.events.append("render")
            return np.full((2, 2, 3), 7, dtype=np.uint8)

        def set_goal_from_state(self, qpos, qvel):
            self.goal_calls.append((np.asarray(qpos).copy(), np.asarray(qvel).copy()))
            self.events.append("goal")
            return np.full((2, 2, 3), 9, dtype=np.uint8)

    class Handle:
        def __enter__(self):
            return {
                "ep_len": np.array([30]), "ep_offset": np.array([0]),
                "qpos": np.arange(60, dtype=np.float64).reshape(30, 2),
                "qvel": np.arange(60, dtype=np.float64).reshape(30, 2) + 100.0,
                "action": np.zeros((30, 2), dtype=np.float32),
            }

        def __exit__(self, *args):
            return False

    class RNG:
        def __init__(self):
            self.values = iter((0, 2))

        def integers(self, *args, **kwargs):
            return next(self.values)

    monkeypatch.setattr(audit.h5py, "File", lambda *args, **kwargs: Handle())
    monkeypatch.setattr(audit.np.random, "default_rng", lambda seed: RNG())
    monkeypatch.setattr(audit, "capture_execution_snapshot", lambda env: {"snapshot": True})
    env = Env()
    context = audit._source_state_context(env, dataset=tmp_path / "source.h5", seed=19)
    assert context["warmup_policy"] == "official_source_state_history_v3_goal_bound_before_history_render"
    assert np.array_equal(context["goal"], np.full((2, 2, 3), 9, dtype=np.uint8))
    assert len(env.goal_calls) == 1
    assert env.events[0] == "goal"
    np.testing.assert_allclose(env.goal_calls[0][0], [34.0, 35.0])
    np.testing.assert_allclose(env.goal_calls[0][1], [134.0, 135.0])


def test_successor_error_encodes_advanced_successor_not_goal(monkeypatch):
    context = {
        "frames": [np.zeros((1, 1, 3), dtype=np.uint8)] * 3,
        "root_frame": np.zeros((1, 1, 3), dtype=np.uint8),
        "goal": np.full((1, 1, 3), 9, dtype=np.uint8),
        "history_action_blocks": [np.zeros(2, dtype=np.float32)] * 3,
        "execution_snapshot": {"ignored": True},
    }
    advanced = {
        "frames": context["frames"],
        "root_frame": np.full((1, 1, 3), 3, dtype=np.uint8),
        "goal": context["goal"],
        "history_action_blocks": context["history_action_blocks"],
    }
    calls = []
    monkeypatch.setattr(audit.p0, "_prepare", lambda *args: (torch.zeros((1, 3, 2)), torch.zeros((1, 1, 2)), torch.zeros((1, 3, 2))))
    monkeypatch.setattr(audit.p0, "_terminal_batch", lambda *args, **kwargs: torch.tensor([[3.0, 4.0]]))
    monkeypatch.setattr(audit.p0, "_to_raw_actions", lambda *args, **kwargs: (np.zeros((5, 2), dtype=np.float32), None))
    monkeypatch.setattr(audit.p0, "_latent_context", lambda *args: calls.append("successor") or torch.zeros((1, 3, 2)))
    monkeypatch.setattr(audit, "restore_execution_snapshot", lambda *args, **kwargs: (context["root_frame"], context["goal"]))
    monkeypatch.setattr(audit, "advance_executed_context", lambda *args, **kwargs: advanced)
    env = SimpleNamespace(action_space=SimpleNamespace(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0])))

    error = audit._successor_error(
        model=object(), preprocessor=object(), env=env, context=context, seed=1,
        mean=np.zeros(2), std=np.ones(2), lower=torch.full((2,), -1.0),
        upper=torch.full((2,), 1.0), device=torch.device("cpu"),
    )

    assert calls == ["successor"]
    assert error == 5.0
