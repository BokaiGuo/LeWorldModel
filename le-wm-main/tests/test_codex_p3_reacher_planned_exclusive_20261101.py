import json

import numpy as np
import pytest
import torch
from torch import nn

from jepa import JEPA
from experiments.spectral_reachability import codex_p3_chunk_extension_20261101 as kernel
from experiments.spectral_reachability import codex_p3_reacher_planned_exclusive_20261101 as runner
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0


class Predictor(nn.Module):
    def forward(self, states, actions):
        return states + actions[..., :2]


class Env:
    def __init__(self, path):
        self.path = path
        self.actions = []

    def step(self, action):
        assert (self.path / "action_binding.json").is_file()
        self.actions.append(np.array(action))
        return None, 0.0, False, False, {}


def test_real_cem_conversion_and_both_schedulers(monkeypatch, tmp_path):
    # Exercise the real CEM and bound conversion, using a CPU objective fixture.
    monkeypatch.setattr(p0, "_cost_fn", lambda model, pixels, goal, history, params, **kw:
        (p0._bounded_normalized_actions(params, kw["action_lower"], kw["action_upper"]) - 0.4).square().mean((1, 2)))
    mean = np.zeros(10, np.float32)
    std = np.ones(10, np.float32)
    low, high = -np.ones(2, np.float32), np.ones(2, np.float32)
    history = torch.zeros(1, 3, 10)
    args = (p0, None, None, None, history, mean, std, low, high, 44, 8, 2, 2)
    raw, receipt = runner.plan_actions(*args)
    raw2, _ = runner.plan_actions(*args)
    assert np.array_equal(raw, raw2)
    assert raw.shape == (15, 2) and np.any(raw != 0)
    assert receipt["forward_rollout_evaluations"] == 16
    for arm in ("fixed", "dynamic"):
        output = tmp_path / arm
        output.mkdir()
        env = Env(output)
        model = JEPA(nn.Identity(), Predictor(), nn.Identity())
        meter = kernel.Meter({key: 8 if key in ("predict", "predictor_samples") else 0 for key in kernel.AXES})
        result = runner.run_bound_chunk(output=output, model=model, env=env,
            states=torch.zeros(1, 3, 2), history_actions=torch.zeros(1, 2, 10),
            raw=raw, mean=mean, std=std, low=low, high=high,
            goal=torch.ones(2), projection=torch.eye(2), block=5,
            config=kernel.Config(5, 5, 2, 0, 100), arm=arm, meter=meter)
        seal = json.loads((output / "action_binding.json").read_text())
        assert np.array_equal(np.array(env.actions), raw[:len(env.actions)])
        assert np.array_equal(np.asarray(seal["physical_actions"], np.float32), raw)
        assert len(env.actions) == result["chunk_steps"]
        assert not result["equal_total_compute"]


def test_planner_rejects_invalid_budget():
    with pytest.raises(ValueError):
        runner.plan_actions(None, None, None, None, None, None, None, None, None, 0, 2, 1, 1)
