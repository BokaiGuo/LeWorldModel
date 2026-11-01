import json

import numpy as np
import pytest
import torch
from torch import nn

from jepa import JEPA
from experiments.spectral_reachability import codex_p3_chunk_extension_20261101 as kernel
from experiments.spectral_reachability import codex_p3_reacher_live_exclusive_20261101 as runner


class Env:
    def __init__(self, output):
        self.output = output
        self.actions = []

    def step(self, action):
        assert (self.output / "action_binding.json").is_file()
        self.actions.append(np.array(action))
        return None, 0, False, False, {}


class Predictor(nn.Module):
    def forward(self, states, actions):
        return states + actions


def test_live_binding_sealed_before_predict_and_steps(tmp_path):
    model = JEPA(nn.Identity(), Predictor(), nn.Identity())
    env = Env(tmp_path)
    original = model.predict

    def checked(*args):
        assert (tmp_path / "action_binding.json").is_file()
        assert not env.actions
        return original(*args)

    model.predict = checked
    meter = kernel.Meter({axis: 8 if axis in ("predict", "predictor_samples") else 0
                          for axis in kernel.AXES})
    result = runner.run_bound_chunk(output=tmp_path, model=model, env=env,
        states=torch.zeros(1, 3, 2), history_actions=torch.zeros(1, 2, 2),
        raw=np.array([[1., 3.], [2., 4.], [3., 5.]], dtype=np.float32),
        mean=np.array([1., 1.]), std=np.array([2., 2.]),
        low=np.array([-10., -10.]), high=np.array([10., 10.]),
        goal=torch.tensor([5., 5.]), projection=torch.eye(2), block=1,
        config=kernel.Config(1, 1, 2, 0, 100), arm="dynamic", meter=meter)
    seal = json.loads((tmp_path / "action_binding.json").read_text())
    assert seal["normalized_tokens"] == [[0., 1.], [0.5, 1.5], [1., 2.]]
    assert np.array_equal(np.asarray(env.actions), np.asarray(seal["physical_actions"]))
    assert result["resources"]["attempted"]["predict"] == 8
    assert not result["equal_total_compute"]
    with pytest.raises(FileExistsError):
        runner.write_once(tmp_path / "action_binding.json", seal)


def test_binding_rejects_invalid_values():
    with pytest.raises(ValueError):
        runner.bind_actions([[2., 0.]], [0., 0.], [1., 1.], [-1., -1.], [1., 1.], 1)
    with pytest.raises(ValueError):
        runner.bind_actions([[0., 0.]], [0., 0.], [1., 0.], [-1., -1.], [1., 1.], 1)
