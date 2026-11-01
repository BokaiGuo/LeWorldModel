import pytest
import torch
from torch import nn

from jepa import JEPA
from experiments.spectral_reachability import codex_p3_chunk_extension_20261101 as kernel
from experiments.spectral_reachability import codex_p3_live_chunk_adapter_20261101 as adapter


class Predictor(nn.Module):
    def forward(self, states, actions):
        return states + actions


class Env:
    def __init__(self, stop=None):
        self.actions = []
        self.stop = stop

    def step(self, action):
        self.actions.append(action)
        return None, 0, len(self.actions) == self.stop, False, {}


def setup(arm="dynamic", cap=20, stop=None):
    model = JEPA(nn.Identity(), Predictor(), nn.Identity())
    env = Env(stop)
    meter = kernel.Meter({key: cap if key in ("predict", "predictor_samples") else 0
                          for key in kernel.AXES})
    return dict(model=model, env=env, states=torch.zeros(1, 3, 1),
                history_actions=torch.zeros(1, 2, 1),
                action_tokens=torch.ones(1, 3, 1),
                physical_actions=[1, 2, 3], goal=torch.tensor([3.]),
                projection=torch.ones(1, 1), token_steps=1,
                config=kernel.Config(1, 1, 2, 0.1, 1.1), arm=arm, meter=meter)


def test_real_jepa_abi_decides_before_environment_steps_and_counts():
    args = setup()
    observed = []
    original = args["model"].predict
    def predict(*values):
        observed.append(len(args["env"].actions))
        return original(*values)
    args["model"].predict = predict
    result = adapter.run_chunk(**args)
    assert observed == [0] * 8  # 1+2, then 2+3
    assert result["chunk_steps"] == 3
    assert args["env"].actions == [1, 2, 3]
    assert result["resources"]["attempted"]["predict"] == 8
    assert result["resources"]["returned"]["predictor_samples"] == 8
    assert result["resources"]["physical_env_steps"] == {"attempted": 3, "returned": 3}
    assert not result["P3_admission"]


def test_fixed_and_early_termination_do_not_predict():
    args = setup(arm="fixed", stop=1)
    result = adapter.run_chunk(**args)
    assert result["terminated"]
    assert result["resources"]["attempted"]["predict"] == 0
    assert result["resources"]["physical_env_steps"]["returned"] == 1


def test_budget_failure_before_any_execution():
    args = setup(cap=2)
    with pytest.raises(ValueError, match="cap exceeded"):
        adapter.run_chunk(**args)
    assert args["env"].actions == []
    assert args["meter"].attempted["predict"] == 2


def test_rejected_extension_still_charged():
    args = setup()
    args["config"] = kernel.Config(1, 1, 2, 100, 1.1)
    result = adapter.run_chunk(**args)
    assert result["chunk_steps"] == 1
    assert result["decisions"][0]["reason"] == "insufficient_predicted_improvement"
    assert result["resources"]["attempted"]["predict"] == 3


def test_model_failure_charged_without_execution():
    args = setup()
    def fail(*_):
        raise RuntimeError("predict failed")
    args["model"].predict = fail
    with pytest.raises(RuntimeError, match="predict failed"):
        adapter.run_chunk(**args)
    assert args["meter"].attempted["predict"] == 1
    assert args["meter"].returned["predict"] == 0
    assert args["env"].actions == []
