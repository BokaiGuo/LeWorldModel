import numpy as np
import pytest
import torch

from experiments.pc_fixed_prefix_compiler_20261107_codex import compile_fixed_prefix
from experiments.pc_fixed_prefix_shared_20261107_codex import search
from experiments.pc_fixed_prefix_state_adapter_20261107_codex import compile_state
from experiments.pc_fixed_prefix_rgb_shared_adapter_20261107_codex import compile_rgb
from experiments.pc_input_permissions import RGB, STATE, RGBLeWM, StateAssistedPointMaze, freeze_audit


class Identity:
    def transform(self, x):
        return x


class RGBModel(torch.nn.Module):
    def get_cost(self, ctx, actions):
        return ((actions - 0.3) ** 2).sum((2, 3))


class StateCore(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.use_proprio = True
        self.proprio_encoder = torch.nn.Identity()


class StateModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = StateCore()

    def encode(self, inputs):
        return {"visual": inputs["visual"], "proprio": inputs["proprio"]}

    def unroll(self, encoded, act_suffix):
        return {"cost": ((act_suffix - 0.3) ** 2).sum()}


def arrays():
    rng = np.random.default_rng(23)
    return (rng.uniform(-1, 1, (4, 10, 2)).astype(np.float32),
            rng.uniform(-1, 1, (3, 5, 2)).astype(np.float32),
            rng.uniform(-1, 1, (2, 5, 2)).astype(np.float32))


@pytest.mark.parametrize("strategy", ["parent", "lookahead", "vector", "shuffled"])
def test_shared_core_matches_existing_rgb_compiler(tmp_path, strategy):
    dep = tmp_path / "checkpoint"
    dep.write_text("frozen")
    audit = freeze_audit(RGB, "lewm", [dep], preprocessing="test", clock="2x5")
    boundary = RGBLeWM(RGBModel().eval(), audit)
    ctx = {"pixels": torch.zeros(1, 1, 1), "goal": torch.zeros(1, 1, 1), "action": torch.zeros(1, 1, 0)}
    parents, probes, tails = arrays()
    params = dict(prefix_steps=5, retain_count=2, strategy=strategy, seed=9)
    old = compile_fixed_prefix(boundary, ctx, parents, probes, tails, scaler=Identity(), max_queries=20, **params)
    new = search(lambda a: ((torch.from_numpy(a) - 0.3) ** 2).sum((1, 2)).numpy(),
                 parents, probes, tails, budget=20, **params)
    assert old["retained_prefixes"] == new["retained_prefixes"]
    assert old["choice"].source == new["choice"].source and old["choice"].index == new["choice"].index
    np.testing.assert_array_equal(old["actions"], new["actions"])
    rgb = compile_rgb(boundary, ctx, parents, probes, tails, scaler=Identity(), budget=20, **params)
    assert rgb["track"] == RGB and rgb["retained_prefixes"] == old["retained_prefixes"]
    assert rgb["choice"] == old["choice"]
    np.testing.assert_array_equal(rgb["actions"], old["actions"])


def test_state_boundary_receipts_and_no_rgb_promotion(tmp_path):
    dep = tmp_path / "checkpoint"
    dep.write_text("frozen")
    audit = freeze_audit(STATE, "pointmaze_jepa", [dep], proprio_fields=("x", "y", "vx", "vy"),
                         preprocessing="test", clock="2x5")
    boundary = StateAssistedPointMaze(StateModel().eval(), audit)
    observation = {"visual": torch.zeros(1, 1, 3, 2, 2), "proprio": torch.zeros(1, 1, 4)}
    parents, probes, tails = arrays()
    result = compile_state(boundary, observation, observation, parents, probes, tails,
                           mean=torch.zeros(2), std=torch.ones(2),
                           objective_factory=lambda _: lambda prediction, actions: prediction["cost"],
                           prefix_steps=5, retain_count=2, strategy="lookahead", budget=20, seed=9)
    assert result["track"] == STATE and result["predict_calls"] == 20
    assert len(boundary.receipts) == 21 and boundary.receipts[0]["operation"] == "encode_goal"
    assert all(item["track_id"] == STATE for item in boundary.receipts)
    with pytest.raises(ValueError):
        compile_state(boundary, {"visual": observation["visual"]}, observation, parents, probes, tails,
                      mean=torch.zeros(2), std=torch.ones(2), objective_factory=lambda _: None,
                      prefix_steps=5, retain_count=2, strategy="lookahead", budget=20, seed=9)
