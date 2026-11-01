import json
from pathlib import Path

import pytest
import torch

from experiments.spectral_reachability import codex_p1_cp5_common_suffix_20261030_v1 as bridge


def row():
    path = Path(__file__).resolve().parents[1] / "results/codex_p1_three_prefix_blind_inputs_20261030_v1.json"
    return json.loads(path.read_text())["rows"][0]


def test_context_uses_actual_cp5_and_own_raw_prefix():
    source = row()
    cp5 = torch.arange(192, dtype=torch.float32)
    states, actions, goal, mean, std = bridge.context(source, cp5, device="cpu")
    assert states.shape == (1, 3, 192) and actions.shape == (1, 2, 10)
    assert torch.equal(states[0, :2], torch.tensor(source["root_latent_history"][1:]))
    assert torch.equal(states[0, 2], cp5)
    assert torch.equal(actions[0, 0], torch.tensor(source["root_action_history"][-1]))
    import numpy as np
    raw = np.frombuffer(bytes.fromhex(source["prefix_raw_f32_hex"]), "<f4").copy().reshape(5, 2)
    assert torch.equal(actions[0, 1], ((torch.from_numpy(raw)-mean)/std).reshape(10))
    assert torch.equal(goal[0], torch.tensor(source["goal_latent"]))


def test_plan_uses_same_optimizer_and_zero_initialization(monkeypatch):
    seen = {}
    def fake(leaf, states, actions, initial, goal, mean, std, config):
        seen.update(leaf=leaf, states=states, actions=actions, initial=initial, config=config)
        return initial
    monkeypatch.setattr(bridge.optimizer, "optimize", fake)
    marker, config = object(), {"steps": 2}
    result = bridge.plan(row(), torch.ones(192), marker, config)
    assert result.shape == (1, 2, 5, 2) and torch.count_nonzero(result) == 0
    assert seen["leaf"] is marker and seen["config"] is config
    assert torch.equal(seen["states"][0, 2], torch.ones(192))


def test_rejects_outcome_leak_and_bad_latent():
    source = row()
    source["future_state"] = [1]
    with pytest.raises(ValueError, match="whitelist"):
        bridge.context(source, torch.zeros(192), device="cpu")
    with pytest.raises(ValueError, match="finite"):
        bridge.context(row(), torch.full((192,), float("nan")), device="cpu")


def test_actual_optimizer_uses_metered_calls_and_own_history(tmp_path):
    from experiments.spectral_reachability import codex_four_arm_durable_20261030 as durable
    from experiments.spectral_reachability import codex_four_arm_full_cost_20261030 as spec
    from experiments.spectral_reachability import codex_four_arm_leaf_suffix_20261030 as leaves

    class Model:
        def __init__(self):
            self.contexts = []
        def encode(self, info):
            return info
        def action_encoder(self, value):
            self.contexts.append(value.detach().clone())
            return value
        def predict(self, state, action):
            # Differentiable analytic fixture, not a model ABI or CUDA test.
            return state + action.mean(dim=-1, keepdim=True)

    cfg = {"steps": 2, "learning_rate": 0.05, "history_size": 3}
    identity = {key: "a" * 64 for key in spec.IDENTITY}
    identity["optimizer_config"] = spec.digest(cfg)
    arms = {arm: {"initializer_sha256": "b" * 64, "optimizer_sha256": identity["optimizer"],
                  "optimizer_config_sha256": identity["optimizer_config"]} for arm in spec.ARMS}
    caps = dict.fromkeys(spec.AXES, 100)
    contract = {"schema": spec.NAME + ".contract.v1", "identity": identity, "arms": arms,
                "scope": "standalone_per_root_per_arm_including_attributable_setup",
                "phase_policy_sha256": "c" * 64,
                "levels": [{"id": "a", "caps": caps},
                           {"id": "b", "caps": {k: v + 1 for k, v in caps.items()}}]}
    digest = spec.digest(contract)
    driver = durable.Driver(tmp_path / "ledger", contract, digest, "structured", "a", "cp5")
    model = Model()
    leaf = leaves.LeafAdapter(model, driver, contract, digest, analytic_fixture=True)
    cp5 = torch.arange(192, dtype=torch.float32)
    expected = bridge.context(row(), cp5, device="cpu")
    result = bridge.plan(row(), cp5, leaf, cfg)
    assert result.shape == (1, 2, 5, 2) and bool((result.abs() <= 1).all())
    assert torch.equal(model.contexts[0][:, :2], expected[1])
    assert driver.charged["action_encode_calls"] == 4
    assert driver.charged["predict_calls"] == 4
    assert driver.charged["backward_calls"] == 2
    driver.finish()
