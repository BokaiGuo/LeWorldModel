import copy

import pytest
import torch

from experiments.spectral_reachability import codex_four_arm_durable_20261030 as durable
from experiments.spectral_reachability import codex_four_arm_full_cost_20261030 as spec
from experiments.spectral_reachability import codex_four_arm_leaf_suffix_20261030 as leafmod
from experiments.spectral_reachability import codex_four_arm_final_replay_20261030 as replay
from experiments.spectral_reachability import codex_p1_fixed_prefix_common_suffix_20261030_main as module


class Model:
    def __init__(self):
        self.contexts = []

    def encode(self, value):
        return value

    def action_encoder(self, value):
        self.contexts.append(value.detach().clone())
        return value

    def predict(self, states, actions):
        return states + actions


def build(tmp_path, bad_budget=False):
    config = {"steps": 1, "learning_rate": .2, "history_size": 3}
    policy = {"schema": replay.POLICY_SCHEMA, "candidate_ids": ["a", "b"], "horizon": 3,
              "history_size": 3, "metric": "terminal_latent_mean_squared_error",
              "tie": "first_declared_candidate", "replay_batch": 1,
              "action_chart": "raw_packet_5x2_box_minus1_to1",
              "mean": [.25, -.5], "std": [.5, 2.]}
    identity = {key: "a" * 64 for key in spec.IDENTITY}
    identity["optimizer_config"] = spec.digest(config)
    arms = {arm: {"initializer_sha256": "b" * 64, "optimizer_sha256": identity["optimizer"],
                  "optimizer_config_sha256": identity["optimizer_config"]} for arm in spec.ARMS}
    caps = module.required_cost(2, 1)
    if bad_budget:
        caps["predict_calls"] -= 1
    contract = {"schema": spec.NAME + ".contract.v1", "identity": identity, "arms": arms,
                "scope": "standalone_per_root_per_arm_including_attributable_setup",
                "phase_policy_sha256": spec.digest(policy),
                "levels": [{"id": "a", "caps": caps},
                           {"id": "b", "caps": {k: v + 1 for k, v in caps.items()}}]}
    digest = spec.digest(contract)
    driver = durable.Driver(tmp_path / "ledger", contract, digest, "structured", "a", "test")
    model = Model()
    leaf = leafmod.LeafAdapter(model, driver, contract, digest, analytic_fixture=True)
    return config, policy, contract, driver, model, leaf


def invoke(leaf, config, policy, contract, events):
    prefixes = torch.stack((torch.full((1, 5, 2), .2), torch.full((1, 5, 2), -.2)))
    return module.plan(leaf, torch.zeros(1, 3, 10), torch.arange(20.).reshape(1, 2, 10),
                       prefixes, torch.zeros(1, 2, 5, 2), torch.ones(1, 10),
                       torch.tensor(policy["mean"]), torch.tensor(policy["std"]),
                       config, policy, contract, emit=lambda k, v: events.append((k, v)))


def test_prefix_history_common_initialization_and_full_replay(tmp_path):
    config, policy, contract, driver, model, leaf = build(tmp_path)
    events = []
    result = invoke(leaf, config, policy, contract, events)
    assert result["actions"].shape == (2, 3, 5, 2)
    assert torch.all(result["actions"][0, 0] == .2)
    assert torch.all(result["actions"][1, 0] == -.2)
    token = (torch.tensor([.2, .2]) - torch.tensor(policy["mean"])) / torch.tensor(policy["std"])
    assert torch.allclose(model.contexts[1][0, 1], token.repeat(5))
    normalized = (result["actions"] - torch.tensor(policy["mean"])) / torch.tensor(policy["std"])
    predicted = normalized.reshape(2, 3, 10).sum(dim=1)
    expected = (predicted - 1.).square().mean(dim=1)
    assert result["scores"] == pytest.approx(expected.tolist(), abs=1e-6)
    assert result["selected_index"] == int(torch.argmin(expected))
    assert [kind for kind, _ in events] == ["inputs", "optimized", "optimized", "selection"]
    assert driver.charged == module.required_cost(2, 1)
    assert result["P1_admission"] is False
    driver.finish()


def test_insufficient_budget_aborts_before_model_calls(tmp_path):
    config, policy, contract, driver, model, leaf = build(tmp_path, bad_budget=True)
    with pytest.raises(ValueError, match="reserve"):
        invoke(leaf, config, policy, contract, [])
    assert model.contexts == [] and driver.failed
    driver.finish()


def test_policy_mismatch_aborts_before_model_calls(tmp_path):
    config, policy, contract, driver, model, leaf = build(tmp_path)
    altered = copy.deepcopy(policy)
    altered["std"] = [1., 1.]
    with pytest.raises(ValueError, match="roster"):
        invoke(leaf, config, altered, contract, [])
    assert model.contexts == [] and driver.failed
    driver.finish()
