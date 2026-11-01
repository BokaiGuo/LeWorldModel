import pytest
import torch

from experiments.spectral_reachability import codex_four_arm_durable_20261030 as durable
from experiments.spectral_reachability import codex_four_arm_full_cost_20261030 as spec
from experiments.spectral_reachability import codex_four_arm_leaf_suffix_20261030 as leaf_module
from experiments.spectral_reachability import codex_four_arm_raw_suffix_20261030_main as module


class Analytic:
    def __init__(self):
        self.contexts = []

    def encode(self, info):
        return info

    def action_encoder(self, value):
        self.contexts.append(value.detach().clone())
        return value

    def predict(self, state, actions):
        return state + actions


def setup(tmp_path):
    cfg = {"steps": 1, "learning_rate": 1.0, "history_size": 3}
    ident = {key: "a" * 64 for key in spec.IDENTITY}
    ident["optimizer_config"] = spec.digest(cfg)
    arms = {arm: {"initializer_sha256": "b" * 64, "optimizer_sha256": ident["optimizer"],
                  "optimizer_config_sha256": ident["optimizer_config"]} for arm in spec.ARMS}
    caps = dict.fromkeys(spec.AXES, 100)
    contract = dict(schema=spec.NAME + ".contract.v1", identity=ident, arms=arms,
                    scope="standalone_per_root_per_arm_including_attributable_setup",
                    phase_policy_sha256="c" * 64,
                    levels=[dict(id="a", caps=caps), dict(id="b", caps={k: v + 1 for k, v in caps.items()})])
    digest = spec.digest(contract)
    driver = durable.Driver(tmp_path / "ledger", contract, digest, "structured", "a", "test")
    model = Analytic()
    leaf = leaf_module.LeafAdapter(model, driver, contract, digest, analytic_fixture=True)
    return cfg, driver, model, leaf


def test_raw_normalization_history_gradient_and_bounds(tmp_path):
    cfg, driver, model, leaf = setup(tmp_path)
    raw = torch.zeros(1, 2, 5, 2)
    past = torch.arange(20, dtype=torch.float32).reshape(1, 2, 10)
    mean, std = torch.tensor([.25, -.5]), torch.tensor([.5, 2.])
    out = module.optimize(leaf, torch.zeros(1, 3, 10), past, raw, torch.ones(1, 10),
                          mean, std, cfg)
    expected_token = ((raw[0, 0] - mean) / std).flatten()
    assert torch.equal(model.contexts[0][0, :2], past[0])
    assert torch.equal(model.contexts[0][0, 2], expected_token)
    assert torch.equal(model.contexts[1][0, 0], past[0, 1])
    assert torch.equal(model.contexts[1][0, 1], expected_token)
    assert torch.all(out.abs() <= 1)
    # Two normalized packets add to the final state. With the chosen mean
    # and std, d(mean squared error)/d(raw) is (-0.8, -0.05) per step.
    assert torch.allclose(out, torch.tensor([.8, .05]).expand_as(out), atol=1e-6, rtol=0)
    assert driver.charged["predict_calls"] == 2
    assert driver.charged["backward_calls"] == 1
    driver.finish()


def test_bad_normalizer_fails_closed_before_any_model_call(tmp_path):
    cfg, driver, model, leaf = setup(tmp_path)
    with pytest.raises(ValueError, match="dimensions or bounds"):
        module.optimize(leaf, torch.zeros(1, 3, 10), torch.zeros(1, 2, 10),
                        torch.zeros(1, 1, 5, 2), torch.zeros(1, 10),
                        torch.zeros(2), torch.tensor([1., 0.]), cfg)
    assert not model.contexts
    assert driver.failed
    driver.finish()
