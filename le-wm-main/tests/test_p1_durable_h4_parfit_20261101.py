"""Real durable ledger with a non-module analytic tensor fixture."""
import pytest
import torch

from experiments.spectral_reachability import codex_four_arm_consumer_20261030 as consumer
from experiments.spectral_reachability import codex_four_arm_durable_20261030 as durable
from experiments.spectral_reachability import codex_four_arm_full_cost_20261030 as spec
from experiments.spectral_reachability import codex_four_arm_leaf_suffix_20261030 as leafmod
from experiments.spectral_reachability import p1_h4_parfit_20261101 as h4
from experiments.spectral_reachability import p1_node_h4_parfit_20261101 as node


class Analytic:
    def encode(self, info):
        return info

    def action_encoder(self, actions):
        return actions

    def predict(self, states, actions):
        return states + actions[:, -1:, :].mean(dim=-1, keepdim=True)


@pytest.mark.parametrize("arm,boundary", [("random", 0), ("structured", 15)])
def test_real_durable_node_account(tmp_path, arm, boundary):
    caps = h4.node_cost(arm, boundary)
    identity = dict.fromkeys(spec.IDENTITY, "a" * 64)
    arms = {a: dict(initializer_sha256="b" * 64,
                    optimizer_sha256=identity["optimizer"],
                    optimizer_config_sha256=identity["optimizer_config"]) for a in h4.ARMS}
    contract = dict(schema=spec.NAME + ".contract.v1", identity=identity, arms=arms,
                    scope="standalone_per_root_per_arm_including_attributable_setup",
                    phase_policy_sha256="c" * 64,
                    levels=[dict(id="node", caps=caps),
                            dict(id="larger", caps={k: v + 1 for k, v in caps.items()})])
    sha = spec.digest(contract)
    driver = durable.Driver(tmp_path / "ledger", contract, sha, arm, "node", arm)
    leaf = leafmod.LeafAdapter(Analytic(), driver, contract, sha, analytic_fixture=True)
    h, carry = h4.dimensions(boundary)
    previous = torch.zeros(h + 1, 5, 2) if carry else None
    result = node.run_node(
        leaf, arm=arm, boundary=boundary, states=torch.zeros(1, 3, 192),
        history=torch.zeros(1, 2, 10), goal=torch.ones(1, 192),
        mean=torch.zeros(2), std=torch.ones(2), matrix=torch.eye(192),
        previous=previous, issued=previous[0] if carry else None, emit=lambda *_: None)
    receipt = driver.finish()
    account = consumer.consume(receipt["journal"], receipt["sha256"], sha, arm, "node", arm)
    assert account["status"] == "closed_unverified"
    assert account["charged_upper_bounds"] == result["charges"] == caps
