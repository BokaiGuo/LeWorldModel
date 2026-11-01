"""Only analytic tensors, real ledger, CPU; never load a neural model."""
from copy import deepcopy

import numpy as np
import torch

from experiments.spectral_reachability import p1_coherent_own_suffix_20261101_codex as own


class AnalyticModel:
    def encode(self, info):
        assert set(info) == {"pixels"}
        return {"emb": torch.full((1, 1, 192), 0.125)}

    def action_encoder(self, actions):
        return actions

    def predict(self, states, actions):
        return states + actions[:, -1, :].mean(dim=-1, keepdim=True)[:, None, :]


def test_five_prefix_contexts_and_real_durable_ledgers(tmp_path):
    _, runner = own.configure()
    value = own.build(runner)
    assert [r["prefix"] for r in value["rows"]] == list(own.PREFIXES)
    assert value["hold_reference"] == "actual_prefix5_then_zero15_not_original_plan"
    for row in value["rows"]:
        task = own.make_task(value["base_task"], row, [0.125] * 192, value["warmup_raw"], runner)
        assert task["root_context"]["latent_history"][:2] == value["base_task"]["root_context"]["latent_history"][-2:]
        raw10 = np.concatenate((np.asarray(value["warmup_raw"], np.float32),
                                np.asarray(row["prefix_raw"], np.float32)))
        assert task["root_context"]["action_history"] == runner.rgb.moving.normalized(task, raw10)
        assert "state" not in task and "cp5_state_identity_sha256" not in task
        hold = own.hold_plan(row)
        assert np.array_equal(hold[0], np.asarray(row["prefix_raw"], np.float32))
        assert np.count_nonzero(hold[1:]) == 0

    row = value["rows"][0]
    task = own.make_task(value["base_task"], row, [0.125] * 192, value["warmup_raw"], runner)
    _, enc_contract, _ = own.encode_contract(task, value["model_sha256"], runner)
    digest = own.spec.digest(enc_contract)
    driver = own.durable.Driver(tmp_path / "encode", enc_contract, digest, "structured", "node", "encode")
    leaf = own.leafmod.LeafAdapter(AnalyticModel(), driver, enc_contract, digest, analytic_fixture=True)
    encoded = leaf.encode({"pixels": torch.zeros(1, 3, 224, 224)})
    receipt = driver.finish()
    account = own.consumer.consume(receipt["journal"], receipt["sha256"], digest, "structured", "node", "encode")
    assert encoded["emb"].shape == (1, 1, 192)
    assert account["charged_upper_bounds"]["encode_calls"] == 1
    assert account["charged_upper_bounds"]["encode_samples"] == 1

    policy, contract, _ = own.node_contract(task, value["model_sha256"], runner)
    digest = own.spec.digest(contract)
    driver = own.durable.Driver(tmp_path / "node", contract, digest, "structured", "node", "node")
    leaf = own.leafmod.LeafAdapter(AnalyticModel(), driver, contract, digest, analytic_fixture=True)
    prefix = torch.tensor(row["prefix_raw"])
    previous = torch.cat((prefix[None], torch.zeros(3, 5, 2)))
    events = []
    result = own.coherent.node.run_node(
        leaf, arm="structured", boundary=5,
        states=torch.tensor(task["root_context"]["latent_history"])[None],
        history=torch.tensor(task["root_context"]["action_history"])[None],
        goal=torch.tensor(task["goal"])[None], mean=torch.tensor(task["action_mean"]),
        std=torch.tensor(task["action_std"]), matrix=torch.eye(192, dtype=torch.float64),
        previous=previous, issued=prefix, emit=lambda kind, payload: events.append((kind, deepcopy(payload))))
    receipt = driver.finish()
    account = own.consumer.consume(receipt["journal"], receipt["sha256"], digest, "structured", "node", "node")
    assert policy["horizon"] == 3
    assert result["selected_raw"].shape == (3, 5, 2)
    assert result["charges"] == account["charged_upper_bounds"] == value["node_cost"]
    assert account["charged_upper_bounds"]["encode_calls"] == 0
    assert len(next(p["scores"] for k, p in events if k == "suffix_exchange")) == 64
    assert len(result["scores"]) == 9
