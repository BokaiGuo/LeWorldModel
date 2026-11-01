"""CPU-only analytic fixture and read-only frozen-bank source checks."""
from copy import deepcopy
import numpy as np
import pytest
import torch

from experiments.spectral_reachability import p1_coherent_same_bank_greedy_20261101_codex as greedy


class AnalyticFixture:
    def encode(self, info):
        raise AssertionError("encode is prohibited")

    def action_encoder(self, actions):
        return actions

    def predict(self, states, actions):
        return states + actions[:, -1, :].mean(-1)[:, None, None]


def test_frozen_source_and_prefix_protection():
    value = greedy.build()
    assert np.asarray(value["prefixes"]).shape == (8, 5, 2)
    assert value["expected_cost"] == greedy.COST
    assert value["expected_cost"]["encode_calls"] == 0
    assert value["expected_cost"]["replay_calls"] == 8
    assert value["expected_cost"]["selection_evaluations"] == 8
    events = [
        dict(kind="initial_candidates", payload=dict(raw=np.zeros((8, 4, 5, 2)).tolist())),
        dict(kind="optimized_candidates_before_exchange", payload=dict(raw=np.zeros((8, 4, 5, 2)).tolist())),
    ]
    assert greedy.frozen_candidates(events).shape == (8, 5, 2)
    changed = deepcopy(events)
    changed[1]["payload"]["raw"][0][0][0][0] = .1
    with pytest.raises(ValueError, match="prefix identity"):
        greedy.frozen_candidates(changed)


def test_one_packet_scores_ties_and_real_durable_ledger(tmp_path):
    value = greedy.build()
    _, contract, _ = greedy.make_contract(value)
    digest = greedy.spec.digest(contract)
    driver = greedy.durable.Driver(tmp_path / "ledger", contract, digest, "structured", "node", "same_bank")
    leaf = greedy.leafmod.LeafAdapter(AnalyticFixture(), driver, contract, digest, analytic_fixture=True)
    prefix = torch.arange(7, -1, -1, dtype=torch.float32)[:, None, None].expand(8, 5, 2) / 8
    states = torch.zeros(1, 3, 192)
    history = torch.zeros(1, 2, 10)
    scores, index = greedy.score_bank(leaf, prefix, states, history, torch.zeros(1, 192),
                                      torch.zeros(2), torch.ones(2), torch.eye(192))
    np.testing.assert_allclose(scores, (np.arange(7, -1, -1) / 8) ** 2)
    assert index == 7
    receipt = driver.finish()
    account = greedy.consumer.consume(receipt["journal"], receipt["sha256"], digest,
                                      "structured", "node", "same_bank")
    assert account["charged_upper_bounds"] == greedy.COST
    events = greedy.saved.chain(receipt["journal"])
    intents = [event["payload"]["kind"] for event in events if event["kind"] == "intent"]
    assert intents == ["replay", "action_encode", "predict", "selection"] * 8

    tie = torch.zeros(8, 5, 2)
    class Leaf:
        def _dispatch(self, phase, kind, cost, fn): return fn()
        def action_encode(self, x, **kwargs): return x
        def predict(self, states, actions, **kwargs): return states
    scores, index = greedy.score_bank(Leaf(), tie, states, history, torch.zeros(1, 192),
                                      torch.zeros(2), torch.ones(2), torch.eye(192))
    assert scores == [0.] * 8 and index == 0
