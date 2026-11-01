import numpy as np
import pytest
import torch

from experiments.spectral_reachability import codex_p1_diag32_cp5_hold_20261101_exclusive as hold


class Store:
    def __init__(self):
        self.events = []

    def emit(self, kind, **payload):
        self.events.append((kind, payload))


class Model:
    def action_encoder(self, context):
        return context

    def predict(self, states, encoded):
        return states + encoded.mean() * torch.ones_like(states)


def test_contract_binds_each_original_tail():
    value = hold.build()
    assert set(value["hold_tails"]) == {"selected", "zero", "same_bank_greedy"}
    assert np.count_nonzero(value["hold_tails"]["zero"]["raw"]) == 0
    assert value["neural_caps_per_branch"]["predict"] == 532
    assert value["physical_future_used_for_selection"] is False


def test_equal_replay_and_hold_tie():
    store = Store()
    meter = hold.NeuralMeter(store, "zero", lambda: None)
    task = {"root_context": {"latent_history": np.zeros((3, 192)).tolist(),
                             "action_history": np.zeros((2, 10)).tolist()},
            "goal": np.ones(192).tolist(), "action_mean": [0., 0.], "action_std": [1., 1.]}
    zero = np.zeros((2, 5, 2), np.float32)
    q1 = hold.replay_score(Model(), torch.device("cpu"), task, zero, meter, "zero", "optimized")
    q2 = hold.replay_score(Model(), torch.device("cpu"), task, zero, meter, "zero", "hold")
    assert meter.attempts["predict"] == meter.attempts["action_encoder"] == 4
    choice, selected = hold.select_tail({"q2": q1, "actions": zero.tolist()}, q1, zero, q2)
    assert choice == "hold" and np.array_equal(selected, zero)
    with pytest.raises(ValueError):
        hold.select_tail({"q2": q1 + 1, "actions": zero.tolist()}, q1, zero, q2)
