import numpy as np
import pytest
import torch

from experiments.spectral_reachability import codex_p1_diag32_cp5_live_20261101_exclusive as live


class Store:
    def __init__(self):
        self.events = []

    def emit(self, kind, **value):
        self.events.append((kind, value))


class Model:
    def action_encoder(self, context):
        return context

    def predict(self, states, encoded):
        return states + encoded.mean() * torch.ones_like(states)


def test_contract_is_bound_to_existing_preseal():
    c = live.build()
    assert c["preseal_sha256"] == live.PRESEAL_SHA
    assert c["roster"] == ["selected", "zero", "same_bank_greedy"]
    assert c["same_episode"] and not c["physical_future_used_for_selection"]


def test_model_wiring_has_fixed_call_counts_and_no_extra_dispatch():
    store = Store()
    meter = live.NeuralMeter(store, "zero", lambda: None)
    meter.call("cp5_rgb_encode", {}, lambda: None)
    task = {"root_context": {"latent_history": np.zeros((3, 192)).tolist(),
                             "action_history": np.zeros((2, 10)).tolist()},
            "goal": np.ones(192).tolist(), "action_mean": [0., 0.], "action_std": [1., 1.]}
    result = live.optimize(Model(), torch.device("cpu"), task, np.zeros((8, 2, 5, 2)), meter, store, "zero")
    meter.complete()
    assert result["candidate"] == 0
    assert meter.attempts == {"cp5_rgb_encode": 1, "action_encoder": 528, "predict": 528, "reverse": 256}
    assert sum(kind == "candidate_score" for kind, _ in store.events) == 264
    with pytest.raises(ValueError):
        meter.call("predict", {}, lambda: None)
