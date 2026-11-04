import json

import numpy as np
import torch

from experiments.a1_pointmaze_cp5_history_diagnostic_20261102 import (
    ARRAYS, OUTPUT, PREDICTIONS, SOURCE, SOURCE_SHA, history_prediction,
)
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import sha256_file


def test_history_alignment_uses_actual_prefix_and_tail():
    class Core:
        def encode_act(self, actions):
            self.actions = actions
            return actions

        def forward_pred(self, visual, actions, proprio):
            assert visual.shape == (8, 2, 1, 1, 1, 2)
            assert proprio.shape == (8, 2, 1, 2)
            assert torch.equal(actions[:, 0], torch.arange(10).expand(8, -1))
            assert torch.equal(actions[:, 1], tails)
            return visual, None, proprio

    model = type("Wrapper", (), {"model": Core()})()
    first = {"visual": torch.zeros(1, 1, 1, 1, 1, 2), "proprio": torch.zeros(1, 1, 1, 2)}
    second = {"visual": torch.ones(1, 1, 1, 1, 1, 2), "proprio": torch.ones(1, 1, 1, 2)}
    tails = torch.arange(80).reshape(8, 10)
    prediction = history_prediction(model, first, second, torch.arange(10), tails)
    assert prediction.shape == (8, 2)
    assert torch.all(prediction == 1)


def test_saved_predictions_recompute_costs_and_mse():
    result, source = json.loads(OUTPUT.read_text()), json.loads(SOURCE.read_text())
    assert sha256_file(SOURCE) == SOURCE_SHA == result["source_sha256"]
    assert sha256_file(PREDICTIONS) == result["predictions_sha256"]
    assert sha256_file(ARRAYS) == result["source_arrays_sha256"]
    with np.load(ARRAYS, allow_pickle=False) as observations, np.load(PREDICTIONS, allow_pickle=False) as preds:
        for row in result["rows"]:
            root = row["root"]
            goal = observations[f"r{root}_goal_latent"]
            for branch in row["branches"]:
                key = f"r{root}_{branch['method']}"
                selected = branch["executed_tail"]
                actual = observations[f"{key}_latent10"]
                for kind in ("single", "history"):
                    prediction = preds[f"{key}_{kind}"]
                    np.testing.assert_allclose(np.mean((prediction - goal) ** 2, axis=1),
                                               branch[f"{kind}_costs"], rtol=1e-5, atol=1e-7)
                    np.testing.assert_allclose(np.mean((prediction[selected] - actual) ** 2),
                                               branch[f"{kind}_executed_latent_mse"], rtol=1e-5, atol=1e-7)
