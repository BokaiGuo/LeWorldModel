import json

import numpy as np
import pytest
import torch

from experiments.a1_pointmaze_rgb_position_history_diagnostic_20261103 import (
    ARRAYS, CAMERA, CAMERA_SHA, OUTPUT, PREFIX_SHA, PREFIX_SOURCE, PROTOCOL, SOURCE,
    SOURCE_SHA, masked_cost, predictions, sha256_file,
)


def test_history_action_token_order_and_shape():
    class Inner:
        def encode_act(self, actions):
            self.actions = actions.clone()
            return actions

        def forward_pred(self, visual, action, proprio):
            assert visual.shape[1] == proprio.shape[1] == action.shape[1] == 2
            return torch.zeros(2, 2, 1, 16, 16, 384), None, None

    class Model:
        model = Inner()

        def unroll(self, context, act_suffix):
            assert act_suffix.shape == (1, 2, 10)
            return {"visual": torch.zeros(1, 2, 1, 16, 16, 384)}

    first = {"visual": torch.zeros(1, 1, 1, 16, 16, 384), "proprio": torch.zeros(1, 1, 1, 4)}
    prefix = torch.arange(10, dtype=torch.float32)
    tails = torch.stack((prefix + 10, prefix + 20))
    model = Model()
    single, history = predictions(model, first, first, prefix, tails)
    assert single.shape == history.shape == (2, 1, 16, 16, 384)
    torch.testing.assert_close(model.model.actions[:, 0], prefix.expand(2, -1))
    torch.testing.assert_close(model.model.actions[:, 1], tails)
    with pytest.raises(ValueError):
        predictions(model, first, first, prefix[:2], tails)


def test_saved_latent_metrics_and_actions():
    assert OUTPUT.exists() and ARRAYS.exists(), "diagnostic output missing"
    result, source, prefix = (json.loads(p.read_text()) for p in (OUTPUT, SOURCE, PREFIX_SOURCE))
    assert sha256_file(SOURCE) == SOURCE_SHA == result["source_sha256"]
    assert sha256_file(PREFIX_SOURCE) == PREFIX_SHA == result["prefix_source_sha256"]
    assert sha256_file(CAMERA) == CAMERA_SHA == result["camera_sha256"]
    assert sha256_file(PROTOCOL) == result["protocol_sha256"]
    assert sha256_file(ARRAYS) == result["arrays_sha256"]
    assert result["model_before"] == result["model_after"] == source["model_before"]
    with np.load(ARRAYS, allow_pickle=False) as arrays:
        for row, prior, origin in zip(result["rows"], source["rows"], prefix["rows"], strict=True):
            assert row["root"] == prior["root"] == origin["root"]
            for i, (branch, parent) in enumerate(zip(row["branches"], origin["branches"], strict=True)):
                group = prior["branches"][3*i:3*i+3]
                assert branch["initial_method"] == parent["method"]
                np.testing.assert_array_equal(branch["prefix_action"], parent["initial_plan"][0])
                np.testing.assert_array_equal(branch["tail_actions"], [b["tail"][0] for b in group])
                assert branch["rgb10_sha256"] == [b["rgb10_sha256"] for b in group]
                assert branch["distance_eval_only"] == [b["distance_eval_only"] for b in group]
                key = f"r{row['root']}_{branch['initial_method']}"
                actual = arrays[f"{key}_actual"]
                for method in ("single", "history"):
                    predicted = arrays[f"{key}_{method}"]
                    mse = np.mean((predicted - actual) ** 2, axis=(1, 2, 3, 4))
                    np.testing.assert_allclose(mse, branch["prediction_mse"][method], rtol=2e-5, atol=1e-7)
                    assert np.argmin(branch["predicted_goal_cost"][method]) == branch["selected_index"][method]


def test_masked_cost_rejects_incompatible_shape():
    with pytest.raises(ValueError):
        masked_cost(torch.zeros(1, 2), torch.zeros(2), torch.ones(2, dtype=torch.bool))
