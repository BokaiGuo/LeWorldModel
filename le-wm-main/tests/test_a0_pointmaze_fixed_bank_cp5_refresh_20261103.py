"""Saved-data and arithmetic checks; does not rerun physical or CUDA execution."""
import json

import numpy as np
import torch

from experiments.a0_pointmaze_fixed_bank_cp5_refresh_20261103 import (
    ARRAYS, OUTPUT, REAL, SOURCE, metrics, pairwise_ranking, sha256_file,
)


def test_masked_cost_and_prediction_error():
    prediction = torch.zeros(2, 1, 16, 16, 384)
    actual = torch.zeros_like(prediction)
    goal = torch.zeros(1, 16, 16, 384)
    mask = torch.zeros(16, 16, dtype=torch.bool)
    mask[3, 5] = True
    prediction[0, 0, 3, 5] = 2
    actual[0, 0, 3, 5] = 1
    goal[0, 3, 5] = 3
    prediction[1, 0, 0, 0] = 100
    cost, mse = metrics(prediction, actual, goal, mask)
    torch.testing.assert_close(cost, torch.tensor([1., 9.]))
    torch.testing.assert_close(mse, torch.tensor([1., 0.]))


def test_saved_result_bindings():
    report = json.loads(OUTPUT.read_text())
    assert sha256_file(ARRAYS) == report["arrays_sha256"]
    assert report["cp10_exact_image_and_endpoint_count"] == 384
    assert report["model_before"] == report["model_after"]
    with np.load(ARRAYS, allow_pickle=False) as arrays, \
            np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as source, \
            np.load(REAL.with_suffix(".npz"), allow_pickle=False) as actual:
        assert arrays["rgb5"].shape == (384, 224, 224, 3)
        assert arrays["rgb5"].dtype == np.uint8
        for row in report["rows"]:
            key = f"r{row['root']}_{row['method']}"
            open_score, refresh = arrays[f"{key}_open_cost"], arrays[f"{key}_refresh_cost"]
            distance, actual_score = source[f"{key}_distance"], actual[f"{key}_real_cost"]
            np.testing.assert_allclose(open_score, source[f"{key}_prediction"], rtol=1e-4, atol=1e-6)
            assert pairwise_ranking(open_score, actual_score) == row["open_vs_real_rgb"]
            assert pairwise_ranking(refresh, actual_score) == row["refresh_vs_real_rgb"]
            assert int(np.argmin(refresh)) == row["refresh_oracle_choice"]
            assert distance[row["refresh_oracle_choice"]] == row["refresh_oracle_choice_distance"]
            assert np.isclose(arrays[f"{key}_open_mse"].mean(), row["open_masked_mse_mean"])
            assert np.isclose(arrays[f"{key}_refresh_mse"].mean(), row["refresh_masked_mse_mean"])
