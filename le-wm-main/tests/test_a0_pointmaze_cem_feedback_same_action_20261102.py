import json

import numpy as np
import pytest
import torch

from experiments.a0_pointmaze_cem_all_candidates_20261102 import pairwise
from experiments.a0_pointmaze_cem_feedback_same_action_20261102 import (
    ARRAYS, OUTPUT, PREV_SHA, SOURCE_SHA, image_hash, sha256_file, suffix, task_error,
)


def test_suffix_normalizes_and_preserves_packet_order():
    actions = torch.tensor([[[1.0, 2.0], [-1.0, 4.0]]])
    actual = suffix(actions, torch.tensor([0.0, 2.0]), torch.tensor([1.0, 2.0]))
    assert actual.shape == (2, 1, 10)
    torch.testing.assert_close(actual[0, 0], torch.tensor([1.0, 0.0] * 5))
    torch.testing.assert_close(actual[1, 0], torch.tensor([-1.0, 1.0] * 5))


def test_task_error_uses_only_mask():
    features = torch.zeros(2, 1, 2, 2, 3)
    features[0, :, 0, 0] = 2
    features[1, :, 1, 1] = 10
    mask = torch.tensor([[True, False], [False, False]])
    torch.testing.assert_close(task_error(features, torch.zeros(1, 2, 2, 3), mask),
                               torch.tensor([4.0, 0.0]))


def test_saved_results():
    if not OUTPUT.exists():
        pytest.skip("CUDA physical diagnostic has not completed")
    report = json.loads(OUTPUT.read_text())
    assert report["source_sha256"] == SOURCE_SHA
    assert report["prior_sha256"] == PREV_SHA
    assert report["arrays_sha256"] == sha256_file(ARRAYS)
    assert report["model_before"] == report["model_after"]
    with np.load(ARRAYS, allow_pickle=False) as arrays:
        for row in report["rows"]:
            for branch in row["branches"]:
                key = f"r{row['root']}_{branch['method']}_rgb5"
                assert [image_hash(image) for image in arrays[key]] == branch["rgb5_sha256"]
                assert pairwise(branch["open_scores"], branch["actual_scores"]) == branch["open_vs_actual_rgb"]
                assert pairwise(branch["refresh_scores"], branch["actual_scores"]) == branch["refresh_vs_actual_rgb"]
                np.testing.assert_allclose(np.mean(branch["open_prediction_mse"]), branch["open_mean_mse"])
                np.testing.assert_allclose(np.mean(branch["refresh_prediction_mse"]), branch["refresh_mean_mse"])
