import json

import numpy as np
import pytest

from experiments.a0_pointmaze_cem_action_response_20261103 import decompose
from experiments.a0_pointmaze_cem_all_candidates_20261102 import pairwise
from experiments.a0_pointmaze_proprio_action_response_20261103 import (
    ARRAYS, OUTPUT, REPLAY, REPLAY_SHA, RESPONSE, RESPONSE_SHA, SOURCE, SOURCE_SHA, sha256_file,
)


def test_saved_privileged_diagnostic_arithmetic_and_ancestry():
    if not OUTPUT.exists():
        pytest.skip("CUDA diagnostic not complete")
    report = json.loads(OUTPUT.read_text())
    replay, response = json.loads(REPLAY.read_text()), json.loads(RESPONSE.read_text())
    assert sha256_file(SOURCE) == report["source_sha256"] == SOURCE_SHA
    assert sha256_file(REPLAY) == report["replay_sha256"] == REPLAY_SHA
    assert sha256_file(RESPONSE) == report["response_sha256"] == RESPONSE_SHA
    assert sha256_file(ARRAYS) == report["arrays_sha256"]
    assert report["model_before"] == report["model_after"] == response["model_before"]
    with np.load(ARRAYS, allow_pickle=False) as arrays, np.load(RESPONSE.with_suffix(".npz")) as geometry:
        for row, old, before in zip(report["rows"], replay["rows"], response["rows"]):
            assert row["root"] == old["root"] == before["root"]
            for branch, old_branch in zip(row["branches"], old["branches"]):
                key = f"r{row['root']}_{branch['method']}"
                actual = geometry[f"{key}_actual_features"]
                zero = geometry[f"{key}_predicted_features"]
                predicted = arrays[f"{key}_predicted_features"]
                direction = geometry[f"{key}_goal_direction"]
                assert branch["state_choice"] == int(np.argmin(branch["state_scores"]))
                assert branch["original_choice"] == int(np.argmin(old_branch["prediction"]))
                assert branch["state_distance"] == old_branch["distance_eval_only"][branch["state_choice"]]
                assert branch["state_prediction_vs_rgb"] == pairwise(branch["state_scores"], old_branch["actual_rgb_cost"])
                assert branch["zero_proprio_prediction_mse"] == pytest.approx(float(np.mean((zero-actual)**2)))
                assert branch["state_proprio_prediction_mse"] == pytest.approx(float(np.mean((predicted-actual)**2)))
                reproduced = decompose(predicted, actual, direction)
                for field, value in reproduced.items():
                    np.testing.assert_allclose(branch["state_response"][field], value, rtol=1e-12, atol=1e-12)
