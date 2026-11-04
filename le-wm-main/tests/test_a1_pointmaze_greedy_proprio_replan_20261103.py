"""Validate saved greedy CEM traces, actions, metrics and prior bindings."""
import json

import numpy as np

from experiments.a1_pointmaze_greedy_proprio_replan_20261103 import (
    ARRAYS, NOISE, OUTPUT, PROTOCOL, SOURCE, ZERO, cem, sha256_file,
)


def rechoose(noise, recorded):
    trace = iter(recorded)

    def scorer(proposals):
        row = next(trace)
        np.testing.assert_array_equal(proposals, np.asarray(row["proposals"], np.float32))
        return row["scores"]

    return cem(noise, scorer)


def test_greedy_feedback_bindings():
    report, previous, zero = (json.loads(path.read_text()) for path in (OUTPUT, SOURCE, ZERO))
    assert sha256_file(ARRAYS) == report["arrays_sha256"]
    assert sha256_file(PROTOCOL) == report["protocol_sha256"]
    assert report["model_before"] == report["model_after"]
    assert report["physical_executions"] == 8
    with np.load(ARRAYS, allow_pickle=False) as saved, np.load(NOISE.with_suffix(".npz"), allow_pickle=False) as noise:
        for i, row in enumerate(report["rows"]):
            root = row["root"]
            prefix, cost0, _ = rechoose(noise[f"r{root}_initial_noise"][:3, :, :1], row["initial_trace"])
            tail, cost5, _ = rechoose(noise[f"r{root}_tail_noise"], row["tail_trace"])
            np.testing.assert_array_equal(prefix, np.asarray(row["prefix"], np.float32))
            np.testing.assert_array_equal(tail, np.asarray(row["tail"], np.float32))
            assert (cost0, cost5) == (row["initial_cost"], row["tail_cost"])
            assert row["prediction_counts"] == {
                "initial_batches": 3, "initial_samples": 48, "tail_batches": 3, "tail_samples": 48,
            }
            np.testing.assert_array_equal(saved["sequences"][i, :5], np.repeat(prefix[:1], 5, axis=0))
            np.testing.assert_array_equal(saved["sequences"][i, 5:], np.repeat(tail[:1], 5, axis=0))
            np.testing.assert_array_equal(saved["cp5_eval_only"][i], row["cp5_state_eval_only"])
            np.testing.assert_array_equal(saved["endpoints_eval_only"][i], row["endpoint_eval_only"])
            goal = np.asarray(row["goal_position_eval_only"])
            distance = np.linalg.norm(saved["endpoints_eval_only"][i, :2] - goal)
            assert distance == row["distance_eval_only"]
            assert (distance < .5) == row["strict_endpoint_success_eval_only"]
            assert row["prior_zero_distance"] == zero["rows"][i]["distance_eval_only"]
            assert row["prior_random_distance"] == previous["rows"][i]["branches"][0]["distance_eval_only"]
            assert row["prior_spectral_distance"] == previous["rows"][i]["branches"][1]["distance_eval_only"]
