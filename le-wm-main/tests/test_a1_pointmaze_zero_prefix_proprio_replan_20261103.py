"""Recompute saved zero-prefix optimizer and outcome bindings."""
import json

import numpy as np

from experiments.a1_pointmaze_zero_prefix_proprio_replan_20261103 import (
    ARRAYS, COMMON, NOISE, OUTPUT, PROTOCOL, SOURCE, cem, sha256_file,
)


def test_zero_prefix_feedback_results():
    report, common, previous = (json.loads(path.read_text()) for path in (OUTPUT, COMMON, SOURCE))
    assert report["model_before"] == report["model_after"]
    assert sha256_file(ARRAYS) == report["arrays_sha256"]
    assert sha256_file(PROTOCOL) == report["protocol_sha256"]
    assert report["physical_executions"] == 8
    cp5_by_root = {r["root"]: r["cp5_state"] for r in common["rows"]
                   if r["prefix"] == "zero" and r["suffix"] == "zero"}
    with np.load(ARRAYS, allow_pickle=False) as saved, np.load(NOISE.with_suffix(".npz"), allow_pickle=False) as noise:
        for i, row in enumerate(report["rows"]):
            root = row["root"]
            trace = iter(row["tail_trace"])

            def scorer(proposals):
                recorded = next(trace)
                np.testing.assert_array_equal(proposals, np.asarray(recorded["proposals"], np.float32))
                return recorded["scores"]

            tail, cost, _ = cem(noise[f"r{root}_tail_noise"], scorer)
            np.testing.assert_array_equal(tail, np.asarray(row["tail"], np.float32))
            assert cost == row["tail_cost"]
            assert row["prediction_counts"] == {"batches": 3, "samples": 48}
            np.testing.assert_array_equal(saved["sequences"][i, :5], np.zeros((5, 2)))
            np.testing.assert_array_equal(saved["sequences"][i, 5:], np.repeat(tail[:1], 5, axis=0))
            np.testing.assert_array_equal(saved["cp5_eval_only"][i], cp5_by_root[root])
            np.testing.assert_array_equal(saved["endpoints_eval_only"][i], row["endpoint_eval_only"])
            distance = np.linalg.norm(saved["endpoints_eval_only"][i, :2] - row["goal_position_eval_only"])
            assert distance == row["distance_eval_only"]
            assert (distance < .5) == row["strict_endpoint_success_eval_only"]
            assert row["prior_random_distance"] == previous["rows"][i]["branches"][0]["distance_eval_only"]
            assert row["prior_spectral_distance"] == previous["rows"][i]["branches"][1]["distance_eval_only"]
