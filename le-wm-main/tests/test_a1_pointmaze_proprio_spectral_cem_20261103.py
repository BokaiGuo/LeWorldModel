"""Recompute saved CEM, spectral and endpoint bindings, without rerunning physics."""
import json

import numpy as np

from experiments.a1_pointmaze_proprio_spectral_cem_20261103 import (
    ARRAYS, OUTPUT, PROTOCOL, SOURCE, cem, sha256_file,
)


def recompute(noise, trace, injected=None):
    iterator = iter(trace)

    def score(proposals):
        row = next(iterator)
        np.testing.assert_array_equal(proposals, np.asarray(row["proposals"], np.float32))
        return row["scores"]

    return cem(noise, score, injected)


def test_saved_spectral_and_feedback_bindings():
    report = json.loads(OUTPUT.read_text())
    assert sha256_file(ARRAYS) == report["arrays_sha256"]
    assert sha256_file(PROTOCOL) == report["protocol_sha256"]
    assert report["model_before"] == report["model_after"]
    assert report["physical_executions"] == 16
    with np.load(ARRAYS, allow_pickle=False) as saved, \
            np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as source:
        for i, row in enumerate(report["rows"]):
            root = row["root"]
            for j, branch in enumerate(row["branches"]):
                spectral = branch["arm"] == "spectral"
                assert (branch["jvp_calls"], branch["svd_calls"]) == ((2, 1) if spectral else (0, 0))
                assert branch["prediction_counts"] == {
                    "initial_batches": 3, "initial_samples": 48,
                    "tail_batches": 3, "tail_samples": 48,
                }
                injected = None if not spectral else np.asarray(branch["injected"], np.float32)
                if spectral:
                    jac = saved[f"r{root}_jacobian"]
                    singular = saved[f"r{root}_singular"]
                    direction = injected[0, 0] / .3
                    np.testing.assert_allclose(np.linalg.norm(direction), 1., atol=1e-6)
                    np.testing.assert_allclose(np.linalg.norm(jac @ direction), singular[0], rtol=1e-5)
                    np.testing.assert_allclose(injected[:, 0], np.array([.3, -.3, .6, -.6])[:, None] * direction,
                                               atol=1e-7)
                    np.testing.assert_array_equal(injected[:, 1], np.zeros((4, 2)))
                plan, cost, _ = recompute(source[f"r{root}_initial_noise"][:3], branch["initial_trace"], injected)
                tail, tail_cost, _ = recompute(source[f"r{root}_tail_noise"], branch["tail_trace"])
                np.testing.assert_array_equal(plan, np.asarray(branch["initial_plan"], np.float32))
                np.testing.assert_array_equal(tail, np.asarray(branch["tail"], np.float32))
                assert (cost, tail_cost) == (branch["initial_cost"], branch["tail_cost"])
                sequence = np.concatenate((np.repeat(plan[:1], 5, axis=0), np.repeat(tail[:1], 5, axis=0)))
                np.testing.assert_array_equal(sequence, saved["sequences"][2*i+j])
                endpoint = saved["endpoints_eval_only"][2*i+j]
                np.testing.assert_array_equal(endpoint, branch["endpoint_eval_only"])
                distance = np.linalg.norm(endpoint[:2] - row["goal_position_eval_only"])
                assert distance == branch["distance_eval_only"]
                assert (distance < .5) == branch["strict_endpoint_success_eval_only"]
