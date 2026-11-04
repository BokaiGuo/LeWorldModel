"""Recompute saved planning traces and replay bindings, without GPU/physics reruns."""
import json

import numpy as np

from experiments.a1_pointmaze_proprio_score_feedback_cem_20261103 import (
    ARRAYS, OUTPUT, PROTOCOL, SOURCE, cem, cem_four, sha256_file,
)


def replay_cem(noise, trace, fn):
    remaining = iter(trace)

    def scores(proposals):
        row = next(remaining)
        np.testing.assert_array_equal(proposals, np.asarray(row["proposals"], np.float32))
        return row["scores"]

    return fn(noise, scores)


def test_saved_feedback_execution_and_optimizer():
    report, prior = json.loads(OUTPUT.read_text()), json.loads(SOURCE.read_text())
    assert sha256_file(ARRAYS) == report["arrays_sha256"]
    assert sha256_file(PROTOCOL) == report["protocol_sha256"]
    assert report["model_before"] == report["model_after"]
    assert report["physical_executions"] == 16
    assert [r["root"] for r in report["rows"]] == list(range(1888, 1896))
    with np.load(ARRAYS, allow_pickle=False) as saved, \
            np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as source:
        for i, row in enumerate(report["rows"]):
            root = row["root"]
            target = np.asarray(row["goal_position_eval_only"])
            for j, branch in enumerate(row["branches"]):
                assert branch["prediction_counts"] == {
                    "initial_batches": 4, "initial_samples": 64,
                    "tail_batches": 3, "tail_samples": 48,
                }
                plan, cost, _ = replay_cem(source[f"r{root}_initial_noise"], branch["initial_trace"], cem_four)
                tail, tail_cost, _ = replay_cem(source[f"r{root}_tail_noise"], branch["tail_trace"], cem)
                np.testing.assert_array_equal(plan, np.asarray(branch["initial_plan"], np.float32))
                np.testing.assert_array_equal(tail, np.asarray(branch["tail"], np.float32))
                assert cost == branch["initial_cost"]
                assert tail_cost == branch["tail_cost"]
                sequence = np.concatenate((np.repeat(plan[:1], 5, axis=0), np.repeat(tail[:1], 5, axis=0)))
                np.testing.assert_array_equal(sequence, saved["sequences"][2*i+j])
                endpoint = saved["endpoints_eval_only"][2*i+j]
                np.testing.assert_array_equal(endpoint, branch["endpoint_eval_only"])
                distance = np.linalg.norm(endpoint[:2] - target)
                assert distance == branch["distance_eval_only"]
                assert (distance < .5) == branch["strict_endpoint_success_eval_only"]
                if branch["arm"] == "visual":
                    old = next(b for b in prior["rows"][i]["branches"] if b["method"] == "random_extra")
                    np.testing.assert_array_equal(endpoint, old["endpoint_eval_only"])
                    np.testing.assert_array_equal(saved[f"r{root}_visual_history"],
                                                  source[f"r{root}_random_extra_history"])
                    np.testing.assert_array_equal(saved[f"r{root}_visual_rgb10"],
                                                  source[f"r{root}_random_extra_rgb10"])
