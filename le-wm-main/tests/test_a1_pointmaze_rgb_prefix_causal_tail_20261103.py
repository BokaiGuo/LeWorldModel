"""Verify saved pair bindings and CEM traces, not a physical/CUDA rerun."""
import json

import numpy as np

from experiments.a1_pointmaze_rgb_prefix_causal_tail_20261103 import (
    ARRAYS, ORIGINAL, OUTPUT, SOURCE, cem, image_hash, sha256_file,
)


def test_saved_paired_executions_and_cem():
    report = json.loads(OUTPUT.read_text())
    assert sha256_file(ARRAYS) == report["arrays_sha256"]
    assert report["model_before"] == report["model_after"]
    assert report["physical_executions"] == 16
    source = json.loads(SOURCE.read_text())
    with np.load(ARRAYS, allow_pickle=False) as data, \
            np.load(ORIGINAL.with_suffix(".npz"), allow_pickle=False) as initial:
        sequence_index = 0
        for row, source_row in zip(report["rows"], source["rows"], strict=True):
            assert row["root"] == source_row["root"]
            root = row["root"]
            for k in range(0, 4, 2):
                a, b = row["branches"][k:k+2]
                assert (a["policy"], b["policy"]) == ("zero_velocity", "causal_velocity")
                assert a["method"] == b["method"]
                assert a["prefix"] == b["prefix"]
                assert a["history_sha256"] == b["history_sha256"]
                parent = next(x for x in source_row["branches"]
                              if x["policy"] == "rgb_position" and x["method"] == a["method"])
                np.testing.assert_array_equal(a["prefix"], np.asarray(parent["initial_plan"])[0])
                for arm in (a, b):
                    key = f"r{root}_{arm['method']}_{arm['policy']}"
                    assert image_hash(data[f"{key}_history"][-1]) == parent["rgb5_sha256"]
                    assert image_hash(data[f"{key}_rgb10"]) == arm["rgb10_sha256"]
                    assert arm["counts"] == {"batches": 3, "samples": 48}
                    trace = iter(arm["trace"])
                    def scoring(proposals):
                        step = next(trace)
                        np.testing.assert_array_equal(proposals, np.asarray(step["proposals"], np.float32))
                        return np.asarray(step["scores"])
                    tail, cost, _ = cem(initial[f"r{root}_tail_noise"], scoring)
                    np.testing.assert_array_equal(tail, np.asarray(arm["tail"], np.float32))
                    assert cost == arm["tail_cost"]
                    sequence = data["sequences"][sequence_index]
                    np.testing.assert_array_equal(sequence[:5], np.tile(arm["prefix"], (5, 1)))
                    np.testing.assert_array_equal(sequence[5:], np.tile(tail[0], (5, 1)))
                    position = data["legacy_endpoints_eval_only"][sequence_index, :2]
                    distance = np.linalg.norm(position - np.asarray(row["goal_position_eval_only"]))
                    assert np.isclose(distance, arm["distance_eval_only"])
                    assert (distance < .5) == arm["strict_endpoint_success_eval_only"]
                    sequence_index += 1
