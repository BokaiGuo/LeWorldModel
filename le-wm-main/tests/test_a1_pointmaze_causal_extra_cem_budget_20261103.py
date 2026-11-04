"""Saved traces/inputs and accounting; does not rerun physics or CUDA."""
import json

import numpy as np

from experiments.a1_pointmaze_causal_extra_cem_budget_20261103 import (
    ARRAYS, CAMERA, OLD, OUTPUT, ROOTS, causal_state, cem, cem_four, image_hash, sha256_file,
)


def test_saved_cem_traces_and_action_bindings():
    report, camera, old = (json.loads(p.read_text()) for p in (OUTPUT, CAMERA, OLD))
    assert sha256_file(ARRAYS) == report["arrays_sha256"]
    assert report["model_before"] == report["model_after"]
    assert [row["root"] for row in report["rows"]] == list(ROOTS)
    with np.load(ARRAYS, allow_pickle=False) as arrays:
        index = 0
        for row in report["rows"]:
            root = row["root"]
            for branch in row["branches"]:
                method = branch["method"]
                key = f"r{root}_{method}"
                history = arrays[f"{key}_history"]
                np.testing.assert_array_equal(history[0], arrays[f"r{root}_rgb0"])
                input5 = causal_state(history[-2], history[-1], camera["metadata"],
                                      old["body_offset_from_static_scene"], .05)
                np.testing.assert_array_equal(input5, branch["input5"])
                assert image_hash(arrays[f"{key}_rgb10"]) == branch["rgb10_sha256"]
                expected_n = 64 if method == "random_extra" else 48
                assert branch["counts"] == {"initial_batches": expected_n//16, "initial_samples": expected_n,
                                             "tail_batches": 3, "tail_samples": 48}
                assert (branch["jvp_calls"], branch["svd_calls"]) == ((0, 0) if method == "random_extra" else (2, 1))
                for phase in ("initial", "tail"):
                    entries = iter(branch[f"{phase}_trace"])
                    def scorer(proposals):
                        item = next(entries)
                        np.testing.assert_array_equal(proposals, np.asarray(item["proposals"], np.float32))
                        return np.asarray(item["scores"])
                    if phase == "initial" and method == "random_extra":
                        plan, cost, _ = cem_four(arrays[f"r{root}_initial_noise"], scorer)
                    elif phase == "initial":
                        plan, cost, _ = cem(arrays[f"r{root}_initial_noise"][:3], scorer, branch["injected"])
                    else:
                        plan, cost, _ = cem(arrays[f"r{root}_tail_noise"], scorer)
                    np.testing.assert_array_equal(plan, np.asarray(branch["initial_plan" if phase == "initial" else "tail"], np.float32))
                    assert cost == branch[f"{phase}_cost"]
                seq = arrays["sequences"][index]
                np.testing.assert_array_equal(seq[:5], np.tile(np.asarray(branch["initial_plan"])[0], (5, 1)))
                np.testing.assert_array_equal(seq[5:], np.tile(np.asarray(branch["tail"])[0], (5, 1)))
                dist = np.linalg.norm(arrays["endpoints_eval_only"][index, :2] - row["goal_position_eval_only"])
                assert np.isclose(dist, branch["distance_eval_only"])
                assert bool(dist < .5) == branch["strict_endpoint_success_eval_only"]
                index += 1
