"""Saved trace and replay bindings; not a new CUDA/physical execution."""
import json

import numpy as np

from experiments.a1_pointmaze_full_causal_spectral_dev_20261103 import (
    ARRAYS, OLD, ORIGINAL, OUTPUT, PRIOR, cem, image_hash, sha256_file,
)


def test_complete_recompilation_saved_bindings():
    report, old, prior = (json.loads(p.read_text()) for p in (OUTPUT, OLD, PRIOR))
    assert sha256_file(ARRAYS) == report["arrays_sha256"]
    assert report["model_before"] == report["model_after"]
    assert (report["jvp_calls"], report["svd_calls"]) == (8, 4)
    with np.load(ARRAYS, allow_pickle=False) as saved, \
            np.load(ORIGINAL.with_suffix(".npz"), allow_pickle=False) as noise, \
            np.load(PRIOR.with_suffix(".npz"), allow_pickle=False) as prev:
        index = 0
        for row, oldrow, priorrow in zip(report["rows"], old["rows"], prior["rows"], strict=True):
            assert row["root"] == oldrow["root"] == priorrow["root"]
            root = row["root"]
            for branch in row["branches"]:
                method = branch["method"]
                parent = next(x for x in oldrow["branches"] if x["method"] == method and x["policy"] == "rgb_position")
                expected = next(x for x in priorrow["branches"] if x["method"] == method and x["policy"] == "causal_velocity")
                np.testing.assert_array_equal(branch["initial_plan"], parent["initial_plan"])
                np.testing.assert_array_equal(branch["tail"], expected["tail"])
                key = f"r{root}_{method}"
                np.testing.assert_array_equal(saved[f"{key}_history"], prev[f"{key}_causal_velocity_history"])
                assert image_hash(saved[f"{key}_rgb10"]) == branch["rgb10_sha256"] == expected["rgb10_sha256"]
                assert branch["counts"] == {"initial_batches": 3, "initial_samples": 48,
                                             "tail_batches": 3, "tail_samples": 48}
                for phase, noise_key, injected in (
                    ("initial", "initial_noise", parent["injected"]),
                    ("tail", "tail_noise", None),
                ):
                    trace = iter(branch[f"{phase}_trace"])
                    def scorer(proposals):
                        item = next(trace)
                        np.testing.assert_array_equal(proposals, np.asarray(item["proposals"], np.float32))
                        return np.asarray(item["scores"])
                    plan, cost, _ = cem(noise[f"r{root}_{noise_key}"], scorer, injected)
                    np.testing.assert_array_equal(plan, np.asarray(branch["initial_plan" if phase == "initial" else "tail"], np.float32))
                    assert cost == branch["initial_cost" if phase == "initial" else "tail_cost"]
                pos = saved["endpoints_eval_only"][index, :2]
                assert np.isclose(np.linalg.norm(pos - row["goal_position_eval_only"]), branch["distance_eval_only"])
                index += 1
