"""Verify the saved intervention design and metric arithmetic."""
import json

import numpy as np

from experiments.a1_pointmaze_proprio_spectral_common_suffix_20261103 import (
    ARRAYS, OUTPUT, PROTOCOL, SOURCE, sha256_file,
)


def test_common_suffix_bindings_and_outcomes():
    report, prior = json.loads(OUTPUT.read_text()), json.loads(SOURCE.read_text())
    assert sha256_file(ARRAYS) == report["arrays_sha256"]
    assert sha256_file(PROTOCOL) == report["protocol_sha256"]
    assert report["physical_sequences"] == 72
    assert report["same_arm_original_rgb_and_endpoint_exact"] == 16
    lookup = {(r["root"], r["prefix"], r["suffix"]): i for i, r in enumerate(report["rows"])}
    assert len(lookup) == 72
    with np.load(ARRAYS, allow_pickle=False) as saved, \
            np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as source:
        np.testing.assert_array_equal(saved["legacy"][:, :2], saved["cp10"][:, :2])
        for root_row in prior["rows"]:
            root = root_row["root"]
            branches = {b["arm"]: b for b in root_row["branches"]}
            for prefix in ("random", "spectral", "zero"):
                for suffix in ("zero", "random", "spectral"):
                    i = lookup[root, prefix, suffix]
                    row = report["rows"][i]
                    plan = np.zeros(2, np.float32) if prefix == "zero" else branches[prefix]["initial_plan"][0]
                    tail = np.zeros(2, np.float32) if suffix == "zero" else branches[suffix]["tail"][0]
                    np.testing.assert_array_equal(saved["sequences"][i, :5], np.tile(plan, (5, 1)))
                    np.testing.assert_array_equal(saved["sequences"][i, 5:], np.tile(tail, (5, 1)))
                    np.testing.assert_array_equal(saved["cp5"][i], row["cp5_state"])
                    np.testing.assert_array_equal(saved["cp10"][i], row["cp10_state"])
                    goal = np.asarray(row["goal_position_eval_only"])
                    distance = np.linalg.norm(saved["cp10"][i, :2] - goal)
                    zero = saved["cp10"][lookup[root, prefix, "zero"], :2]
                    assert distance == row["cp10_distance"]
                    assert (distance < .5) == row["strict_endpoint_success_eval_only"]
                    np.testing.assert_array_equal(saved["cp10"][i, :2] - zero,
                                                  row["suffix_displacement_minus_zero"])
                    assert distance - np.linalg.norm(zero - goal) == row["suffix_distance_change_minus_zero"]
                    if prefix == suffix and prefix != "zero":
                        np.testing.assert_array_equal(saved["rgb10"][i], source[f"r{root}_{prefix}_rgb10"])
