"""Recompute fixed intervention matrix and provenance bindings."""
import json

import numpy as np

from experiments.a1_pointmaze_four_prefix_common_suffix_20261103 import (
    ARRAYS, COMMON, GREEDY, OUTPUT, PROTOCOL, SPECTRAL, ZERO, sha256_file,
)


def test_factorial_bindings():
    report, previous, greedy, spectral, zero = (
        json.loads(p.read_text()) for p in (OUTPUT, COMMON, GREEDY, SPECTRAL, ZERO))
    assert sha256_file(ARRAYS) == report["arrays_sha256"]
    assert sha256_file(PROTOCOL) == report["protocol_sha256"]
    assert (report["physical_sequences"], report["old_panel_exact"],
            report["greedy_and_zero_original_exact"]) == (160, 72, 16)
    lookup = {(r["root"], r["prefix"], r["suffix"]): i for i, r in enumerate(report["rows"])}
    old = {(r["root"], r["prefix"], r["suffix"]): i for i, r in enumerate(previous["rows"])}
    assert len(lookup) == 160
    with np.load(ARRAYS, allow_pickle=False) as saved, \
            np.load(COMMON.with_suffix(".npz"), allow_pickle=False) as earlier:
        np.testing.assert_array_equal(saved["legacy"][:, :2], saved["cp10"][:, :2])
        for i, row in enumerate(report["rows"]):
            root, p, s = row["root"], row["prefix"], row["suffix"]
            k = root-1888
            arms = {b["arm"]: b for b in spectral["rows"][k]["branches"]}
            prefixes = {"greedy": greedy["rows"][k]["prefix"][0], "zero": [0., 0.],
                        "random": arms["random"]["initial_plan"][0], "spectral": arms["spectral"]["initial_plan"][0]}
            tails = {"zero_action": [0., 0.], "greedy": greedy["rows"][k]["tail"][0],
                     "zero_replanned": zero["rows"][k]["tail"][0], "random": arms["random"]["tail"][0],
                     "spectral": arms["spectral"]["tail"][0]}
            np.testing.assert_array_equal(saved["sequences"][i, :5], np.tile(prefixes[p], (5, 1)))
            np.testing.assert_array_equal(saved["sequences"][i, 5:], np.tile(tails[s], (5, 1)))
            np.testing.assert_array_equal(saved["cp5"][i], row["cp5_state"])
            np.testing.assert_array_equal(saved["cp10"][i], row["cp10_state"])
            goal = np.asarray(row["goal_position_eval_only"])
            distance = np.linalg.norm(saved["cp10"][i, :2]-goal)
            control = saved["cp10"][lookup[root, p, "zero_action"], :2]
            assert distance == row["cp10_distance"]
            assert distance - np.linalg.norm(control-goal) == row["suffix_distance_change_minus_zero"]
            assert (distance < .5) == row["strict_endpoint_success_eval_only"]
            old_s = "zero" if s == "zero_action" else s
            if p in ("zero", "random", "spectral") and old_s in ("zero", "random", "spectral"):
                idx = old[root, p, old_s]
                np.testing.assert_array_equal(saved["rgb10"][i], earlier["rgb10"][idx])
                np.testing.assert_array_equal(saved["legacy"][i], earlier["legacy"][idx])
