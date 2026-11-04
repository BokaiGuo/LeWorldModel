"""Saved intervention arithmetic and source bindings; no replay is rerun."""
import json

import numpy as np
import torch

from experiments.pc_pointmaze_ap_spectral_cross_suffix_20261103 import (
    ARRAYS, DEFAULT_DATASET, OUTPUT, PROTOCOL, SOURCE, SOURCE_SHA, sha256_file,
)


def test_saved_cross_suffix():
    report = json.loads(OUTPUT.read_text())
    assert sha256_file(ARRAYS) == report["arrays_sha256"]
    assert sha256_file(PROTOCOL) == report["protocol_sha256"]
    assert sha256_file(SOURCE) == report["source_sha256"] == SOURCE_SHA
    assert report["physical_sequences"] == report["independent_endpoint_checks"] == 72
    assert report["source_chains_reproduced"] == 16 and not report["new_model_inference"]
    states = torch.load(DEFAULT_DATASET / "states.pth", weights_only=True, map_location="cpu").float().numpy()
    rows = report["rows"]
    index = {(r["root"], r["prefix"], r["suffix"]): i for i, r in enumerate(rows)}
    assert len(index) == 72
    with np.load(ARRAYS, allow_pickle=False) as data, \
            np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as source:
        for i, row in enumerate(rows):
            root, prefix, suffix = row["root"], row["prefix"], row["suffix"]
            np.testing.assert_array_equal(data["starts"][i], states[root, 0].astype(np.float64))
            np.testing.assert_array_equal(data["cp5"][i], row["cp5_state"])
            np.testing.assert_array_equal(data["cp10"][i], row["cp10_state"])
            np.testing.assert_array_equal(data["legacy"][i, :2], data["cp10"][i, :2])
            expected_prefix = np.zeros((5, 2), np.float32) if prefix == "zero" else source[f"r{root}_{prefix}_actions"][:5]
            expected_suffix = np.zeros((5, 2), np.float32) if suffix == "zero" else source[f"r{root}_{suffix}_actions"][5:]
            np.testing.assert_array_equal(data["sequences"][i], np.concatenate((expected_prefix, expected_suffix)))
            if prefix == suffix and prefix != "zero":
                np.testing.assert_array_equal(data["rgb10"][i], source[f"r{root}_{prefix}_frames"][-1])
            zero = data["cp10"][index[root, prefix, "zero"], :2]
            goal = np.asarray(row["goal_eval_only"])
            distance = np.linalg.norm(data["cp10"][i, :2] - goal)
            assert distance == row["cp10_distance"]
            np.testing.assert_array_equal(data["cp10"][i, :2] - zero, row["suffix_displacement_minus_zero"])
            assert distance - np.linalg.norm(zero-goal) == row["suffix_distance_change_minus_zero"]
