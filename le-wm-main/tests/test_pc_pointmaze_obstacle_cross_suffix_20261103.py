"""Saved intervention bindings and arithmetic; does not rerun physics."""
import json

import numpy as np

from experiments.pc_pointmaze_obstacle_cross_suffix_20261103 import (
    ARMS, ARRAYS, GOAL, OUTPUT, PROTOCOL, ROOTS, SOURCE, SOURCE_SHA, sha256_file,
)


def test_obstacle_intervention_receipt():
    report = json.loads(OUTPUT.read_text())
    assert sha256_file(SOURCE) == report["source_sha256"] == SOURCE_SHA
    assert sha256_file(PROTOCOL) == report["protocol_sha256"]
    assert sha256_file(ARRAYS) == report["arrays_sha256"]
    assert report["physical_sequences"] == 36
    assert report["original_chains_replayed"] == 8
    rows = report["rows"]
    expected = [(r, p, s) for r in ROOTS for p in ARMS for s in ARMS]
    assert [(r["root"], r["prefix"], r["suffix"]) for r in rows] == expected
    index = {key: i for i, key in enumerate(expected)}
    with np.load(ARRAYS, allow_pickle=False) as data, np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as source:
        assert data["sequences"].shape == (36, 25, 2)
        for i, (root, prefix, suffix) in enumerate(expected):
            row, seq = rows[i], data["sequences"][i]
            expected_prefix = (np.zeros((5, 2)) if prefix == "zero" else source[f"r{root}_{prefix}_actions"][:5])
            expected_suffix = (np.zeros((20, 2)) if suffix == "zero" else source[f"r{root}_{suffix}_actions"][5:])
            np.testing.assert_array_equal(seq[:5], expected_prefix)
            np.testing.assert_array_equal(seq[5:], expected_suffix)
            np.testing.assert_array_equal(data["cp5"][i], row["cp5_state"])
            np.testing.assert_array_equal(data["cp25"][i], row["cp25_state"])
            zero = data["cp25"][index[root, prefix, "zero"], :2]
            final = data["cp25"][i, :2]
            assert np.isclose(row["initial_distance"], np.linalg.norm(data["starts"][i, :2] - GOAL[:2]))
            assert np.isclose(row["cp5_distance"], np.linalg.norm(data["cp5"][i, :2] - GOAL[:2]))
            assert np.isclose(row["cp25_distance"], np.linalg.norm(final - GOAL[:2]))
            np.testing.assert_allclose(row["suffix_displacement_minus_zero"], final - zero, atol=1e-12)
            assert np.isclose(row["suffix_distance_minus_zero"],
                              np.linalg.norm(final-GOAL[:2])-np.linalg.norm(zero-GOAL[:2]))
