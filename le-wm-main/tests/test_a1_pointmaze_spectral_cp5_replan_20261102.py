import json

import numpy as np
import pytest

from experiments.a1_pointmaze_spectral_cp5_replan_20261102 import (
    ARRAYS, BANK_SHA, METHODS, OUTPUT, SOURCE, SOURCE_SHA, choose, image_hash, validate_bank,
)
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import sha256_file


def test_bank_is_bounded_and_finite():
    assert validate_bank([[0, 1], [-1, 0]], (2, 2)).dtype == np.float32
    for value in ([[0, 1.1], [0, 0]], [[0, np.nan], [0, 0]], [[0, 0]]):
        with pytest.raises(ValueError):
            validate_bank(value, (2, 2))


def test_selection_is_finite_and_first_index_tie():
    assert choose([2, 1, 1, 3, 4, 5, 6, 7]) == 1
    for costs in ([0] * 7, [0] * 7 + [np.nan]):
        with pytest.raises(ValueError):
            choose(costs)


def test_saved_result_matches_arrays_and_source():
    if not OUTPUT.exists():
        pytest.skip("development run not present")
    report, source = json.loads(OUTPUT.read_text()), json.loads(SOURCE.read_text())
    assert sha256_file(SOURCE) == SOURCE_SHA == report["source_sha256"]
    assert report["bank_source_sha256"] == BANK_SHA
    assert sha256_file(ARRAYS) == report["arrays_sha256"]
    assert report["model_before"] == report["model_after"]
    assert [r["root"] for r in report["rows"]] == [1880, 1881, 1882, 1883]
    with np.load(ARRAYS, allow_pickle=False) as data:
        for row, source_row in zip(report["rows"], source["rows"]):
            root = row["root"]
            assert image_hash(data[f"r{root}_goal_rgb"]) == row["goal_rgb_sha256"]
            assert data[f"r{root}_goal_latent"].shape == (384,)
            by_method = {b["method"]: b for b in source_row["branches"]}
            assert [b["method"] for b in row["branches"]] == list(METHODS)
            for branch in row["branches"]:
                method, key = branch["method"], f"r{root}_{branch['method']}"
                assert image_hash(data[f"{key}_rgb5"]) == branch["rgb5_sha256"]
                assert image_hash(data[f"{key}_rgb10"]) == branch["rgb10_sha256"]
                assert data[f"{key}_latent5"].shape == data[f"{key}_latent10"].shape == (384,)
                assert choose(branch["tail_costs"]) == branch["tail_index"]
                np.testing.assert_array_equal(branch["tail_action"], report["bank"][branch["tail_index"]])
                if method != "zero":
                    np.testing.assert_array_equal(branch["prefix_action"], by_method[method]["selected_action"])
                    assert branch["rgb5_sha256"] == by_method[method]["rgb5_sha256"]
                else:
                    assert branch["prefix_action"] == [0, 0]
                distance = np.linalg.norm(np.array(branch["endpoint_eval_only"][:2]) -
                                          np.array(row["goal_position_eval_only"]))
                assert distance == pytest.approx(branch["distance_eval_only"], abs=1e-12)
