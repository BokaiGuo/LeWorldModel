import json

import numpy as np
import pytest

from experiments.a1_pointmaze_spatial_online_dev_20261102 import ARRAYS, OUTPUT, ROOTS, image_hash
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import sha256_file


def test_saved_decisions_and_physical_arithmetic():
    if not OUTPUT.exists():
        pytest.skip("development run absent")
    report = json.loads(OUTPUT.read_text())
    assert sha256_file(ARRAYS) == report["arrays_sha256"]
    assert report["model_before"] == report["model_after"]
    assert [row["root"] for row in report["rows"]] == list(ROOTS)
    with np.load(ARRAYS, allow_pickle=False) as data:
        for row in report["rows"]:
            root = row["root"]
            assert image_hash(data[f"r{root}_initial_rgb"]) == row["initial_rgb_sha256"]
            assert image_hash(data[f"r{root}_goal_rgb"]) == row["goal_rgb_sha256"]
            assert [(b["method"], b["metric"]) for b in row["branches"]] == [
                (method, metric) for method in ("spectral", "random", "axis")
                for metric in ("pooled", "spatial")]
            for branch in row["branches"]:
                key = f"r{root}_{branch['method']}_{branch['metric']}"
                assert image_hash(data[f"{key}_rgb5"]) == branch["rgb5_sha256"]
                assert image_hash(data[f"{key}_rgb10"]) == branch["rgb10_sha256"]
                assert np.argmin(branch[f"prefix_{branch['metric']}_costs"]) == branch["prefix_index"]
                assert np.argmin(branch[f"tail_{branch['metric']}_costs"]) == branch["tail_index"]
                np.testing.assert_array_equal(branch["tail_action"], report["tail_bank"][branch["tail_index"]])
                distance = np.linalg.norm(np.asarray(branch["endpoint_eval_only"][:2]) -
                                          np.asarray(row["goal_position_eval_only"]))
                assert distance == pytest.approx(branch["distance_eval_only"], abs=1e-12)
