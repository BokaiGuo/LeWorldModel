import numpy as np
import pytest
import json

from experiments.a0_pointmaze_cem_all_candidates_20261102 import (
    ARRAYS, OUTPUT, SOURCE, SOURCE_SHA, image_hash, pairwise, sha256_file,
)


def test_pairwise_counts_order_and_ties():
    x = np.arange(48)
    assert pairwise(x, x) == {"reversed": 0, "ties": 0, "pairs": 1128}
    assert pairwise(x, -x) == {"reversed": 1128, "ties": 0, "pairs": 1128}
    assert pairwise(x, np.zeros(48)) == {"reversed": 0, "ties": 1128, "pairs": 1128}


def test_pairwise_rejects_nonfinite_and_wrong_shape():
    with pytest.raises(ValueError):
        pairwise(np.zeros(47), np.zeros(47))
    with pytest.raises(ValueError):
        pairwise(np.full(48, np.nan), np.zeros(48))


def test_sealed_results_match_source_and_saved_images():
    if not OUTPUT.exists():
        pytest.skip("physical run not yet available")
    report = json.loads(OUTPUT.read_text())
    source = json.loads(SOURCE.read_text())
    assert sha256_file(SOURCE) == report["source_sha256"] == SOURCE_SHA
    assert sha256_file(ARRAYS) == report["arrays_sha256"]
    assert report["model_before"] == report["model_after"] == source["model_before"]
    with np.load(ARRAYS, allow_pickle=False) as arrays:
        assert len(report["rows"]) == len(source["rows"]) == 4
        for row, old in zip(report["rows"], source["rows"]):
            assert row["root"] == old["root"]
            for branch, previous in zip(row["branches"], old["branches"]):
                assert branch["method"] == previous["method"]
                key = f"r{row['root']}_{branch['method']}"
                proposals = arrays[f"{key}_proposals"]
                expected = np.asarray([t["proposals"] for t in previous["initial_trace"]],
                                      np.float32).reshape(48, 2, 2)
                assert np.array_equal(proposals, expected)
                assert [image_hash(x) for x in arrays[f"{key}_rgb10"]] == branch["rgb10_sha256"]
                endpoints = arrays[f"{key}_endpoints"]
                distances = np.linalg.norm(endpoints[:, :2] - row["goal_position_eval_only"], axis=1)
                np.testing.assert_allclose(distances, branch["distance_eval_only"], rtol=0, atol=1e-12)
                assert pairwise(branch["prediction"], branch["actual_rgb_cost"]) == branch["prediction_vs_actual_rgb"]
                assert pairwise(branch["actual_rgb_cost"], distances) == branch["actual_rgb_vs_physical"]
