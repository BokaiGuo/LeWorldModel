import json

import numpy as np

from experiments.a0_pointmaze_rgb_initial_real_successor_20261103 import (
    ARRAYS, OUTPUT, PROTOCOL, SOURCE, SOURCE_SHA, WORKER, pairwise_ranking, sha256_file,
)


def test_saved_scores_and_physical_bindings():
    assert OUTPUT.exists() and ARRAYS.exists(), "real-successor run incomplete"
    result, source = json.loads(OUTPUT.read_text()), json.loads(SOURCE.read_text())
    assert sha256_file(SOURCE) == SOURCE_SHA == result["source_sha256"]
    assert sha256_file(PROTOCOL) == result["protocol_sha256"]
    assert sha256_file(WORKER) == result["worker_sha256"]
    assert sha256_file(ARRAYS) == result["arrays_sha256"]
    assert result["model_before"] == result["model_after"]
    assert result["position_exact_match_count"] == 384
    with np.load(ARRAYS, allow_pickle=False) as observed, np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as bank:
        assert observed["rgb10"].shape == (384, 224, 224, 3)
        for i, (row, prior) in enumerate(zip(result["rows"], source["rows"], strict=True)):
            assert (row["root"], row["method"]) == (prior["root"], prior["method"])
            key = f"r{row['root']}_{row['method']}"
            real, prediction, distance = observed[f"{key}_real_cost"], bank[f"{key}_prediction"], bank[f"{key}_distance"]
            np.testing.assert_array_equal(real, row["actual_rgb_score"])
            np.testing.assert_array_equal(observed["legacy_endpoints"][i*48:(i+1)*48, :2],
                                          bank[f"{key}_endpoints"][:, :2])
            assert row["model_vs_real_rgb"] == pairwise_ranking(prediction, real)
            assert row["real_rgb_vs_physical"] == pairwise_ranking(real, distance)
            assert row["model_vs_physical"] == pairwise_ranking(prediction, distance)
            assert row["model_selected_index"] == int(np.argmin(prediction))
            assert row["actual_rgb_selected_index"] == int(np.argmin(real))
            assert row["physical_oracle_index"] == int(np.argmin(distance))
            assert row["real_rgb_choice_distance"] == distance[np.argmin(real)]
