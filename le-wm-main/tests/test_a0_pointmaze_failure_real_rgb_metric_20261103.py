"""Saved ranking and oracle bindings; does not rerun model or simulator."""
import json

import numpy as np

from experiments.a0_pointmaze_failure_real_rgb_metric_20261103 import (
    ARRAYS, ORACLE, OUTPUT, pairwise_ranking, sha256_file,
)


def test_saved_rgb_metric_decomposition():
    report = json.loads(OUTPUT.read_text())
    assert sha256_file(ARRAYS) == report["arrays_sha256"]
    assert report["model_before"] == report["model_after"]
    assert report["position_exact_count"] == 624
    with np.load(ARRAYS, allow_pickle=False) as saved, \
            np.load(ORACLE.with_suffix(".npz"), allow_pickle=False) as oracle:
        assert saved["rgb10"].shape == (624, 224, 224, 3)
        offset = 0
        for row in report["rows"]:
            key = f"r{row['root']}_{row['method']}_{row['phase']}"
            n = row["n"]
            actual, prediction, distance = (saved[f"{key}_real_cost"], oracle[f"{key}_scores"],
                                             oracle[f"{key}_distances"])
            np.testing.assert_array_equal(saved["legacy_endpoints"][offset:offset+n, :2],
                                          oracle[f"{key}_endpoints"][:, :2])
            offset += n
            assert row["model_vs_real_rgb"] == pairwise_ranking(prediction, actual)
            assert row["real_rgb_vs_physical"] == pairwise_ranking(actual, distance)
            assert row["model_vs_physical"] == pairwise_ranking(prediction, distance)
            assert row["real_rgb_oracle_choice"] == int(np.argmin(actual))
            assert row["real_rgb_oracle_distance"] == distance[row["real_rgb_oracle_choice"]]
            assert row["model_choice_distance"] == distance[np.argmin(prediction)]
        assert offset == 624
