"""Saved candidate/score/endpoint arithmetic, not a physical rerun."""
import json

import numpy as np

from experiments.a0_pointmaze_extra_cem_failure_oracle_20261103 import (
    ARRAYS, OUTPUT, SOURCE, pairwise_ranking, sha256_file,
)


def test_saved_oracle_bindings():
    report, source = (json.loads(p.read_text()) for p in (OUTPUT, SOURCE))
    assert sha256_file(ARRAYS) == report["arrays_sha256"]
    assert report["physical_sequences"] == 624
    source_rows = {row["root"]: row for row in source["rows"]}
    with np.load(ARRAYS, allow_pickle=False) as saved:
        for row in report["rows"]:
            key = f"r{row['root']}_{row['method']}_{row['phase']}"
            score, distance = saved[f"{key}_scores"], saved[f"{key}_distances"]
            candidates, endpoints = saved[f"{key}_candidates"], saved[f"{key}_endpoints"]
            target = np.asarray(source_rows[row["root"]]["goal_position_eval_only"])
            np.testing.assert_allclose(distance, np.linalg.norm(endpoints[:, :2] - target, axis=1))
            assert len(score) == len(candidates) == row["n"]
            assert int(np.argmin(score)) == row["chosen_index"]
            assert int(np.argmin(distance)) == row["oracle_index"]
            assert row["chosen_distance"] == distance[row["chosen_index"]]
            assert row["oracle_distance"] == distance[row["oracle_index"]]
            assert row["successful_candidates"] == int(np.sum(distance < .5))
            assert row["prediction_vs_physical"] == pairwise_ranking(score, distance)
            if row["phase"] == "tail":
                branch = next(b for b in source_rows[row["root"]]["branches"] if b["method"] == row["method"])
                np.testing.assert_array_equal(endpoints[row["chosen_index"], :2],
                                              np.asarray(branch["endpoint_eval_only"])[:2])
