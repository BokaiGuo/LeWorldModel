import json

import numpy as np
import pytest

from experiments.a0_pointmaze_rgb_initial_candidate_oracle_20261103 import (
    ARRAYS, NOISE_SHA, NOISE_SOURCE, OUTPUT, PROTOCOL, SOURCE, SOURCE_SHA, pairwise_ranking, sha256_file,
)


def test_pairwise_ranking_respects_ties_and_direction():
    assert pairwise_ranking([0, 1, 2], [2, 1, 0]) == {
        "comparable": 3, "reversed": 3, "prediction_ties": 0, "outcome_ties": 0}
    assert pairwise_ranking([0, 0, 1], [0, 1, 1]) == {
        "comparable": 1, "reversed": 0, "prediction_ties": 1, "outcome_ties": 1}
    with pytest.raises(ValueError):
        pairwise_ranking([float("nan")], [0])


def test_sealed_candidate_binding_and_outcome_arithmetic():
    assert OUTPUT.exists() and ARRAYS.exists(), "physical candidate replay incomplete"
    report, source = json.loads(OUTPUT.read_text()), json.loads(SOURCE.read_text())
    assert sha256_file(SOURCE) == SOURCE_SHA == report["source_sha256"]
    assert sha256_file(NOISE_SOURCE) == NOISE_SHA == report["noise_source_sha256"]
    assert sha256_file(PROTOCOL) == report["protocol_sha256"]
    assert sha256_file(ARRAYS) == report["arrays_sha256"]
    expected = [(r["root"], b) for r in source["rows"] for b in r["branches"] if b["policy"] == "rgb_position"]
    assert len(report["rows"]) == len(expected) == 8
    with np.load(ARRAYS, allow_pickle=False) as data:
        for item, (root, branch) in zip(report["rows"], expected, strict=True):
            assert (item["root"], item["method"]) == (root, branch["method"])
            key = f"r{root}_{branch['method']}"
            plans, score = data[f"{key}_plans"], data[f"{key}_prediction"]
            endpoints, distances = data[f"{key}_endpoints"], data[f"{key}_distance"]
            np.testing.assert_array_equal(plans, np.asarray(
                [t["proposals"] for t in branch["initial_trace"]], np.float32).reshape(48, 2, 2))
            np.testing.assert_array_equal(score, np.asarray(
                [t["scores"] for t in branch["initial_trace"]]).reshape(48))
            np.testing.assert_allclose(distances, np.linalg.norm(
                endpoints[:, :2] - item["goal_position_eval_only"], axis=1), rtol=0, atol=1e-12)
            selected, oracle = int(np.argmin(score)), int(np.argmin(distances))
            assert (item["selected_index"], item["oracle_index"]) == (selected, oracle)
            assert item["selected_distance"] == distances[selected]
            assert item["oracle_distance"] == distances[oracle]
            assert item["selected_regret"] == distances[selected] - distances[oracle]
            assert item["prediction_physics_rank"] == pairwise_ranking(score, distances)
