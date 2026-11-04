import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "results/pc_spectral_prefix_exact_ceiling_20261103_CODEX.json"


def test_saved_searches_have_equal_evaluations_and_valid_elites():
    result = json.loads(RECEIPT.read_text())
    assert result["mujoco_version"] == "2.2.0"
    assert result["privileged_evaluation_only"] and len(result["rows"]) == 16
    for index, row in enumerate(result["rows"]):
        assert row["index"] == index and row["samples"] == 100
        assert np.asarray(row["optimized_raw_suffix"]).shape == (5, 2)
        assert np.max(np.abs(row["optimized_raw_suffix"])) <= 1 + 1e-12
        assert len(row["prefix_distances"]) == 6
        for search in row["searches"]:
            assert len(search["trace"]) == 3 and len(search["scores"]) == 2
            for iteration in search["trace"]:
                scores = np.asarray(iteration["scores"])
                assert len(scores) == 16
                np.testing.assert_array_equal(iteration["elite"], np.argsort(scores, kind="stable")[:4])
                assert np.asarray(iteration["samples"]).shape == (16, 10)
        assert row["optimized_distance"] == min(min(s["scores"]) for s in row["searches"])


def test_nonprogress_candidates_do_not_beat_both_controls_in_this_receipt():
    result = json.loads(RECEIPT.read_text())
    candidates = [row for row in result["rows"] if row["non_immediate_progress"]]
    assert len(candidates) == 2
    assert all(not row["beats_zero_and_greedy"] for row in candidates)
