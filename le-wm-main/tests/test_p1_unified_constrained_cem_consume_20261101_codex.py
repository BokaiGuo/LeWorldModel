import json

import numpy as np

from experiments.spectral_reachability import p1_unified_constrained_cem_consume_20261101_codex as subject


def test_saved_search_reconstructs_and_has_two_distinct_non_greedy_programs():
    result = subject.consume()
    assert result["new_neural_calls"] == result["new_physical_calls"] == 0
    assert len(result["rows"]) == 2
    for row in result["rows"]:
        assert row["selected"]["q1"] >= row["root_mse"]
        assert row["selected"]["q3"] < row["root_mse"]
        assert row["selected_is_distinct_from_greedy"]
        selected = np.asarray(row["candidates"]["selected"], np.float32)
        zero = np.asarray(row["candidates"]["zero_prefix"], np.float32)
        greedy = np.asarray(row["candidates"]["same_bank_greedy"], np.float32)
        assert selected.shape == zero.shape == greedy.shape == (3, 5, 2)
        assert not zero[0].any()
        np.testing.assert_array_equal(zero[1:], selected[1:])


def test_consumed_artifacts_are_json_serializable():
    json.dumps(subject.consume(), allow_nan=False)
