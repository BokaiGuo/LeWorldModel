import json

import numpy as np
import pytest

from experiments.a0_pointmaze_cem_action_response_20261103 import (
    ARRAYS, OUTPUT, PREV_SHA, SOURCE_SHA, decompose, sha256_file,
)


def test_decomposition_known_gain_and_mean_offsets():
    rng = np.random.default_rng(21)
    x = rng.normal(size=(48, 9))
    direction = rng.normal(size=9)
    result = decompose(x + 17, 2*x - 8, direction)
    assert result["centered_cosine"] == pytest.approx(1)
    assert result["real_on_predicted_gain"] == pytest.approx(2)
    assert result["predicted_to_real_norm"] == pytest.approx(.5)
    assert result["centered_relative_squared_error"] == pytest.approx(.25)
    with pytest.raises(ValueError):
        decompose(np.zeros((48, 9)), np.zeros((48, 9)), direction)


def test_saved_decomposition_recomputes_from_sealed_features():
    if not OUTPUT.exists():
        pytest.skip("CUDA diagnostic not complete")
    result = json.loads(OUTPUT.read_text())
    assert result["previous_sha256"] == PREV_SHA
    assert result["source_sha256"] == SOURCE_SHA
    assert result["arrays_sha256"] == sha256_file(ARRAYS)
    assert result["model_before"] == result["model_after"]
    assert len(result["rows"]) == 4
    with np.load(ARRAYS, allow_pickle=False) as arrays:
        for row in result["rows"]:
            assert len(row["branches"]) == 2
            for branch in row["branches"]:
                key = f"r{row['root']}_{branch['method']}"
                recomputed = decompose(arrays[f"{key}_predicted_features"], arrays[f"{key}_actual_features"],
                                       arrays[f"{key}_goal_direction"])
                for name, expected in recomputed.items():
                    np.testing.assert_allclose(branch[name], expected, rtol=1e-12, atol=1e-12)
