import json

import numpy as np
import pytest

from experiments.a1_pointmaze_history_cem_native_endpoint_20261103 import (
    BACKEND, OUTPUT, PROTOCOL, SOURCE, SOURCE_SHA, WRAPPER, evaluate, sha256_file,
)


def test_wrapper_success_is_strict_and_rejects_invalid_states():
    assert evaluate([0, 0, 0, 0], [0.499, 0, 0, 0])["success"]
    assert not evaluate([0, 0, 0, 0], [0.5, 0, 0, 0])["success"]
    assert not evaluate([0, 0, 0, 0], [0.501, 0, 0, 0])["success"]
    assert evaluate([0, 0, 0, 0], [0, 0, 3, 4])["state_distance"] == 5
    with pytest.raises(ValueError):
        evaluate([0, 0, 0, 0], [0, 0, float("nan"), 0])


def test_saved_replay_and_metrics():
    assert OUTPUT.exists(), "native replay incomplete"
    report, source = json.loads(OUTPUT.read_text()), json.loads(SOURCE.read_text())
    assert sha256_file(SOURCE) == SOURCE_SHA == report["source_sha256"]
    assert sha256_file(PROTOCOL) == report["protocol_sha256"]
    assert sha256_file(WRAPPER) == report["wrapper_sha256"]
    assert sha256_file(BACKEND) == report["legacy_backend_sha256"]
    branches = [(row["root"], b) for row in source["rows"] for b in row["branches"]]
    assert len(report["rows"]) == len(branches) == 16
    for result, (root, branch) in zip(report["rows"], branches, strict=True):
        assert (result["root"], result["prefix_method"], result["mode"]) == (
            root, branch["prefix_method"], branch["mode"])
        assert result["rgb5_matches"] and result["rgb10_matches"] and result["saved_position_matches"]
        np.testing.assert_array_equal(result["position_state_eval_only"], branch["endpoint_eval_only"])
        assert result["wrapper_eval"] == evaluate(result["goal_eval_only"], result["legacy_state_eval_only"])
        legacy, standard = np.asarray(result["legacy_state_eval_only"]), np.asarray(result["position_state_eval_only"])
        np.testing.assert_allclose(np.linalg.norm(legacy[:2] - standard[:2]),
                                   result["legacy_minus_position_l2"], atol=1e-12)
