import json
from pathlib import Path

import numpy as np
import pytest

from experiments import pc_pointmaze_affine_common_suffix_20261103 as exp
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import sha256_file


@pytest.mark.parametrize("bad", [[1.01, 0], [float("nan"), 0], [0], [0, 0, 0]])
def test_invalid_action_fails_closed(bad):
    with pytest.raises(ValueError):
        exp.validate_action(bad)


def test_saved_matrix_and_physical_outcome_calculation():
    if not exp.OUTPUT.exists():
        pytest.skip("physical development panel unavailable")
    report = json.loads(exp.OUTPUT.read_text())
    assert report["script_sha256"] == sha256_file(Path(exp.__file__))
    assert report["protocol_sha256"] == sha256_file(exp.PROTOCOL)
    assert report["arrays_sha256"] == sha256_file(exp.ARRAYS)
    assert report["trajectory_count"] == 168 and report["root_count"] == 8
    rows = report["rows"]
    index = {(r["root"], r["prefix"], r["suffix"]): i for i, r in enumerate(rows)}
    assert len(index) == 168
    with np.load(exp.ARRAYS, allow_pickle=False) as arrays:
        assert arrays["actions"].shape == (168, 15, 2)
        for i, r in enumerate(rows):
            key = (r["root"], r["prefix"], r["suffix"])
            passive = index[(r["root"], r["prefix"], "zero")]
            zero_prefix = index[(r["root"], "zero", r["suffix"])]
            for name in ("starts", "cp5", "cp10", "cp15"):
                assert np.isfinite(arrays[name][i]).all()
            np.testing.assert_array_equal(arrays["actions"][i, :5],
                                          np.tile(r["prefix_action"], (5, 1)))
            np.testing.assert_array_equal(arrays["actions"][i, 5:],
                                          np.tile(r["suffix_action"], (10, 1)))
            target = np.asarray(r["goal_eval_only"])
            start = np.linalg.norm(arrays["starts"][i, :2] - target)
            cp5 = np.linalg.norm(arrays["cp5"][i, :2] - target)
            cp5zero = np.linalg.norm(arrays["cp5"][zero_prefix, :2] - target)
            assert r["prefix_delta_from_start"] == pytest.approx(cp5 - start)
            assert r["prefix_delta_from_zero_prefix"] == pytest.approx(cp5 - cp5zero)
            assert r["non_immediate_position_progress"] == bool(cp5 - start >= -1e-8)
            for h, name in (("5", "cp10"), ("10", "cp15")):
                expected = arrays[name][i]
                zero = arrays[name][passive]
                np.testing.assert_array_equal(r["horizons"][h]["state"], expected)
                np.testing.assert_allclose(r["horizons"][h]["position_response_from_own_passive"],
                                           expected[:2] - zero[:2], rtol=0, atol=0)
                assert r["horizons"][h]["delta_from_own_passive_goal_error"] == pytest.approx(
                    np.linalg.norm(expected[:2] - target) - np.linalg.norm(zero[:2] - target))
