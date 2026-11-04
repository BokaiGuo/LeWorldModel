import json
from pathlib import Path

import numpy as np

from experiments import a1_pointmaze_native_spectral_20261102 as run
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import sha256_file


def test_candidate_construction_is_bounded_and_symmetric():
    candidates = run.directional_candidates(np.array([0.6, 0.8], dtype=np.float32))
    assert candidates.shape == (5, 2)
    np.testing.assert_allclose(candidates[1], -candidates[2])
    np.testing.assert_allclose(candidates[3], -candidates[4])
    assert np.max(np.abs(candidates)) <= 0.6


def test_saved_jacobians_and_model_selections():
    report = json.loads(run.OUTPUT.read_text())
    assert report["code_sha256"] == sha256_file(Path(run.__file__))
    assert report["protocol_sha256"] == sha256_file(run.PROTOCOL)
    assert report["model_before"] == report["model_after"]
    assert [row["root"] for row in report["rows"]] == list(run.ROOTS)
    for row in report["rows"]:
        jacobian = np.asarray(row["jacobian"])
        assert jacobian.shape == (384, 2)
        np.testing.assert_allclose(np.linalg.svd(jacobian, compute_uv=False), row["singular_values"], rtol=1e-5)
        direction = np.asarray(row["spectral_direction"])
        np.testing.assert_allclose(np.linalg.norm(jacobian @ direction), row["singular_values"][0], rtol=1e-5)
        assert [branch["method"] for branch in row["branches"]] == ["spectral", "random", "axis"]
        for branch in row["branches"]:
            assert branch["choice"] == int(np.argmin(branch["model_costs"]))
            np.testing.assert_array_equal(branch["selected_action"], branch["candidate_actions"][branch["choice"]])
            assert branch["prediction_samples"] == 5
