import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.pc_pointmaze_affine_feedback_cem_20261103 import (
    ARRAYS, OUTPUT, PROTOCOL, projected_cost,
)
from experiments import pc_pointmaze_affine_feedback_cem_20261103 as feedback
from experiments.pc_pointmaze_affine_inverse_20261103 import project_tokens
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import sha256_file


def test_projection_matches_fixed_bank_implementation():
    rng = np.random.default_rng(17)
    weight = rng.normal(size=(16, 4))
    predicted = rng.normal(size=(48, 256, 16))
    goal = rng.normal(size=(256, 16))
    std = np.array([0.5, 1.2, 2.0, 3.0])
    expected = project_tokens(predicted, goal, weight, std)[0]
    actual = projected_cost(torch.from_numpy(predicted), torch.from_numpy(goal),
                            torch.from_numpy(np.linalg.pinv(weight)), torch.from_numpy(std[:2]))
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("which", range(4))
def test_projection_rejects_nonfinite_inputs(which):
    values = [torch.ones(16, 256, 16), torch.ones(256, 16), torch.ones(4, 16), torch.ones(2)]
    values[which].view(-1)[0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        projected_cost(*values)


def test_projection_rejects_nonpositive_scale_and_bad_shape():
    with pytest.raises(ValueError):
        projected_cost(torch.ones(16, 255, 16), torch.ones(256, 16), torch.ones(4, 16), torch.ones(2))
    with pytest.raises(ValueError):
        projected_cost(torch.ones(16, 256, 16), torch.ones(256, 16), torch.ones(4, 16), torch.zeros(2))


def test_saved_feedback_result_binding_and_traces():
    if not OUTPUT.exists() or not ARRAYS.exists():
        pytest.skip("development feedback run not available")
    report = json.loads(OUTPUT.read_text())
    assert report["arrays_sha256"] == sha256_file(ARRAYS)
    assert report["protocol_sha256"] == sha256_file(PROTOCOL)
    assert report["script_sha256"] == sha256_file(Path(feedback.__file__))
    assert report["model_before"] == report["model_after"]
    assert [row["root"] for row in report["rows"]] == list(range(1888, 1896))
    assert report["physical_executions"] == 8
    with np.load(ARRAYS, allow_pickle=False) as arrays:
        for index, row in enumerate(report["rows"]):
            root = row["root"]
            assert row["prediction_counts"] == {
                "initial_samples": 64, "tail_samples": 48, "initial_batches": 4, "tail_batches": 3
            }
            for phase, plan_key in (("initial", "initial_plan"), ("tail", "tail")):
                entries = row[phase + "_trace"]
                best = min((float(score), proposal)
                           for entry in entries
                           for score, proposal in zip(entry["scores"], entry["proposals"]))
                assert row[phase + "_cost"] == pytest.approx(best[0])
                np.testing.assert_allclose(row[plan_key], best[1], rtol=0, atol=0)
            np.testing.assert_array_equal(arrays["sequences"][index, :5],
                                          np.repeat(np.asarray(row["initial_plan"])[0:1], 5, axis=0))
            np.testing.assert_array_equal(arrays["sequences"][index, 5:],
                                          np.repeat(np.asarray(row["tail"])[0:1], 5, axis=0))
            np.testing.assert_array_equal(arrays["endpoints_eval_only"][index], row["endpoint_eval_only"])
            assert arrays[f"r{root}_history"].shape[0] == 6
            assert row["distance_eval_only"] == pytest.approx(np.linalg.norm(
                arrays["endpoints_eval_only"][index, :2] - row["goal_position_eval_only"]))
