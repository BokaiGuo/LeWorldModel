"""Recompute saved bindings; no new neural inference or physical replay."""
import json

import numpy as np

from experiments.a1_pointmaze_spectral_cp5_replan_20261102 import image_hash
from experiments.pc_pointmaze_ap_spectral_cem_20261103 import ARRAYS, AP_SHA, AP_SOURCE, OUTPUT, PROTOCOL, SOURCE, sha256_file


def test_saved_spectral_physical_bindings():
    report = json.loads(OUTPUT.read_text())
    assert sha256_file(ARRAYS) == report["arrays_sha256"]
    assert sha256_file(PROTOCOL) == report["protocol_sha256"]
    assert sha256_file(SOURCE) == report["source_sha256"]
    assert sha256_file(AP_SOURCE) == report["prior_ap_sha256"] == AP_SHA
    assert report["model_before"] == report["model_after"]
    assert report["physical_executions"] == report["verification_replays"] == 16
    assert [row["root"] for row in report["rows"]] == list(range(1888, 1896))
    with np.load(ARRAYS, allow_pickle=False) as data, \
            np.load(AP_SOURCE.with_suffix(".npz"), allow_pickle=False) as prior:
        for i, row in enumerate(report["rows"]):
            root = row["root"]
            assert [branch["arm"] for branch in row["branches"]] == ["random", "spectral"]
            for j, branch in enumerate(row["branches"]):
                key = f"r{root}_{branch['arm']}"
                frames, actions = data[f"{key}_frames"], data[f"{key}_actions"]
                assert frames.shape == (11, 224, 224, 3)
                assert actions.shape == (10, 2) and np.isfinite(actions).all()
                assert np.abs(actions).max() <= 1
                for t, decision in enumerate(branch["decisions"]):
                    np.testing.assert_array_equal(actions[5*t:5*t+5], np.asarray(decision["actions"], np.float32))
                    assert image_hash(frames[5*t]) == decision["input_rgb_sha256"]
                    assert image_hash(frames[5*t+5]) == decision["output_rgb_sha256"]
                    assert (decision["prediction_batches"], decision["prediction_samples"]) == (4, 50)
                    assert (decision["jvp_calls"], decision["svd_calls"]) == (
                        (2, 1) if branch["arm"] == "spectral" else (0, 0))
                    if branch["arm"] == "spectral":
                        jac = data[f"r{root}_d{t}_jacobian"]
                        singular = data[f"r{root}_d{t}_singular"]
                        raw = data[f"r{root}_d{t}_injection_raw"]
                        direction = raw[0] / .3
                        np.testing.assert_allclose(np.linalg.norm(direction), 1, atol=1e-6)
                        np.testing.assert_allclose(np.linalg.norm(jac @ direction), singular[0], rtol=1e-5)
                        np.testing.assert_allclose(raw, np.array([.3, -.3, .6, -.6])[:, None] * direction, atol=1e-7)
                if root in (1888, 1889) and branch["arm"] == "random":
                    np.testing.assert_array_equal(actions, prior[f"r{root}_final_goal_actions"])
                    np.testing.assert_array_equal(frames, prior[f"r{root}_final_goal_frames"])
                endpoint = data["endpoints_eval_only"][2*i+j]
                np.testing.assert_array_equal(endpoint, branch["endpoint_eval_only"])
                distance = np.linalg.norm(endpoint[:2] - np.asarray(row["goal_position_eval_only"]))
                assert distance == branch["distance_eval_only"]
                assert (distance < .5) == branch["strict_endpoint_success_eval_only"]
