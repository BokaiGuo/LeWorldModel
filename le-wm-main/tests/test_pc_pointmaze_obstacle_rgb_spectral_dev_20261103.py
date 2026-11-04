"""Saved-data checks only; does not repeat CUDA inference or physical replay."""
import hashlib
import json

import numpy as np

from experiments.pc_pointmaze_obstacle_rgb_spectral_dev_20261103 import (
    ARRAYS, DATA, GOAL, OUTPUT, PANEL, PROTOCOL, ROOTS, sha256_file,
)


def test_saved_obstacle_dev_bindings():
    report = json.loads(OUTPUT.read_text())
    assert report["protocol_sha256"] == sha256_file(PROTOCOL)
    assert report["states_sha256"] == sha256_file(DATA / "states.pth")
    assert report["panel_sha256"] == sha256_file(PANEL)
    assert report["arrays_sha256"] == sha256_file(ARRAYS)
    assert report["model_before"] == report["model_after"]
    assert report["physical_executions"] == report["rgb_replays"] == report["position_evaluations"] == 8
    assert [r["root"] for r in report["rows"]] == list(ROOTS)
    with np.load(ARRAYS, allow_pickle=False) as saved:
        for i, row in enumerate(report["rows"]):
            assert [b["arm"] for b in row["branches"]] == ["random", "spectral"]
            for j, branch in enumerate(row["branches"]):
                key = f"r{row['root']}_{branch['arm']}"
                frames, actions = saved[f"{key}_frames"], saved[f"{key}_actions"]
                assert frames.shape == (26, 224, 224, 3)
                assert actions.shape == (25, 2) and np.isfinite(actions).all()
                assert np.max(np.abs(actions)) <= 1
                assert branch["physical_steps"] == 25
                assert len(branch["decisions"]) == 5
                for k, decision in enumerate(branch["decisions"]):
                    assert decision["input_rgb_sha256"] == hashlib.sha256(frames[5*k].tobytes()).hexdigest()
                    assert decision["output_rgb_sha256"] == hashlib.sha256(frames[5*k+5].tobytes()).hexdigest()
                    assert (decision["prediction_batches"], decision["prediction_samples"]) == (4, 50)
                    assert decision["jvp_calls"] == (2 if j else 0)
                    assert decision["svd_calls"] == (1 if j else 0)
                end = saved["endpoints_eval_only"][2*i+j]
                np.testing.assert_array_equal(end, branch["endpoint_eval_only"])
                distance = np.linalg.norm(end[:2] - GOAL[:2])
                assert abs(distance - branch["distance_eval_only"]) < 1e-12
                assert bool(distance <= .25) == branch["offline_endpoint_le_025"]
