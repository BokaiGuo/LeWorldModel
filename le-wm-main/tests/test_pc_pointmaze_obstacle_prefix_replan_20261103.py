"""Verify saved actions, RGB bindings and accounting; no physical rerun."""
import hashlib
import json

import numpy as np

from experiments.pc_pointmaze_obstacle_prefix_replan_20261103 import (
    ARMS, ARRAYS, GOAL, OUTPUT, PROTOCOL, ROOTS, SOURCE, SOURCE_SHA, sha256_file,
)


def test_prefix_replan_receipt():
    report = json.loads(OUTPUT.read_text())
    assert sha256_file(SOURCE) == report["source_sha256"] == SOURCE_SHA
    assert sha256_file(PROTOCOL) == report["protocol_sha256"]
    assert sha256_file(ARRAYS) == report["arrays_sha256"]
    assert report["model_before"] == report["model_after"]
    assert report["physical_executions"] == report["rgb_replays"] == 12
    assert [r["root"] for r in report["rows"]] == list(ROOTS)
    with np.load(ARRAYS, allow_pickle=False) as data, np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as source:
        for i, row in enumerate(report["rows"]):
            assert [x["prefix"] for x in row["branches"]] == list(ARMS)
            for j, branch in enumerate(row["branches"]):
                key = f"r{row['root']}_{branch['prefix']}"
                actions, frames = data[f"{key}_actions"], data[f"{key}_frames"]
                assert actions.shape == (25, 2) and frames.shape == (26, 224, 224, 3)
                assert np.isfinite(actions).all() and (np.abs(actions) <= 1).all()
                prefix = np.zeros((5, 2)) if j == 2 else source[f"{key}_actions"][:5]
                np.testing.assert_array_equal(actions[:5], prefix)
                assert branch["physical_steps"] == 25 and len(branch["decisions"]) == 4
                for k, decision in enumerate(branch["decisions"], start=1):
                    assert decision["input_rgb_sha256"] == hashlib.sha256(frames[5*k].tobytes()).hexdigest()
                    assert decision["output_rgb_sha256"] == hashlib.sha256(frames[5*k+5].tobytes()).hexdigest()
                    assert (decision["prediction_batches"], decision["prediction_samples"]) == (4, 50)
                cp5, cp25 = data["cp5_eval_only"][3*i+j], data["cp25_eval_only"][3*i+j]
                assert np.isclose(branch["cp5_distance"], np.linalg.norm(cp5[:2]-GOAL[:2]))
                dist = np.linalg.norm(cp25[:2]-GOAL[:2])
                assert np.isclose(branch["endpoint_distance"], dist)
                assert branch["offline_endpoint_le_025"] == bool(dist <= .25)
                if j == 0:
                    np.testing.assert_array_equal(actions, source[f"{key}_actions"])
                    np.testing.assert_array_equal(frames, source[f"{key}_frames"])
