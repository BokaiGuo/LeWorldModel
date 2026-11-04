"""Validate saved execution bindings; does not rerun model or physics."""
import json

import numpy as np
import torch

from experiments.a1_pointmaze_spectral_cp5_replan_20261102 import image_hash
from experiments.pc_ap_pointmaze_rgb_dev_20261103 import (
    ARRAYS, DEFAULT_DATASET, MEMORY, OUTPUT, PROTOCOL, ROOTS, SOURCE, sha256_file,
)


def test_saved_ap_rgb_and_actions():
    report = json.loads(OUTPUT.read_text())
    assert sha256_file(ARRAYS) == report["arrays_sha256"]
    assert sha256_file(PROTOCOL) == report["protocol_sha256"]
    assert sha256_file(SOURCE) == report["source_sha256"]
    assert report["model_before"] == report["model_after"]
    assert report["memory_frames_encoded"] == 63
    assert report["physical_executions"] == report["verification_replays"] == 4
    assert [row["root"] for row in report["rows"]] == list(ROOTS)
    memory = {}
    for episode in MEMORY:
        path = DEFAULT_DATASET / "obses" / f"episode_{episode:03d}.pth"
        assert sha256_file(path) == report["memory_archive_sha256"][str(episode)]
        memory[str(episode)] = torch.load(path, weights_only=True, map_location="cpu").numpy()
    with np.load(ARRAYS, allow_pickle=False) as data:
        for i, row in enumerate(report["rows"]):
            assert [b["arm"] for b in row["branches"]] == ["memory", "final_goal"]
            for j, branch in enumerate(row["branches"]):
                key = f"r{row['root']}_{branch['arm']}"
                frames, actions = data[f"{key}_frames"], data[f"{key}_actions"]
                assert frames.shape == (11, 224, 224, 3)
                assert actions.shape == (10, 2) and np.isfinite(actions).all()
                assert np.abs(actions).max() <= 1
                assert branch["physical_steps"] == 10
                for t, decision in enumerate(branch["decisions"]):
                    np.testing.assert_array_equal(actions[5*t:5*t+5], np.asarray(decision["actions"], np.float32))
                    assert image_hash(frames[5*t]) == decision["input_rgb_sha256"]
                    assert image_hash(frames[5*t+5]) == decision["output_rgb_sha256"]
                    assert (decision["prediction_batches"], decision["prediction_samples"]) == (4, 50)
                    hit = decision["retrieval"]
                    if branch["arm"] == "memory":
                        assert hit["episode"] in memory and hit["target_index"] == hit["start"] + 5
                        assert hit["horizon"] == 10 - 5*t
                        assert image_hash(memory[hit["episode"]][hit["target_index"]]) == hit["target_rgb_sha256"]
                    else:
                        assert hit is None
                endpoint = data["endpoints_eval_only"][2*i+j]
                np.testing.assert_array_equal(endpoint, branch["endpoint_eval_only"])
                distance = np.linalg.norm(endpoint[:2] - np.asarray(row["goal_position_eval_only"]))
                assert distance == branch["distance_eval_only"]
                assert (distance < .5) == branch["strict_endpoint_success_eval_only"]
