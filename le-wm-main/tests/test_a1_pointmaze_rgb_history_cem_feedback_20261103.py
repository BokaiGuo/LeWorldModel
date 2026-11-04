import json

import numpy as np

from experiments.a1_pointmaze_rgb_history_cem_feedback_20261103 import (
    ARRAYS, CAMERA, CAMERA_SHA, NOISE_SHA, NOISE_SOURCE, OUTPUT, PROTOCOL,
    SOURCE, SOURCE_SHA, image_hash, sha256_file,
)
from experiments.a1_pointmaze_task_spectral_cem_dev_20261102 import cem


def test_saved_prefix_feedback_and_cem_binding():
    assert OUTPUT.exists() and ARRAYS.exists(), "physical run not complete"
    result, source = (json.loads(p.read_text()) for p in (OUTPUT, SOURCE))
    assert sha256_file(SOURCE) == SOURCE_SHA == result["source_sha256"]
    assert sha256_file(NOISE_SOURCE) == NOISE_SHA == result["noise_source_sha256"]
    assert sha256_file(CAMERA) == CAMERA_SHA == result["camera_sha256"]
    assert sha256_file(PROTOCOL) == result["protocol_sha256"]
    assert sha256_file(ARRAYS) == result["arrays_sha256"]
    assert result["model_before"] == result["model_after"] == source["model_before"]
    with np.load(ARRAYS, allow_pickle=False) as images, np.load(NOISE_SOURCE.with_suffix(".npz"), allow_pickle=False) as noise:
        for row, prior in zip(result["rows"], source["rows"], strict=True):
            root = row["root"]
            assert root == prior["root"]
            assert [(b["prefix_method"], b["mode"]) for b in row["branches"]] == [
                ("random_cem", "single"), ("random_cem", "history"),
                ("spectral_cem", "single"), ("spectral_cem", "history")]
            for branch in row["branches"]:
                parent = next(b for b in prior["branches"]
                              if b["policy"] == "rgb_position" and b["method"] == branch["prefix_method"])
                np.testing.assert_array_equal(branch["prefix"], parent["initial_plan"][0])
                assert branch["rgb5_sha256"] == parent["rgb5_sha256"]
                assert branch["prediction_counts"] == {"batches": 3, "samples": 48}
                assert branch["physical_steps"] == 10 and branch["feedback_encode_calls"] == 1
                assert branch["initial_extra_encode_calls"] == int(branch["mode"] == "history")
                key = f"r{root}_{branch['prefix_method']}_{branch['mode']}"
                for point in ("rgb0", "rgb5", "rgb10"):
                    assert image_hash(images[f"{key}_{point}"]) == branch[f"{point}_sha256"]
                entries = iter(branch["trace"])

                def replay(proposals):
                    entry = next(entries)
                    np.testing.assert_array_equal(proposals, np.asarray(entry["proposals"], np.float32))
                    return np.asarray(entry["scores"])

                tail, cost, _ = cem(noise[f"r{root}_tail_noise"], replay)
                np.testing.assert_array_equal(tail, np.asarray(branch["tail"], np.float32))
                assert cost == branch["tail_cost"]
                np.testing.assert_allclose(
                    np.linalg.norm(np.asarray(branch["endpoint_eval_only"])[:2] - row["goal_position_eval_only"]),
                    branch["distance_eval_only"], rtol=0, atol=1e-12)
                if branch["mode"] == "single":
                    np.testing.assert_array_equal(tail, np.asarray(parent["tail"], np.float32))
                    assert branch["rgb10_sha256"] == parent["rgb10_sha256"]
