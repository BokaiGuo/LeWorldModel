import json

import numpy as np
import pytest

from experiments.a1_pointmaze_rgb_state_full_planning_20261103 import (
    ARRAYS, CAMERA, CAMERA_SHA, OUTPUT, PROTOCOL, SOURCE, SOURCE_SHA, image_hash,
    sha256_file, state_from_rgb,
)
from experiments.a1_pointmaze_task_spectral_cem_dev_20261102 import cem


def test_state_policy_rejects_unknown():
    with pytest.raises(ValueError, match="unknown state policy"):
        state_from_rgb(None, "privileged", None, None)
    np.testing.assert_array_equal(state_from_rgb(None, "zero_state", None, None), np.zeros(4))


def test_physical_run_saved_bindings_and_cem_replay():
    assert OUTPUT.exists() and ARRAYS.exists(), "physical run not finished"
    result, source = json.loads(OUTPUT.read_text()), json.loads(SOURCE.read_text())
    camera = json.loads(CAMERA.read_text())["metadata"]
    offset = result["body_offset_from_static_scene"]
    assert sha256_file(SOURCE) == SOURCE_SHA == result["source_sha256"]
    assert sha256_file(CAMERA) == CAMERA_SHA == result["camera_sha256"]
    assert sha256_file(PROTOCOL) == result["protocol_sha256"]
    assert sha256_file(ARRAYS) == result["arrays_sha256"]
    assert result["model_before"] == result["model_after"] == source["model_before"]
    with np.load(ARRAYS, allow_pickle=False) as arrays, np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as old:
        for row, prior in zip(result["rows"], source["rows"], strict=True):
            root = row["root"]
            assert root == prior["root"]
            assert [(b["policy"], b["method"]) for b in row["branches"]] == [
                ("zero_state", "random_cem"), ("zero_state", "spectral_cem"),
                ("rgb_position", "random_cem"), ("rgb_position", "spectral_cem")]
            for branch in row["branches"]:
                key = f"r{root}_{branch['policy']}_{branch['method']}"
                rgb0 = old[f"r{root}_rgb0"]
                rgb5, rgb10 = arrays[f"{key}_rgb5"], arrays[f"{key}_rgb10"]
                assert image_hash(rgb5) == branch["rgb5_sha256"]
                assert image_hash(rgb10) == branch["rgb10_sha256"]
                np.testing.assert_array_equal(state_from_rgb(rgb0, branch["policy"], camera, offset), branch["input0"])
                np.testing.assert_array_equal(state_from_rgb(rgb5, branch["policy"], camera, offset), branch["input5"])
                assert branch["prediction_counts"] == {
                    "initial_batches": 3, "initial_samples": 48, "tail_batches": 3, "tail_samples": 48}
                assert branch["physical_steps"] == 10 and branch["feedback_encodes"] == 1
                assert branch["jvp_calls"] == (2 if branch["method"] == "spectral_cem" else 0)
                assert branch["svd_calls"] == (1 if branch["method"] == "spectral_cem" else 0)
                for prefix, noise, injected, target in (
                    ("initial", old[f"r{root}_initial_noise"], branch["injected"], branch["initial_plan"]),
                    ("tail", old[f"r{root}_tail_noise"], None, branch["tail"]),
                ):
                    saved_trace = iter(branch[f"{prefix}_trace"])

                    def replay(proposals):
                        entry = next(saved_trace)
                        np.testing.assert_array_equal(proposals, np.asarray(entry["proposals"], np.float32))
                        return np.asarray(entry["scores"])

                    plan, cost, _ = cem(noise, replay, injected)
                    np.testing.assert_array_equal(plan, np.asarray(target, np.float32))
                    assert cost == branch[f"{prefix}_cost"]
                np.testing.assert_allclose(
                    np.linalg.norm(np.asarray(branch["endpoint_eval_only"])[:2] - row["goal_position_eval_only"]),
                    branch["distance_eval_only"], rtol=0, atol=1e-12)
                if branch["policy"] == "zero_state":
                    parent = next(b for b in prior["branches"] if b["method"] == branch["method"])
                    np.testing.assert_array_equal(branch["initial_plan"], parent["initial_plan"])
                    np.testing.assert_array_equal(branch["tail"], parent["tail_plan"])
                    assert branch["rgb5_sha256"] == parent["rgb5_sha256"]
                    assert branch["rgb10_sha256"] == parent["rgb10_sha256"]
