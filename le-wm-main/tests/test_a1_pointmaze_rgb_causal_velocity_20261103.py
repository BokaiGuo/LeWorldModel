import json

import numpy as np
import pytest

from experiments.a0_pointmaze_rgb_geometry_state_20261103 import estimate_state
from experiments.a1_pointmaze_rgb_causal_velocity_20261103 import (
    ARRAYS, CAMERA, CAMERA_SHA, OUTPUT, POLICIES, PROTOCOL, SOURCE, SOURCE_SHA,
    causal_state, image_hash, sha256_file,
)
from experiments.a1_pointmaze_task_spectral_cem_dev_20261102 import cem


def test_causal_state_units_and_missing_marker():
    camera = json.loads(CAMERA.read_text())["metadata"]
    offset = np.array([0., 0., 0.])
    previous = np.zeros((224, 224, 3), dtype=np.uint8)
    current = previous.copy()
    previous[110:113, 110:113, 1] = 200
    current[112:115, 113:116, 1] = 200
    old, metadata = estimate_state(previous, camera, offset)
    now, _ = estimate_state(current, camera, offset)
    state = causal_state(previous, current, camera, offset, 0.05)
    np.testing.assert_allclose(state[:2], now[:2])
    np.testing.assert_allclose(state[2:], (now[:2] - old[:2]) / .05)
    np.testing.assert_allclose(state[2:], np.array([3, 2]) * metadata["world_units_per_pixel"] / .05,
                               rtol=2e-5)
    for dt in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            causal_state(previous, current, camera, offset, dt)
    with pytest.raises(ValueError):
        causal_state(previous, np.zeros_like(current), camera, offset, .05)


def test_saved_rgb_actions_costs_and_replay():
    assert OUTPUT.exists() and ARRAYS.exists(), "physical experiment output missing"
    result, source = json.loads(OUTPUT.read_text()), json.loads(SOURCE.read_text())
    camera = json.loads(CAMERA.read_text())["metadata"]
    assert result["source_sha256"] == SOURCE_SHA == sha256_file(SOURCE)
    assert result["camera_sha256"] == CAMERA_SHA == sha256_file(CAMERA)
    assert result["protocol_sha256"] == sha256_file(PROTOCOL)
    assert result["arrays_sha256"] == sha256_file(ARRAYS)
    assert result["model_before"] == result["model_after"] == source["model_before"]
    offset, dt = result["body_offset_from_static_scene"], result["observation_interval"]
    with np.load(ARRAYS, allow_pickle=False) as arrays, np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as old:
        for row, prior in zip(result["rows"], source["rows"], strict=True):
            root = row["root"]
            assert root == prior["root"]
            for parent_index, parent in enumerate(prior["branches"]):
                group = row["branches"][parent_index * 3:parent_index * 3 + 3]
                assert [b["policy"] for b in group] == list(POLICIES)
                assert all(b["initial_method"] == parent["method"] for b in group)
                assert all(b["history_sha256"] == group[0]["history_sha256"] for b in group)
                for branch in group:
                    key = f"r{root}_{parent['method']}_{branch['policy']}"
                    frames = arrays[f"{key}_history"]
                    assert frames.shape == (6, 224, 224, 3)
                    assert [image_hash(frame) for frame in frames] == branch["history_sha256"]
                    assert branch["history_sha256"][-1] == parent["rgb5_sha256"]
                    assert image_hash(arrays[f"{key}_rgb10"]) == branch["rgb10_sha256"]
                    estimate = causal_state(frames[-2], frames[-1], camera, offset, dt)
                    position, _ = estimate_state(frames[-1], camera, offset)
                    np.testing.assert_array_equal(estimate, branch["estimated_causal_state"])
                    inputs = {"zero_state": np.zeros(4), "rgb_position_zero_velocity": position,
                              "rgb_position_causal_velocity": estimate}
                    np.testing.assert_array_equal(inputs[branch["policy"]], branch["input_state"])
                    mid = np.asarray(branch["cp5_backend_state_eval_only"])
                    np.testing.assert_allclose(np.linalg.norm(estimate[:2] - mid[:2]),
                                               branch["cp5_position_error_eval_only"], atol=1e-12)
                    np.testing.assert_allclose(np.linalg.norm(estimate[2:] - mid[2:]),
                                               branch["cp5_velocity_error_eval_only"], atol=1e-12)
                    np.testing.assert_allclose(
                        np.linalg.norm(np.asarray(branch["endpoint_eval_only"])[:2] - row["goal_position_eval_only"]),
                        branch["distance_eval_only"], atol=1e-12)
                    assert branch["tail_prediction"] == {"batches": 3, "samples": 48}
                    assert branch["feedback_observation_count"] == 5
                    assert branch["feedback_encoder_calls"] == 1
                    assert branch["physical_steps"] == 10
                    counter = iter(branch["trace"])

                    def replay(proposals):
                        saved = next(counter)
                        np.testing.assert_array_equal(proposals, np.asarray(saved["proposals"], np.float32))
                        return np.asarray(saved["scores"])

                    plan, cost, _ = cem(old[f"r{root}_tail_noise"], replay)
                    np.testing.assert_array_equal(plan, np.asarray(branch["tail"], np.float32))
                    assert cost == branch["tail_cost"]
                    if branch["policy"] == "zero_state":
                        np.testing.assert_array_equal(plan, np.asarray(parent["tail_plan"], np.float32))
                        assert abs(cost - parent["tail_cost"]) < 1e-5
                        assert branch["rgb10_sha256"] == parent["rgb10_sha256"]
