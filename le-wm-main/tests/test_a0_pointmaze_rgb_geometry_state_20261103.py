import json

import numpy as np
import pytest

from experiments.a0_pointmaze_rgb_geometry_state_20261103 import (
    ARRAYS, CAMERA, CAMERA_SHA, OUTPUT, PROPRIO, PROPRIO_SHA, REPLAY, REPLAY_SHA,
    SOURCE, SOURCE_SHA, estimate_state, sha256_file,
)


def test_projection_and_missing_marker():
    metadata = {"azimuth": 90., "elevation": -90., "camera_distance": 10.,
                "fovy_degrees": 90., "camera_lookat": [3., 3., 0.],
                "marker_geom_relative_position": [0., 0., 0.]}
    image = np.zeros((224, 224, 3), np.uint8)
    image[110:114, 110:114, 1] = 255
    state, detail = estimate_state(image, metadata, [1.2, 1.2, 0.])
    np.testing.assert_allclose(state, [1.8, 1.8, 0, 0], atol=1e-6)
    assert detail["world_units_per_pixel"] == pytest.approx(20/224)
    with pytest.raises(ValueError):
        estimate_state(np.zeros_like(image), metadata, [1.2, 1.2, 0.])
    with pytest.raises(ValueError):
        estimate_state(image, {**metadata, "azimuth": 0.}, [1.2, 1.2, 0.])


def test_sealed_image_estimates_candidate_bindings_and_costs():
    if not OUTPUT.exists():
        pytest.skip("CUDA experiment not completed")
    result, source, replay = (json.loads(p.read_text()) for p in (OUTPUT, SOURCE, REPLAY))
    camera = json.loads(CAMERA.read_text())
    assert sha256_file(CAMERA) == result["camera_sha256"] == CAMERA_SHA
    assert sha256_file(SOURCE) == result["source_sha256"] == SOURCE_SHA
    assert sha256_file(REPLAY) == result["replay_sha256"] == REPLAY_SHA
    assert sha256_file(PROPRIO) == result["proprio_sha256"] == PROPRIO_SHA
    assert sha256_file(ARRAYS) == result["arrays_sha256"]
    assert result["model_before"] == result["model_after"] == source["model_before"]
    with np.load(ARRAYS, allow_pickle=False) as scores, np.load(SOURCE.with_suffix(".npz")) as images:
        for row, old in zip(result["rows"], replay["rows"]):
            root = row["root"]
            assert root == old["root"]
            state, _ = estimate_state(images[f"r{root}_rgb0"], camera["metadata"],
                                      result["body_offset_from_static_scene"])
            np.testing.assert_array_equal(state, np.asarray(row["rgb_state_estimate"], np.float32))
            expected_error = np.linalg.norm(state[:2]-np.asarray(row["privileged_initial_state_eval_only"])[:2])
            assert row["position_error_eval_only"] == pytest.approx(expected_error, abs=1e-6)
            for branch, previous in zip(row["branches"], old["branches"]):
                assert branch["method"] == previous["method"]
                for name in ("rgb_position_zero_velocity", "privileged_position_zero_velocity"):
                    values = scores[f"r{root}_{branch['method']}_{name}_scores"]
                    index = int(np.argmin(values))
                    assert index == branch["conditions"][name]["choice"]
                    assert previous["distance_eval_only"][index] == branch["conditions"][name]["distance_eval_only"]
