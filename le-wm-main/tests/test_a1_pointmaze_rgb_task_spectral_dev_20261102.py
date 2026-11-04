import json

import numpy as np
import pytest

from experiments.a1_pointmaze_rgb_task_spectral_dev_20261102 import (
    ARRAYS, OUTPUT, ROOTS, image_hash, marker, task_mask,
)
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import sha256_file


def make_image(x=80, y=80):
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    image[y:y+5, x:x+5, 1] = 200
    return image


def test_rgb_mask_fail_closed():
    assert marker(make_image()) == (82.0, 82.0)
    mask, centers = task_mask(make_image(), make_image(130, 100))
    assert mask.shape == (16, 16) and mask.sum() > 9 and len(centers) == 2
    for image in (np.zeros((224, 224, 3), dtype=np.uint8), make_image(0, 0)):
        with pytest.raises(ValueError):
            marker(image)
    bad = make_image()
    bad[120:125, 120:125, 1] = 200
    with pytest.raises(ValueError):
        marker(bad)


def test_saved_result_and_weighted_jacobians():
    if not OUTPUT.exists():
        pytest.skip("development run absent")
    report = json.loads(OUTPUT.read_text())
    assert sha256_file(ARRAYS) == report["arrays_sha256"]
    assert report["model_before"] == report["model_after"]
    assert [r["root"] for r in report["rows"]] == list(ROOTS)
    with np.load(ARRAYS, allow_pickle=False) as data:
        for row in report["rows"]:
            root = row["root"]
            init, goal = data[f"r{root}_initial_rgb"], data[f"r{root}_goal_rgb"]
            mask, _ = task_mask(init, goal)
            np.testing.assert_array_equal(mask, data[f"r{root}_initial_mask"])
            jac = data[f"r{root}_jacobian"]
            full_s = np.linalg.svd(jac.reshape(-1, 2), compute_uv=False)
            task_s = np.linalg.svd(jac[0, mask, :, :].reshape(-1, 2), compute_uv=False)
            np.testing.assert_allclose(full_s, row["full_singular_values"], rtol=1e-4)
            np.testing.assert_allclose(task_s, row["task_singular_values"], rtol=1e-4)
            for branch in row["branches"]:
                key = f"r{root}_{branch['method']}"
                rgb5, rgb10 = data[f"{key}_rgb5"], data[f"{key}_rgb10"]
                assert image_hash(rgb5) == branch["rgb5_sha256"]
                assert image_hash(rgb10) == branch["rgb10_sha256"]
                feedback_mask, _ = task_mask(rgb5, goal)
                np.testing.assert_array_equal(feedback_mask, data[f"{key}_feedback_mask"])
                assert np.argmin(branch["prefix_scores"]) == branch["prefix_index"]
                assert np.argmin(branch["tail_scores"]) == branch["tail_index"]
                distance = np.linalg.norm(np.array(branch["endpoint_eval_only"][:2]) -
                                          np.array(row["goal_position_eval_only"]))
                assert distance == pytest.approx(branch["distance_eval_only"], abs=1e-12)
            for i, first in enumerate(row["branches"]):
                for second in row["branches"][i + 1:]:
                    a = first["prefix_bank"][first["prefix_index"]]
                    b = second["prefix_bank"][second["prefix_index"]]
                    if a == b:
                        assert first["rgb5_sha256"] == second["rgb5_sha256"]
