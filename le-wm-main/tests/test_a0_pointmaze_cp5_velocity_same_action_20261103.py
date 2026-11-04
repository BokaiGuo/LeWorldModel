"""Receipt and input-binding checks, not a simulator or CUDA rerun."""
import json

import numpy as np

from experiments.a0_pointmaze_cp5_velocity_same_action_20261103 import (
    ARRAYS, CAMERA, OUTPUT, PLAN_SOURCE, PREV, REAL, SOURCE, causal_state,
    pairwise_ranking, sha256_file,
)


def test_velocity_ablation_saved_bindings():
    report = json.loads(OUTPUT.read_text())
    assert sha256_file(ARRAYS) == report["arrays_sha256"]
    assert report["model_before"] == report["model_after"]
    camera = json.loads(CAMERA.read_text())["metadata"]
    offset = json.loads(PLAN_SOURCE.read_text())["body_offset_from_static_scene"]
    with np.load(ARRAYS, allow_pickle=False) as data, \
            np.load(PREV.with_suffix(".npz"), allow_pickle=False) as prev, \
            np.load(REAL.with_suffix(".npz"), allow_pickle=False) as actual, \
            np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as bank:
        assert data["rgb4"].shape == (384, 224, 224, 3)
        for i, row in enumerate(report["rows"]):
            key = f"r{row['root']}_{row['method']}"
            privileged = data[f"{key}_privileged_input"]
            causal = data[f"{key}_causal_input"]
            zero = data[f"{key}_zero_input"]
            np.testing.assert_array_equal(zero[:, :2], causal[:, :2])
            np.testing.assert_array_equal(zero[:, :2], privileged[:, :2])
            np.testing.assert_array_equal(zero[:, 2:], 0)
            np.testing.assert_allclose(privileged[:, 2:], data["midstate_privileged"][i*48:(i+1)*48, 2:],
                                       atol=3e-7, rtol=1e-6)
            for j in range(48):
                idx = i*48+j
                expected = causal_state(data["rgb4"][idx], prev["rgb5"][idx], camera, offset, .05)
                np.testing.assert_array_equal(causal[j], expected)
            for arm in ("zero", "causal", "privileged"):
                score = data[f"{key}_{arm}_cost"]
                mse = data[f"{key}_{arm}_mse"]
                stats = row["arms"][arm]
                assert stats["vs_real_rgb"] == pairwise_ranking(score, actual[f"{key}_real_cost"])
                assert stats["oracle_choice"] == int(np.argmin(score))
                assert stats["oracle_distance"] == bank[f"{key}_distance"][stats["oracle_choice"]]
                assert np.isclose(stats["masked_mse_mean"], mse.mean())
            np.testing.assert_allclose(data[f"{key}_zero_cost"], prev[f"{key}_refresh_cost"],
                                       rtol=1e-4, atol=1e-6)
