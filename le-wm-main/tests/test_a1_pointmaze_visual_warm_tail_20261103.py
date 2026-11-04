import json

import numpy as np
import pytest

from experiments.a0_pointmaze_physical_cem_warm_tail_20261103 import warm_cem
from experiments.a1_pointmaze_task_spectral_cem_dev_20261102 import cem
from experiments.a1_pointmaze_visual_warm_tail_20261103 import (
    ARRAYS, OUTPUT, SOURCE, SOURCE_SHA, image_hash, sha256_file,
)


def test_saved_candidates_costs_and_source_bindings():
    if not OUTPUT.exists():
        pytest.skip("visual physical run not complete")
    result, source = json.loads(OUTPUT.read_text()), json.loads(SOURCE.read_text())
    assert result["source_sha256"] == SOURCE_SHA == sha256_file(SOURCE)
    assert result["arrays_sha256"] == sha256_file(ARRAYS)
    assert result["model_before"] == result["model_after"] == source["model_before"]
    with np.load(ARRAYS, allow_pickle=False) as arrays, np.load(SOURCE.with_suffix(".npz")) as old_arrays:
        for row, prior in zip(result["rows"], source["rows"]):
            root = row["root"]
            assert root == prior["root"]
            for index, branch in enumerate(row["branches"]):
                parent = prior["branches"][index // 2]
                assert branch["initial_method"] == parent["method"]
                assert branch["tail_prediction"] == {"model_prediction_batches": 3, "model_prediction_samples": 48}
                assert branch["tail_predicted_block_transitions"] == 48
                assert branch["initial_prediction_samples"] == 48
                assert branch["feedback_encoder_calls"] == 1
                assert branch["physical_steps"] == 10
                assert branch["rgb5_sha256"] == parent["rgb5_sha256"]
                key = f"r{root}_{branch['initial_method']}_{branch['policy']}"
                assert image_hash(arrays[f"{key}_rgb10"]) == branch["rgb10_sha256"]
                np.testing.assert_array_equal(arrays[f"{key}_feedback_mask"],
                                              old_arrays[f"r{root}_{parent['method']}_feedback_mask"])
                np.testing.assert_allclose(np.linalg.norm(np.asarray(branch["position_endpoint_eval_only"])[:2]
                                                         - row["goal_position_eval_only"]),
                                           branch["distance_eval_only"], rtol=0, atol=1e-12)
                trace, counter = branch["trace"], iter(branch["trace"])
                def replay(proposals):
                    saved = next(counter)
                    np.testing.assert_array_equal(proposals, np.asarray(saved["proposals"], np.float32))
                    return np.asarray(saved["scores"])
                noise = old_arrays[f"r{root}_tail_noise"]
                if branch["policy"] == "cold":
                    plan, cost, _ = cem(noise, replay)
                    assert branch["rgb10_sha256"] == parent["rgb10_sha256"]
                else:
                    retained = np.asarray(parent["initial_plan"], np.float32)[1:2]
                    np.testing.assert_array_equal(trace[0]["proposals"][0], retained)
                    plan, cost, _ = warm_cem(noise, retained, replay)
                    assert cost <= trace[0]["scores"][0]
                np.testing.assert_array_equal(plan, np.asarray(branch["tail"], np.float32))
                assert cost == branch["tail_cost"]
