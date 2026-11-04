import json

import numpy as np

from experiments.a1_pointmaze_cp5_history_physical_20261102 import (
    ARRAYS, HISTORY, HISTORY_SHA, OUTPUT, SOURCE, SOURCE_SHA,
)
from experiments.a1_pointmaze_spectral_cp5_replan_20261102 import image_hash
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import sha256_file


def test_physical_result_binding_and_arithmetic():
    result, source = json.loads(OUTPUT.read_text()), json.loads(SOURCE.read_text())
    history = json.loads(HISTORY.read_text())
    assert sha256_file(SOURCE) == result["source_sha256"] == SOURCE_SHA
    assert sha256_file(HISTORY) == result["history_sha256"] == HISTORY_SHA
    assert sha256_file(ARRAYS) == result["arrays_sha256"]
    assert len(result["rows"]) == 4
    with np.load(ARRAYS, allow_pickle=False) as images:
        for row, old_row, history_row in zip(result["rows"], source["rows"], history["rows"]):
            assert row["root"] == old_row["root"] == history_row["root"]
            goal = np.asarray(old_row["goal_position_eval_only"])
            for branch, old, choice in zip(row["branches"], old_row["branches"], history_row["branches"]):
                assert branch["method"] == old["method"] == choice["method"]
                assert branch["source_tail"] == old["tail_index"] == choice["executed_tail"]
                assert branch["history_tail"] == choice["history_choice_not_executed"]
                assert branch["rgb5_sha256"] == old["rgb5_sha256"]
                assert branch["rgb10_sha256"] == image_hash(images[f"r{row['root']}_{branch['method']}_rgb10"])
                if branch["history_tail"] == branch["source_tail"]:
                    assert branch["rgb10_sha256"] == old["rgb10_sha256"]
                distance = np.linalg.norm(np.asarray(branch["endpoint_eval_only"][:2]) - goal)
                np.testing.assert_allclose(distance, branch["history_distance_eval_only"], rtol=0, atol=1e-12)
                np.testing.assert_allclose(distance - old["distance_eval_only"],
                                           branch["distance_change_history_minus_single"], rtol=0, atol=1e-12)
