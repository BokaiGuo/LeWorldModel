import copy

import numpy as np
import pytest

from experiments.pc_pointmaze_live_observation_capture_20261107_codex import array_hash, compare


def fixture():
    frames = np.zeros((8, 5, 224, 224, 3), dtype=np.uint8)
    states = np.zeros((8, 5, 4), dtype=np.float32)
    decisions = [{"step": 5 * (k + 1), "observation_proprio_before": [0.] * 4,
                  "chosen": 0, "rounds": [{"costs": [float(i) for i in range(16)]} for _ in range(3)]}
                 for k in range(4)]
    rows = [{"root": 1884 + i, "decisions": copy.deepcopy(decisions), "final_proprio": [0.] * 4,
             "executed_actions": [[0., 0.]] * 25, "feedback_final_distance": .1} for i in range(8)]
    return {"rows": rows}, frames, states


def test_capture_comparison_and_hash_binding():
    old, frames, states = fixture()
    new = copy.deepcopy(old)
    new["rows"][0]["decisions"][0]["rounds"][0]["costs"][0] = .25
    out = compare(old, new, frames, states)
    assert out["summary"]["max_score_abs_difference"] == .25
    assert out["summary"]["chosen_matches_history"] == 32
    assert array_hash(np.zeros(4, dtype=np.float32)) != array_hash(np.zeros(4, dtype=np.int32))


def test_capture_rejects_misaligned_state():
    data, frames, states = fixture()
    states[0, 0, 0] = 1
    with pytest.raises(AssertionError):
        compare(data, data, frames, states)
