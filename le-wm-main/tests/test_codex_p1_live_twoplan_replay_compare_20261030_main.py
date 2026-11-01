import copy
import hashlib

import numpy as np
import pytest

from experiments.spectral_reachability import codex_p1_live_twoplan_replay_compare_20261030_main as mod


def records():
    action = np.arange(20, dtype="<f4").reshape(2, 5, 2)
    raw = action.tobytes()
    offline = {"index": 0, "selected_actions": action.tolist(),
               "selected_action_sha256": hashlib.sha256(raw).hexdigest(),
               "initial_f32_sha256": "init", "candidate_scores": [1.0, 2.0], "selected_index": 0}
    first = {"index": 0, "initial_sha256": "init", "candidate_scores": [1.0, 2.0],
             "selected_index": 0, "result": {"raw_horizon_f32_hex": raw.hex()}}
    second = {"index": 0, "pixel_sha256": "pixel", "executed_history_sha256": "history",
              "scores": [2.0, 1.0], "selected_step": 1,
              "result": {"raw_horizon_f32_hex": (raw[40:] + bytes(40)).hex()}}
    old_first = {"index": 0, "result": {"raw_horizon_f32_hex": (raw[:40] + bytes(40)).hex()}}
    return offline, first, second, old_first, copy.deepcopy(second)


def test_equal_records():
    row = mod.compare_row(*records())
    assert all(value for key, value in row.items() if key.startswith("same_"))


def test_detect_different_second_packet():
    args = list(records())
    args[4]["result"]["raw_horizon_f32_hex"] = bytes(80).hex()
    assert not mod.compare_row(*args)["same_stage2_packet"]


def test_reject_offline_action_tamper():
    args = list(records())
    args[0]["selected_actions"][0][0][0] = 99
    with pytest.raises(ValueError, match="offline action hash"):
        mod.compare_row(*args)


def test_reject_mismatched_index():
    args = list(records())
    args[3]["index"] = 1
    with pytest.raises(ValueError, match="index"):
        mod.compare_row(*args)
