import numpy as np
import pytest

from experiments.spectral_reachability import p1_exploratory_rgb_history_v4_20261101_hypatia as mod


def test_fixed_target_off_and_scope():
    p = mod.prepare()
    assert p["slots"] == [0, 1] and p["with_target"] is False
    assert p["capture_after_steps"] == [0, 5, 10]
    assert p["freshness"] == "unknown" and not p["selection_by_outcome"]


def test_same_state_goal_does_not_require_old_pixels():
    s = np.arange(7)
    packet = {"state": s, "goal": s, "goal_rgb": np.zeros((1,)),
              "rgb_history": np.zeros((1,))}
    assert mod.equal_observation({"state": s}, {"goal_state": s}, np.ones((1,)), packet)
    assert not mod.equal_observation({"state": s + 1}, {"goal_state": s}, np.ones((1,)), packet)


def test_goal_exclusive(tmp_path):
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    result = mod.save_goal(tmp_path, 0, frame)
    assert result["file"] == "slot0_goal_rgb.npy"
    with pytest.raises(FileExistsError):
        mod.save_goal(tmp_path, 0, frame)


def test_native_and_wrapped_are_separate():
    x = mod.angles([0, 0, 0, 0, -1.5], [0, 0, 0, 0, 5.5], (True, 7))
    assert x["native_eval_success"] and not x["wrapped_diagnostic_success"]
