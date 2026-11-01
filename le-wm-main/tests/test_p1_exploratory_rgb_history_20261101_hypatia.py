import numpy as np

from experiments.spectral_reachability import p1_exploratory_rgb_history_20261101_hypatia as mod


def test_prepare_fixed_budget_and_unknown():
    p = mod.prepare()
    assert p["slots"] == [0, 1] and p["capture_after_steps"] == [0, 5, 10]
    assert p["max_resets"] == 4 and p["max_public_steps"] == 20
    assert p["freshness"] == "unknown" and not p["selection_by_outcome"]


def test_observation_must_match_state_goal_and_pixels():
    state = np.arange(7)
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    packet = {"state": state, "goal": state, "goal_rgb": image, "rgb_history": np.stack([image] * 3)}
    assert mod.equal_observation({"state": state}, {"goal_state": state, "goal": image}, image, packet)
    image2 = image.copy()
    image2[0, 0, 0] = 1
    assert not mod.equal_observation({"state": state}, {"goal_state": state, "goal": image}, image2, packet)


def test_exclusive_write(tmp_path):
    path = tmp_path / "record.json"
    mod.write(path, {"value": 1})
    try:
        mod.write(path, {"value": 2})
    except FileExistsError:
        pass
    else:
        raise AssertionError("overwriting record allowed")
