import numpy as np

from experiments.spectral_reachability import run_pusht_d2p_p1_physical_calibration as physical


def test_pose_label_and_replay_contract_are_state_and_rgb_sensitive():
    state = np.zeros(7); goal = np.zeros(7); state[2:4] = (3.0, 4.0); state[4] = np.pi / 2
    label = physical._pose_label(state, goal)
    assert label["xy_distance"] == 5.0
    first = {"warmup_rgb_sha256": "a", "goal_rgb_sha256": "b", "successor_rgb_sha256": "c", "final_rgb_sha256": "d", "successor_state": [1.0], "final_state": [2.0]}
    assert physical._replay_matches(first, dict(first))
    changed = dict(first); changed["final_rgb_sha256"] = "different"
    assert not physical._replay_matches(first, changed)
