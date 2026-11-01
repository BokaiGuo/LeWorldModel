import copy

from experiments.spectral_reachability.audit_reacher_d2p_zero_replay_layers import compare


def test_separates_render_encoder_and_physics_changes():
    a = {"snapshot": {"qpos": [1, 2], "qvel": [0, 0]}, "rgb_sha256": "a",
         "history_rgb_sha256": ["b"], "observed": [0., 1.]}
    b = copy.deepcopy(a)
    assert all(v is True for k, v in compare(a, b).items() if k != "observed_max_error")
    b["observed"][0] = .1
    result = compare(a, b)
    assert result["full_state_exact"] and result["rgb_exact"] and not result["observed_exact"]
    b["rgb_sha256"] = "changed"
    assert not compare(a, b)["rgb_exact"]
    b["snapshot"]["qvel"][0] = .01
    assert not compare(a, b)["full_state_exact"]
    assert compare(a, b)["qpos_exact"]
