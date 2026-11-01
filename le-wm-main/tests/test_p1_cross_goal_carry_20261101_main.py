from experiments.spectral_reachability.p1_cross_goal_carry_20261101_main import cross_entry


def test_preserve_root_without_inheriting_goal_action_witness():
    root = dict(root=0, root_state=[1, 2], pixels=[3], pixel_sha256="a", goal_state=[4],
                witness10_float32_hex="not_a_cross_pair_witness", historical_first_success_step=10)
    goal = dict(root=1, root_state=[9, 9], pixels=[8], pixel_sha256="b", goal_state=[7])
    result = cross_entry(root, goal)
    assert result["root"] == 0 and result["root_state"] == [1, 2]
    assert result["goal_state"] == [7] and result["pixels"] == [8]
    assert "witness10_float32_hex" not in result
    assert "historical_first_success_step" not in result
    assert root["pixels"] == [3]
