from experiments.spectral_reachability import p1_dev305_two_packet_model_bank_saved_20261102 as saved


def test_model_and_physical_bank_join_is_complete():
    result = saved.analyze()
    assert [row["arm"] for row in result["rows"]] == ["candidate_02", "candidate_00", "zero"]
    assert all(row["model_regret_reward_units"] >= 0 for row in result["rows"])
    assert result["p1_admission"] is False
