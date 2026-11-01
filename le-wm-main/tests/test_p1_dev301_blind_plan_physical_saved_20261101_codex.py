from experiments.spectral_reachability import p1_dev301_blind_plan_physical_saved_20261101_codex as subject


def test_saved_main01_reconstructs_complete_physical_result():
    result = subject.verify(
        subject.ROOT / "results/p1_dev301_blind_plan_physical_20261101_codex_PREP_MAIN01.json",
        "46c5ada195eb75a3ee39fe237762fa3711566fd45baf9bf04b4ac0dd7bcedf01",
        subject.ROOT / "results/p1_dev301_blind_plan_physical_20261101_codex_RUN_MAIN01",
        "cf129af89cad37b845d3373dff6f145a8e65b3343c9a61e81547dd67c29db508",
    )
    assert result["status"] == "saved_dev301_physical_verified_development_only"
    assert result["interpretation"]["model_cost_prefers_gn"] is True
    assert result["interpretation"]["d2p_native_distance_better_than_gn"] is True
    assert result["interpretation"]["d2p_object_position_better_than_gn"] is True
    assert result["interpretation"]["d2p_prefix_has_immediate_object_progress"] is True
    assert result["interpretation"]["non_greedy_reconfiguration_mechanism_established"] is False
    assert result["interpretation"]["P1_admission"] is False
