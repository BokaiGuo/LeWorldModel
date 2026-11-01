from experiments.spectral_reachability import p1_dev302_first_action_common_tail_saved_20261101_codex as subject


def test_saved_holdout_localizes_contact_coverage_failure():
    result = subject.verify(
        subject.ROOT / "results/p1_dev302_first_action_common_tail_20261101_codex_PREP_MAIN01.json",
        "141d147a9e7ee8c6f6fd37a20824e64eca1a95b413e2dc07d5f9559acf540aa9",
        subject.ROOT / "results/p1_dev302_first_action_common_tail_20261101_codex_RUN_MAIN01",
        "3eefbf35033c1069a52eec8f7985b001df94141f9a32f5c6566e12eb400db466",
    )
    view = result["interpretation"]
    assert view["rule_frozen_before_dev302_physics"] is True
    assert view["all_common_tails_have_zero_contact"] is True
    assert view["all_final_object_positions_unchanged"] is True
    assert view["candidate_or_tail_contact_coverage_failure"] is True
    assert view["mechanism_replication"] is False
    assert view["P1_admission"] is False
