from experiments.spectral_reachability import p1_dev301_first_action_common_tail_saved_20261101_codex as subject


def test_saved_main01_establishes_only_development_mechanism_evidence():
    result = subject.verify(
        subject.ROOT / "results/p1_dev301_first_action_common_tail_20261101_codex_PREP_MAIN01.json",
        "82e97199a17624a713905ef7fc94d2f026c2f62eaf20f63e9e70929ce04f25d5",
        subject.ROOT / "results/p1_dev301_first_action_common_tail_20261101_codex_RUN_MAIN01",
        "2bfdac28e8ffdb7e23b4413fa4e6edb3627756002bf0df3b82ad2fcb3fcf7a8f",
    )
    view = result["interpretation"]
    assert view["all_prefixes_leave_object_position_unchanged"] is True
    assert view["d2p_prefix_native_distance_worsens"] is True
    assert view["d2p_prefix_improves_common_tail_object_position_vs_zero"] is True
    assert view["d2p_prefix_improves_common_tail_object_position_vs_gn"] is True
    assert view["d2p_prefix_advances_first_contact"] is True
    assert view["any_native_success"] is False
    assert view["P1_admission"] is False
