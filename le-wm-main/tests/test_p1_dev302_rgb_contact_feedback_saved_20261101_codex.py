from experiments.spectral_reachability import (
    p1_dev302_rgb_contact_feedback_saved_20261101_codex as subject,
)


def test_saved_dev302_rgb_contact_diagnosis():
    result = subject.verify(
        subject.ROOT
        / "results/p1_dev302_rgb_contact_feedback_20261101_codex_PREP_MAIN02.json",
        "615e45c2b0a7e9409ca22f50f2729351564e1f401ea6c81ea6ae542af71d3dc1",
        subject.ROOT
        / "results/p1_dev302_rgb_contact_feedback_20261101_codex_RUN_MAIN02",
        "aacb4d296e0a9aacba9bff9e1803bc82d48345beaddc2035b1e72128fd74e0a0",
    )
    assert result["status"] == "saved_development_diagnosis_verified"
    view = result["interpretation"]
    assert view["compact_moving_sprite_fragmentation_was_code_protocol_issue"] is True
    assert view["rgb_prefix_moved_pusher_toward_object"] is True
    assert view["rgb_prefix_reached_native_contact"] is False
    assert view["same_sealed_tail_produced_object_response"] is False
    assert view["requires_rgb_refreshed_tail_replanning"] is True
    assert view["P1_admission"] is False
