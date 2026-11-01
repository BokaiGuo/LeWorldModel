from experiments.spectral_reachability import (
    p1_dev303_blind_adapter_20261101_codex as subject,
)


def test_dev303_source_is_complete_and_development_only():
    arrays, receipt_sha, audit = subject.source()
    assert arrays["rgb_history"].shape == (3, 224, 224, 3)
    assert arrays["raw_action_history"].shape == (10, 2)
    assert arrays["goal_rgb"].shape == (224, 224, 3)
    assert receipt_sha == "4d7c23525d745278ba29f46676c88dddd1a95aae44c4e46241fb71cdd60af78d"
    assert audit["seed"] == 301000003
    assert audit["P1_admission"] is False
