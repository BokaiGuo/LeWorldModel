from experiments.spectral_reachability import p1_queue_goal_blind_adapter_dev02_20261101_codex as subject


def test_source_is_second_development_root_not_confirmation():
    packet, _, audit = subject.source()
    assert audit["seed"] == 301000002
    assert audit["independent_confirmation"] is False
    assert packet["rgb_history"].shape == (3, 224, 224, 3)
    assert packet["goal_rgb"].shape == (224, 224, 3)
    assert packet["raw_action_history"].shape == (10, 2)
    assert not packet["raw_action_history"].any()
