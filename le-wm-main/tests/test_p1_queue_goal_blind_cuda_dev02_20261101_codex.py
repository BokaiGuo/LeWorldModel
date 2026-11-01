from experiments.spectral_reachability import p1_queue_goal_blind_cuda_dev02_20261101_codex as subject


def test_prepare_binds_second_development_root():
    value = subject.build()
    assert value["seed"] == 301000002
    assert value["classification"] == "development_holdout02_only"
    assert value["independent_confirmation"] is False
    assert value["P1_admission"] is False
    assert len(value["slots"]) == 1
