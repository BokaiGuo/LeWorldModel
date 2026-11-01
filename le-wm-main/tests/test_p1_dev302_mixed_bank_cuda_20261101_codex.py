from experiments.spectral_reachability import p1_dev302_mixed_bank_cuda_20261101_codex as subject


def test_protocol_has_equal_tail_budget_and_required_selection_rules():
    value = subject.protocol()
    assert value["algorithm"]["prefix_candidates"] == 8
    assert value["budget"]["action_encoder_calls"] == 88
    assert value["budget"]["predict_calls"] == 88
    assert value["algorithm"]["same_bank_greedy"].startswith("minimum predicted cp5")
    assert value["algorithm"]["selected"].startswith("minimum robust nonlinear")
    assert value["physical_outcomes_used"] is False
    assert value["fair_total_cost"] is False
    assert value["P1_admission"] is False
