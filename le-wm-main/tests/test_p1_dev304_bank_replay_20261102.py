from experiments.spectral_reachability import p1_dev304_bank_replay_20261102 as subject


def test_saved_scores_reconstruct_selected_candidate():
    result = subject.analyze()
    assert len(result["first_packet"]["comparisons"]) == 5
    assert len(result["terminal_packet"]["comparisons"]) == 5
    assert not result["first_packet"]["model_or_physical_rerun"]
