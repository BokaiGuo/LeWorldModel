from experiments.spectral_reachability import p1_cp10_coherent_saved_20261102 as subject


def test_complete_paired_coherent_trace():
    result = subject.analyze()
    assert len(result["rows"]) == 8
    assert result["frames_verified"] == 112
    assert not result["P1_admission"] and not result["independent_confirmation"]
