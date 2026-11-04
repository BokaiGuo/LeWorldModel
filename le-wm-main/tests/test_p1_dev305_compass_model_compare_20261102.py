from experiments.spectral_reachability import p1_dev305_compass_model_compare_saved_20261102 as saved


def test_complete_paired_bank_and_no_admission():
    result = saved.analyze()
    assert len(result["pair_counts"]) == 6
    assert all(r["same"] + r["reversed"] + r["ties"] == 153 for r in result["pair_counts"])
    assert not result["p1_admission"]
