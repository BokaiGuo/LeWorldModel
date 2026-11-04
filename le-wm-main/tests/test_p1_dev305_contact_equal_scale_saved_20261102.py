from experiments.spectral_reachability.p1_dev305_contact_equal_scale_saved_20261102 import analyze


def test_receipts_and_conservative_zero_steps():
    report = analyze()
    assert len(report["rows"]) == 12
    assert not report["independent_confirmation"]
    conservative = [r for r in report["rows"] if r["arm"].endswith("__guard16")]
    assert len(conservative) == 3
    assert all(row["steps"] == 0 and row["reward"] is None for row in conservative)
