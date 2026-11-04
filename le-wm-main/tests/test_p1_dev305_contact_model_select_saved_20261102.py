from experiments.spectral_reachability.p1_dev305_contact_model_select_saved_20261102 import analyze


def test_fixed_bank_model_selected_physical_receipts():
    report = analyze()
    assert len(report["rows"]) == 12
    selected = [row for row in report["rows"] if row["selected"]]
    assert len(selected) == 3
    assert all(not row["native_success"] for row in selected)
    assert not report["independent_confirmation"]
