from experiments.spectral_reachability.p1_dev305_contact_packet_saved_20261102 import analyze


def test_saved_physical_panel_has_all_arms():
    result = analyze()
    assert len(result["rows"]) == 9
    assert not result["independent_confirmation"]
    assert not any(row["native_success"] for row in result["rows"])
