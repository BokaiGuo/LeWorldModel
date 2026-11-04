from experiments.spectral_reachability.p1_dev305_contact_rgb_feedback_saved_20261102 import analyze


def test_sealed_rgb_decisions_and_terminal_censoring():
    result = analyze()
    assert len(result["rows"]) == 9
    assert not result["independent_confirmation"]
    assert all(not row["native_success"] for row in result["rows"])
    assert any(row["status"] == "visibility_guard_stop" for row in result["rows"])
