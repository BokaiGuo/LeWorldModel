from experiments.spectral_reachability import p1_dev305_third_boundary_saved_20261102 as saved


def test_real_risk_selected_packet_and_zero_controls():
    result = saved.analyze()
    rows = result["rows"]
    selected = [row for row in rows if row["selected"]]
    assert len(rows) == 6
    assert [row["candidate"] for row in selected] == ["zero", "zero", "toward_02"]
    assert [row["final_visible"] for row in selected] == [True, True, False]
    assert not any(row["success"] for row in rows)
    assert not result["p1_admission"]
    assert not result["independent_confirmation"]
