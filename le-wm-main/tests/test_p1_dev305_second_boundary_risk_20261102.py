from experiments.spectral_reachability import p1_dev305_second_boundary_risk_saved_20261102 as saved


def test_projection_and_physical_binding_on_sealed_development_panel():
    result = saved.analyze()
    rows = result["rows"]
    assert len(rows) == 12
    selected = [row for row in rows if row["selected"]]
    assert [row["candidate"] for row in selected] == ["zero", "toward_02", "toward_02"]
    assert all(row["actual_steps"] == 5 and row["actual_final_visible"] for row in selected)
    assert not any(row["predicted_visible"] and not row["actual_final_visible"] for row in rows)
    assert not any(row["actual_success"] for row in rows)
    assert not result["p1_admission"]
    assert not result["independent_confirmation"]
