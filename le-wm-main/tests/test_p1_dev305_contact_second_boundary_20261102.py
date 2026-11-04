from experiments.spectral_reachability import p1_dev305_contact_second_boundary_saved_20261102 as saved


def test_second_boundary_sealed_replay_and_guard():
    result = saved.analyze()
    rows = result["rows"]
    assert len(rows) == 12
    selected = [row for row in rows if row["selected"]]
    assert len(selected) == 3
    assert all(row["status"] == "visibility_guard_stop" for row in selected)
    assert [row["steps"] for row in selected] == [2, 4, 4]
    assert not any(row["success"] for row in rows)
    assert sum(row["status"] == "fixed_horizon_completed" and not row["final_visible"]
               for row in rows) == 1
    assert not result["p1_admission"]
    assert not result["independent_confirmation"]
