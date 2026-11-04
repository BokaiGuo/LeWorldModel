from collections import Counter

from experiments.spectral_reachability import p1_dev305_contact_risk_projection_saved_20261102 as saved


def test_saved_projection_recomputes_and_binds_all_physical_rows():
    result = saved.analyze()
    rows = result["rows"]
    assert len(rows) == 12
    assert len({(row["arm"], row["candidate"]) for row in rows}) == 12
    assert Counter(row["arm"] for row in rows) == {
        "candidate_02": 4,
        "candidate_00": 4,
        "zero": 4,
    }
    assert all(sum(row["selected"] for row in rows if row["arm"] == arm) == 1
               for arm in ("candidate_02", "candidate_00", "zero"))
    assert sum(row["predicted_visible"] and not row["actual_visible"] for row in rows) == 0
    assert sum(not row["predicted_visible"] and row["actual_visible"] for row in rows) == 1
    assert not result["p1_admission"]
    assert not result["independent_confirmation"]
