from experiments.spectral_reachability import p1_dev305_contact_risk_trajectory_20261102 as trajectory


def test_all_saved_frames_are_checked_and_aligned():
    result = trajectory.analyze()
    rows = result["rows"]
    assert len(rows) == 12
    assert all([frame["step"] for frame in row["timeline"]] == list(range(6))
               for row in rows)
    assert all(row["timeline"][-1]["visible"] == row["actual_terminal_visible"]
               for row in rows)
    assert all(row["first_invalid_step"] == next(
        (frame["step"] for frame in row["timeline"] if not frame["visible"]), None)
               for row in rows)
    assert sum(row["predicted_terminal_visible"] and not row["all_frames_visible"]
               for row in rows) == 0
    assert sum(row["all_frames_visible"] for row in rows) == 7
    assert not result["p1_admission"]
    assert not result["independent_confirmation"]
