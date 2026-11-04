import json
from pathlib import Path

from experiments import a0_pointmaze_replay_render_timing_20261102 as probe


def test_fixed_replay_rows_and_source_binding():
    report = json.loads(probe.OUTPUT.read_text())
    assert report["script_sha256"] == probe.sha(Path(probe.__file__))
    assert {(r["root"], r["frame"]) for r in report["rows"]} == {
        (root, frame) for root in probe.ROOTS for frame in probe.FRAMES
    }
    for row in report["rows"]:
        assert row["position_error_l2"] >= 0
        assert row["velocity_error_l2"] >= 0
        for kind in ("raw", "synced"):
            assert 0 <= row[kind]["mae"] <= 255
            assert 0 <= row[kind]["different_pixels"] <= 224 * 224


def test_first_frame_has_no_render_timing_difference():
    report = json.loads(probe.OUTPUT.read_text())
    initial = [row for row in report["rows"] if row["frame"] == 0]
    assert all(row["raw"]["rgb_sha256"] == row["synced"]["rgb_sha256"] for row in initial)
