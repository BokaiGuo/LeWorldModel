import json
from pathlib import Path

from experiments import a0_pointmaze_camera_offset_probe_20261102 as grid
from experiments import a0_pointmaze_camera_validation_20261102 as validation


def test_development_camera_grid_is_fixed_and_source_bound():
    result = json.loads(grid.OUTPUT.read_text())
    assert result["code_sha256"] == validation.sha(Path(grid.__file__))
    assert [row["lookat_xy"] for row in result["rows"]] == [list(c) for c in grid.CAMERAS]
    assert next(row for row in result["rows"] if row["lookat_xy"] == [3.0, 3.0])["mae"] == 0


def test_development_camera_validation_keeps_all_fixed_frames():
    result = json.loads(validation.OUTPUT.read_text())
    assert result["code_sha256"] == validation.sha(Path(validation.__file__))
    assert {(row["root"], row["frame"]) for row in result["rows"]} == {
        (root, frame) for root in validation.ROOTS for frame in validation.FRAMES
    }
    assert result["calibration_sha256"] == validation.sha(grid.OUTPUT)
    assert all(0 <= row["different_pixels"] <= 224 * 224 for row in result["rows"])
