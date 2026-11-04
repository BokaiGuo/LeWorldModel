import json
from pathlib import Path

from experiments import a0_pointmaze_legacy_render_probe_20261102 as probe


def test_legacy_result_fixed_rows_and_source_binding():
    report = json.loads(probe.OUTPUT.read_text())
    assert report["runner_sha256"] == probe.sha(Path(probe.__file__))
    assert {(r["root"], r["frame"]) for r in report["rows"]} == {
        (root, frame) for root in probe.ROOTS for frame in probe.FRAMES
    }
    assert report["versions"]["mujoco_py"] == "2.1.2.14"
    for row in report["rows"]:
        for name in ("raw", "vertical_flip"):
            assert 0 <= row[name]["mae"] <= 255
            assert 0 <= row[name]["fraction_nonidentical_pixels"] <= 1
