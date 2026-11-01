import json
import math

from experiments.spectral_reachability import codex_p1_adam_stop_pair_20261030_main as audit


def test_actual_stop_time_and_native_success():
    result = audit.build()
    saved = json.loads((audit.base.ROOT / f"results/{audit.NAME}_RUN01.json").read_text())
    assert result["rows"] == saved["rows"]
    assert len(result["rows"]) == 5
    for row in result["rows"]:
        terminal = row["steps"][-1]
        assert terminal["step"] == row["original_stop_step"]
        assert terminal["open_terminated"] and terminal["open_metrics"]["native_success"]
        for branch in ("open_metrics", "feedback_metrics"):
            m = terminal[branch]
            assert m["native_success"] == (m["native_joint_xy_error"] < 20 and m["native_angle_term"] < math.pi / 9)
    lost = [r for r in result["rows"] if not r["steps"][-1]["feedback_metrics"]["native_success"]]
    assert len(lost) == 4
    assert all(r["steps"][-1]["feedback_metrics"]["native_joint_xy_error"] >= 20 for r in lost)
