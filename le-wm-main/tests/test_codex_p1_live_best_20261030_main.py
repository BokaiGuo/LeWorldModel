import hashlib
import json

import numpy as np
import pytest

from experiments.spectral_reachability import codex_p1_live_best_pair_20261030_main as paired
from experiments.spectral_reachability import codex_p1_live_best_saved_20261030_main as saved


def test_full_saved_pair_reconstructs():
    result = paired.build()
    assert result["summary"] == {"rows": 24, "same_midpoint": 24, "same_second_packet": 24,
                                 "same_all_transitions": 24, "live_success_records": 2,
                                 "offline_success_records": 2}
    assert all(row["live_endpoint"] is None for row in result["rows"]
               if row["live_status"] == "native_stopped")


def test_stage_refuses_mismatched_history_and_selected_step(tmp_path):
    stage = tmp_path / "stage_2"
    stage.mkdir()
    raw = np.zeros((10, 2), dtype="<f4").tobytes()
    scores = [1.0] * 21
    report = {"index": 0, "stage": 2, "status": "complete", "png_sha256": "png",
              "source_hashes": {}, "executed_history_sha256": "history", "selected_step": 0,
              "scores": scores, "result": {"execute_packet": 0, "raw_horizon_f32_hex": raw.hex()},
              "explicit_path_counts": {"encoder": 1, "predict": 22, "backward": 20}}
    path = stage / "report.json"
    path.write_text(json.dumps(report))
    event = {"stage_report_sha256": saved.sha(path), "plan_sha256": hashlib.sha256(raw).hexdigest(),
             "packet_float32_hex": raw[:40].hex()}
    saved.check_stage(tmp_path, {"index": 0}, event, 2, {}, "png", "unused", "history")
    with pytest.raises(ValueError):
        saved.check_stage(tmp_path, {"index": 0}, event, 2, {}, "png", "unused", "wrong")
    report["selected_step"] = 1
    path.write_text(json.dumps(report))
    event["stage_report_sha256"] = saved.sha(path)
    with pytest.raises(ValueError):
        saved.check_stage(tmp_path, {"index": 0}, event, 2, {}, "png", "unused", "history")
