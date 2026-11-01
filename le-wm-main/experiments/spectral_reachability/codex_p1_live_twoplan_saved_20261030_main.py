"""Saved lifecycle verification for the two-stage live P1 development run."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import codex_p1_two_stage_saved_20261030_v1 as saved
from . import codex_p1_live_best_saved_20261030_main as previous
from . import codex_p1_live_twoplan_20261030_main as live

NAME = "codex_p1_live_twoplan_saved_20261030_main"
need, sha = saved.need, saved.sha


def check_stage(folder, row, event, stage_number, source_hashes, png_sha, previous_sha, packet_sha):
    if stage_number == 2:
        return previous.check_stage(folder, row, event, stage_number, source_hashes, png_sha, previous_sha, packet_sha)
    path = folder / "stage_1/report.json"
    need(sha(path) == event["stage_report_sha256"], "stage1 report hash")
    report = json.loads(path.read_bytes())
    need(report["index"] == row["index"] and report["stage"] == 1 and report["status"] == "complete"
         and report["png_sha256"] == png_sha and report["pixel_sha256"] == row["cp5_pixel_sha256"]
         and report["source_hashes"] == source_hashes, "stage1 input")
    initial = live.adam.prior.initial_actions(row["logical_pair"])
    need(report["initial_sha256"] == hashlib.sha256(initial.tobytes()).hexdigest()
         and len(report["candidate_scores"]) == 4 and np.isfinite(report["candidate_scores"]).all()
         and report["selected_index"] == live.adam.prior.pick(report["candidate_scores"]), "candidate selection")
    raw = bytes.fromhex(report["result"]["raw_horizon_f32_hex"])
    need(len(raw) == 80 and hashlib.sha256(raw).hexdigest() == event["plan_sha256"]
         and raw[:40].hex() == event["packet_float32_hex"] and report["result"]["execute_packet"] == 0,
         "stage1 packet")
    packet = np.frombuffer(raw[:40], dtype="<f4").copy().reshape(5, 2)
    live.live.source.feedback.packet_bytes(packet)
    expected = {"encoder": 1, "predict": 168, "backward": 80}
    need(report["explicit_path_counts"] == expected, "declared counts")
    return report["result"], packet, expected


def consume(directory, report_sha, prepared, prepare_sha):
    live.launcher.NAME, live.launcher.build = live.NAME, live.build
    saved.check_stage = check_stage
    result = saved.consume(directory, report_sha, prepared, prepare_sha)
    for row in result["rows"]:
        if row["status"] == "native_stopped":
            row["endpoint_metrics"] = None
    result.update(schema=NAME, code_sha256=sha(Path(__file__).resolve()),
                  limitations="shared lifecycle consumer; stage1 candidates and previous latent not independently "
                              "saved; declared model counts not hook verified; known roots")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for field in ("directory", "report-sha", "prepared", "prepare-sha", "output"):
        ap.add_argument("--"+field, required=True)
    args = ap.parse_args()
    out = Path(args.output)
    need(out.is_absolute() and out.parent == live.ROOT/"results" and out.name.startswith(NAME+"_")
         and not out.exists() and not out.is_symlink(), "exclusive output")
    result = consume(args.directory, args.report_sha, args.prepared, args.prepare_sha)
    print(json.dumps({"rows": len(result["rows"]), "sha256": live.launcher.h10._write_new(out, result)}))
