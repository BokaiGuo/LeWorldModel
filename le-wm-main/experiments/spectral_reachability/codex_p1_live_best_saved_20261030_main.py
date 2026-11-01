"""Saved lifecycle consumer for fixed-first-packet live best-iterate P1."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import codex_p1_two_stage_saved_20261030_v1 as saved
from . import codex_p1_live_best_20261030_main as live
from . import codex_p1_adam_best_iterate_20261030_main as best

NAME = "codex_p1_live_best_saved_20261030_main"
need, sha = saved.need, saved.sha
ENTRIES = None


def check_stage(folder, row, event, stage_number, source_hashes, png_sha, previous_sha, packet_sha):
    path = folder / f"stage_{stage_number}/report.json"
    need(sha(path) == event["stage_report_sha256"], "stage report hash")
    report = json.loads(path.read_bytes())
    need(report["status"] == "complete" and report["index"] == row["index"] and report["stage"] == stage_number
         and report["png_sha256"] == png_sha and report["source_hashes"] == source_hashes, "stage input")
    raw = bytes.fromhex(report["result"]["raw_horizon_f32_hex"])
    need(len(raw) == 80 and hashlib.sha256(raw).hexdigest() == event["plan_sha256"]
         and raw[:40].hex() == event["packet_float32_hex"]
         and report["result"]["execute_packet"] == 0 and raw[40:] == bytes(40), "packet")
    packet = np.frombuffer(raw[:40], dtype="<f4").copy().reshape(5, 2)
    live.source.feedback.packet_bytes(packet)
    if stage_number == 1:
        need(report["pixel_sha256"] == row["cp5_pixel_sha256"] and report["executed_history_sha256"] is None
             and packet.tobytes() == ENTRIES[row["index"]][2][0].tobytes()
             and report["selected_step"] is None and report["scores"] == [], "first stage")
    else:
        need(report["executed_history_sha256"] == packet_sha and
             report["selected_step"] == best.select_best(report["scores"]), "feedback binding")
    expected = {"encoder": 1, "predict": 22 if stage_number == 2 else 0,
                "backward": 20 if stage_number == 2 else 0}
    need(report["explicit_path_counts"] == expected, "declared counts")
    return report["result"], packet, expected


def consume(directory, report_sha, prepared, prepare_sha):
    global ENTRIES
    ENTRIES = live.source.inputs()
    live.launcher.NAME = live.NAME
    live.launcher.build = live.build
    saved.check_stage = check_stage
    result = saved.consume(directory, report_sha, prepared, prepare_sha)
    for row in result["rows"]:
        if row["status"] == "native_stopped":
            row["endpoint_metrics"] = None
    result.update(schema=NAME, code_sha256=sha(Path(__file__).resolve()),
                  limitations="shared lifecycle consumer; declared model path counts not hook-verified; "
                              "previous latent not independently saved; known roots; no online-vs-offline replay yet")
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
