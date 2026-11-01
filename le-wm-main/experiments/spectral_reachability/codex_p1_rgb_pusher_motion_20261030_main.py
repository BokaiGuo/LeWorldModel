"""Post-outcome physical interpretation of the fixed RGB pusher correction."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import codex_p1_rgb_pusher_ablation_compare_20261030_main as comparison

NAME = "codex_p1_rgb_pusher_motion_20261030_main"
ROOT = comparison.ROOT
DIR = ROOT / "results/codex_p1_rgb_pusher_ablation_20261030_main_run_RUN01"
REPORT_SHA = "9a3afed2ba7abc964dc69c4f8498a8e6cd8256b0ec7439661a4c56b0bd4017e7"
SAVED = ROOT / "results/codex_p1_rgb_pusher_ablation_20261030_main_saved_RUN01.json"
SAVED_SHA = "7d3a232b63e25faa00e59735e8049737075e343f80e5671068214b019e43c40a"
PAIRED = ROOT / "results/codex_p1_rgb_pusher_ablation_compare_20261030_main_RUN01.json"
PAIRED_SHA = "c2eae53b91b1b3bf527dba972d77e116816ee3e3081fcc5829605c8434c3d5c4"
need = comparison.need


def run(output):
    report = comparison.pair.read(DIR / "run_report.json", REPORT_SHA)
    saved = comparison.pair.read(SAVED, SAVED_SHA)
    paired = comparison.pair.read(PAIRED, PAIRED_SHA)
    h10 = comparison.ablation.launch.h10
    design = h10.design.verify(h10.INPUT, h10.INPUT_SHA)
    anchors = design["anchors"]
    need(all(len(x) == 24 for x in (anchors, saved["rows"], paired["rows"], report["shards"])), "roster")
    rows = []
    for i, (anchor, row, pair, shard) in enumerate(zip(
            anchors, saved["rows"], paired["rows"], report["shards"], strict=True)):
        need(anchor["index"] == row["index"] == pair["identity"]["index"] == shard["index"] == i,
             "identity")
        if row["rgb_decision"] != "rgb_pusher_correction":
            rows.append({"index": i, "status": row["status"], "decision": row["rgb_decision"],
                         "measurements": None})
            continue
        folder = DIR / f"row_{i:02d}" / "physical"
        traces = comparison.pair.transitions(folder, shard["report_sha256"])
        need(len(traces) >= 6 and traces[4]["step"] == 5 and traces[5]["step"] == 6,
             "correction transition")
        command = np.frombuffer(bytes.fromhex(traces[5]["actual_bits"]), dtype="<f4").astype(np.float64)
        need(command.shape == (2,) and np.isclose(np.linalg.norm(command), 0.2, atol=1e-6), "command ABI")
        before = np.asarray(traces[4]["state"], np.float64)
        after = np.asarray(traces[5]["state"], np.float64)
        goal = np.asarray(anchor["goal"], np.float64)
        need(before.shape == after.shape == goal.shape == (7,) and np.isfinite(goal).all(), "state ABI")
        movement = after[:2] - before[:2]
        start_error = float(np.linalg.norm(before[:2] - goal[:2]))
        first_error = float(np.linalg.norm(after[:2] - goal[:2]))
        last_error = float(np.linalg.norm(np.asarray(traces[-1]["state"])[:2] - goal[:2]))
        rows.append({"index": i, "identity": pair["identity"], "status": row["status"],
                     "decision": row["rgb_decision"], "steps": len(traces),
                     "measurements": {"command": command.tolist(), "pusher_displacement": movement.tolist(),
                                      "command_alignment": float(np.dot(command, movement)),
                                      "pusher_distance_before": start_error,
                                      "pusher_distance_after_first": first_error,
                                      "pusher_distance_last": last_error,
                                      "delta_first": first_error - start_error,
                                      "delta_last": last_error - start_error,
                                      "last_vs_old_endpoint": pair["new_minus_old"]}})
    measured = [r["measurements"] for r in rows if r["measurements"] is not None]
    result = {"schema": NAME, "status": "complete", "rows": rows,
              "summary": {"count": len(measured), "command_aligned": sum(m["command_alignment"] > 0 for m in measured),
                          "first_reduces_pusher_error": sum(m["delta_first"] < 0 for m in measured),
                          "last_reduces_pusher_error": sum(m["delta_last"] < 0 for m in measured)},
              "sources": {str(DIR / "run_report.json"): REPORT_SHA, str(SAVED): SAVED_SHA,
                          str(PAIRED): PAIRED_SHA, str(h10.INPUT): h10.INPUT_SHA},
              "limitations": "post-outcome physical metrics, two known roots, repeated trajectories; "
                             "command alignment does not prove absence of contact or actuation bugs",
              "P1_admission": False}
    result["code_sha256"] = comparison.ablation.launch.sha(Path(__file__).resolve())
    print(json.dumps({"sha256": h10._write_new(output, result), "summary": result["summary"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    out = Path(parser.parse_args().output)
    need(out.is_absolute() and out.parent == ROOT / "results" and out.name.startswith(NAME + "_")
         and not out.exists() and not out.is_symlink(), "exclusive output")
    run(out)
