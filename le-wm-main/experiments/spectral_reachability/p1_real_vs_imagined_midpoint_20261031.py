"""Pair physical single-start midpoint controls after validating executed packets."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import p1_uphill_feedback_compare_20261031_main as prior
from . import p1_uphill_packet_audit_20261031_main as audit

ROOT = prior.ROOT
NAME = "p1_real_vs_imagined_midpoint_20261031"
IM_DIR = ROOT / "results/p1_uphill_imagined_midpoint_physical_20261031_main_run_RUN01"
IM_REPORT = "3b4a12aa2828c6210753791e5853b41ebc4ee73733e168e0ce90405cbd308432"
IM_SAVED = ROOT / "results/p1_uphill_imagined_midpoint_physical_20261031_main_consume_RUN01.json"
IM_SHA = "dc505c53613a4ec53b59f90ae97fc7c1f2994202642dcfa72d6ceec49155351e"
IM_PLAN = ROOT / "results/p1_uphill_imagined_midpoint_plan_20261031_main_RUN01.json"
IM_PLAN_SHA = "02e6f2fdf66df0d1f2f1bd6743f592dea820edc1d941d343ffecf7b853d6d300"
REAL_PLAN = ROOT / "results/p1_uphill_midpoint_feedback_plan_20261031_main_RUN01.json"
REAL_PLAN_SHA = "8ac6af3059fe7ff70158fdc861f726ec9927f0c0dbc56bdc07c312b8b1ac066a"
need = prior.need


def validate_packets(i, row, imagined_plan, real_plan, imagined_events, real_events):
    need(imagined_plan["index"] == real_plan["index"] == row["index"] == i
         and imagined_plan["role"] == real_plan["role"] == row["role"]
         and imagined_plan["logical_pair"] == real_plan["logical_pair"] == row["logical_pair"]
         and imagined_plan["first_packet_hex"] == real_plan["first_packet_hex"]
         and imagined_plan["midpoint_pixel_sha256"] == real_plan["midpoint_pixel_sha256"], "plan identity")
    for plan, events, count in (
        (imagined_plan, imagined_events, row["open_returned_suffix"]),
        (real_plan, real_events, row["feedback_returned_suffix"]),
    ):
        raw = bytes.fromhex(plan["first_packet_hex"] + plan["second_packet_hex"])
        need(len(raw) == 80, "packet length")
        audit.verify_actions(np.frombuffer(raw, dtype="<f4").reshape(10, 2), events, count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    out = Path(parser.parse_args().output)
    need(out.is_absolute() and out.parent == ROOT / "results" and out.name.startswith(NAME + "_")
         and not out.exists() and not out.is_symlink(), "exclusive output")
    specs = [(prior.open_run.paired.ANCHORS, prior.open_run.paired.ANCHORS_SHA),
             (IM_SAVED, IM_SHA), (prior.FB_SAVED, prior.FB_SAVED_SHA),
             (IM_DIR / "run_report.json", IM_REPORT), (prior.FB_DIR / "run_report.json", prior.FB_REPORT),
             (IM_PLAN, IM_PLAN_SHA), (REAL_PLAN, REAL_PLAN_SHA)]
    anchors, im, real, im_report, real_report, im_plans, real_plans = [
        prior.open_run.paired.read(*x) for x in specs]
    need(len(im_plans["rows"]) == len(real_plans["rows"]) == 24, "plans")
    need(im_plans["model_before"] == im_plans["model_after"]
         == real_plans["model_before"] == real_plans["model_after"], "model identity")
    def transitions(i, a, b):
        return (prior.traces.transitions(IM_DIR / f"row_{i:02d}", a["report_sha256"]),
                prior.traces.transitions(prior.FB_DIR / f"row_{i:02d}", b["report_sha256"]))
    result = prior.compare(anchors, im, real, im_report, real_report, transitions)
    need(len(result["rows"]) == 24, "rows")
    for i, row in enumerate(result["rows"]):
        events = transitions(i, im_report["shards"][i], real_report["shards"][i])
        validate_packets(i, row, im_plans["rows"][i], real_plans["rows"][i], *events)
    fields = ("object_position_error", "object_angle_error_wrapped",
              "pusher_position_error", "native_joint_xy_error")
    complete = [r for r in result["rows"] if r["feedback_minus_open"] is not None]
    result.update({"schema": NAME, "reference_label": "imagined_midpoint",
                   "feedback_label": "real_midpoint",
                   "summary": {"rows": 24, "complete_both": len(complete),
                               "imagined_success": sum(r["open_last_success"] is True for r in result["rows"]),
                               "real_success": sum(r["feedback_last_success"] is True for r in result["rows"]),
                               "real_lower": {k: sum(r["feedback_minus_open"][k] < 0 for r in complete)
                                              for k in fields},
                               "by_role": {role: {
                                   "imagined_success": sum(r["open_last_success"] is True for r in result["rows"] if r["role"] == role),
                                   "real_success": sum(r["feedback_last_success"] is True for r in result["rows"] if r["role"] == role),
                               } for role in ("selected", "zero", "greedy")}},
                   "limitations": "Known roots and repeated arms, post-outcome exploratory control; "
                                  "different midpoint computation cost, not full equal-budget or P1.",
                   "sources": {str(p): sha for p, sha in specs},
                   "analysis_sha256": prior.feedback.sha(Path(__file__).resolve())})
    need(all(prior.feedback.sha(p) == sha for p, sha in specs), "source drift")
    digest = prior.feedback.original.physical.design.run._write_new(out, result)
    print(json.dumps({"summary": result["summary"], "sha256": digest}), flush=True)


if __name__ == "__main__":
    main()
