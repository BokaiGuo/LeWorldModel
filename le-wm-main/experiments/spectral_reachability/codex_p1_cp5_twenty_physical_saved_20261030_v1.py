"""Verify 20-step physical results and pair with fixed two-step results."""
import argparse
import json
from pathlib import Path

from . import codex_p1_cp5_twenty_physical_20261030_v1 as run
from . import codex_p1_cp5_physical_saved_20261030_v1 as saved
from . import codex_p1_h10_paired_table_20261030_main as metrics

ROOT, need, sha = run.ROOT, run.need, run.sha
OLD_DIR = ROOT/"results/codex_p1_cp5_physical_24_20261030_v1_run_CODEX01"
OLD_RUN_SHA = "c11f4e79f999ee9e89ea94ef05cb093a3e32ed1beb1cfe5c9bbfb33d7f29df6a"
OLD_RECEIPT = ROOT/"results/codex_p1_cp5_physical_saved_20261030_v1_CODEX01.json"
OLD_RECEIPT_SHA = "0585d0ecb6299d50491f330da46d8011e8a2669979ce6347617027b67e4984b9"
OLD_PLAN_DIR = run.old.PLAN_DIR


def endpoint(report, goal):
    row = report["row"]
    status, responses = row["status"], row["responses"]
    observed = len(responses)
    need(status in ("probe_completed", "native_stopped", "censored_out_of_domain"), "physical status")
    # A native stop at step 10 is at the requested horizon, not an early stop.
    early = status == "native_stopped" and observed < 10
    at_horizon = status == "native_stopped" and observed == 10
    if status == "native_stopped":
        need(observed > 0 and (responses[-1]["terminated"] or responses[-1]["truncated"]), "stop event")
    full = observed == 10 and status in ("probe_completed", "native_stopped")
    state = responses[-1]["state"] if full else None
    return {"status": status, "accepted_suffix_steps": observed,
            "early_native_stop": early, "native_stop_at_horizon": at_horizon,
            "censored": status == "censored_out_of_domain", "horizon_endpoint_state": state,
            "horizon_metrics": metrics.physical_metrics(state, goal),
            "last_accepted_state": responses[-1]["state"] if responses else None}


def consume(directory, report_sha, prepared, prepare_sha):
    p = run.verify(prepared, prepare_sha)
    directory = Path(directory)
    report = run.read(directory/"run_report.json", report_sha)
    old_report = run.read(OLD_DIR/"run_report.json", OLD_RUN_SHA)
    old_receipt = run.read(OLD_RECEIPT, OLD_RECEIPT_SHA)
    need(report["schema"] == run.NAME+".run" and report["status"] == "recorded_not_consumed"
         and report["prepare_sha256"] == prepare_sha and report["fixed_rows"] == 24
         and len(report["shards"]) == len(old_report["shards"]) == len(old_receipt["rows"]) == 24
         and old_report["status"] == "recorded_not_consumed" and old_receipt["status"] == "saved_checks_passed"
         and sha(directory/"binding.json") == report["binding_sha256"], "complete fixed runs")
    anchors = run.h10.native.read(run.h10.INPUT, cap=64*1024**2)["anchors"]
    old_p = run.read(run.OLD_PREP, run.OLD_SHA)
    planner_receipt = run.read(run.PLAN_RECEIPT, run.PLAN_RECEIPT_SHA)
    old_planner_receipt = run.read(run.old.RECEIPT, run.old.RECEIPT_SHA)
    rows = []
    for i, (anchor, item, old_item, new_s, old_s) in enumerate(zip(
            anchors, p["roster"], old_p["roster"], report["shards"], old_report["shards"], strict=True)):
        need(new_s["index"] == old_s["index"] == anchor["index"] == i and
             new_s["status"] == old_s["status"] == "recorded_not_consumed", "ordered shard status")
        for field in ("index", "logical_pair", "root_slot", "condition", "arm", "role", "task_id",
                      "prefix5_float32_hex", "cp5_file_sha256", "cp5_pixel_sha256"):
            need(item[field] == old_item[field], "paired identity: "+field)
        new_folder, old_folder = directory/f"row_{i:02d}", OLD_DIR/f"row_{i:02d}"
        new_check = saved.check_row(new_folder, anchor, item, new_s["report_sha256"])
        old_check = saved.check_row(old_folder, anchor, old_item, old_s["report_sha256"])
        need(old_check == old_receipt["rows"][i], "two-step saved receipt")
        new_row = run.read(new_folder/"report.json", new_s["report_sha256"])
        old_row = run.read(old_folder/"report.json", old_s["report_sha256"])
        need(new_check["counts"] == new_row["counts"] and old_check["counts"] == old_row["counts"],
             "physical resource receipt")
        new_plan = run.read(run.PLAN_DIR/f"row_{i:02d}_result.json", planner_receipt["rows"][i]["row_sha256"])
        old_plan = run.read(OLD_PLAN_DIR/f"row_{i:02d}_result.json", old_planner_receipt["rows"][i]["row_sha256"])
        rows.append({"index": i, "logical_pair": item["logical_pair"], "root_slot": item["root_slot"],
                     "physical_root_sha256": anchor["physical_root_sha256"], "condition": item["condition"],
                     "arm": item["arm"], "role": item["role"], "task_id": item["task_id"],
                     "prefix5_float32_hex": item["prefix5_float32_hex"],
                     "cp5_file_sha256": item["cp5_file_sha256"], "cp5_pixel_sha256": item["cp5_pixel_sha256"],
                     "two_step": {**endpoint(old_row, anchor["goal"]), "raw10_sha256": old_item["raw10_sha256"],
                                  "physical_resources": old_row["counts"],
                                  "planning_resources": old_plan["result"]["ledger_receipt"]},
                     "twenty_step": {**endpoint(new_row, anchor["goal"]), "raw10_sha256": item["raw10_sha256"],
                                     "physical_resources": new_row["counts"],
                                     "planning_resources": new_plan["result"]["ledger_receipt"]}})
    return {"schema": run.NAME+".saved_pair", "status": "24_pairs_saved_verified", "fixed_rows": 24,
            "prepare_sha256": prepare_sha, "twenty_run_sha256": report_sha, "two_run_sha256": OLD_RUN_SHA,
            "rows": rows, "known_roots": True, "os_isolation_certified": False,
            "compute_budgets_differ": True, "equal_budget_certified": False,
            "independent_confirmation": False, "P1_admission": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--directory", required=True)
    ap.add_argument("--report-sha", required=True)
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--prepare-sha", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    result = consume(a.directory, a.report_sha, a.prepared, a.prepare_sha)
    path = Path(a.output)
    need(path.is_absolute() and path.parent == ROOT/"results"
         and path.name.startswith("codex_p1_cp5_twenty_physical_saved_20261030_v1_"), "exclusive output")
    print(json.dumps({"sha256": run.h10._write_new(path, result), "rows": len(result["rows"])}))


if __name__ == "__main__":
    main()
