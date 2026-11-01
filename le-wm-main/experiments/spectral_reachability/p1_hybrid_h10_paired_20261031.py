"""Pair verified hybrid H10 observations with pinned historical controls."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from . import p1_hybrid_h10_saved_20261031 as saved
from .codex_p1_h10_paired_table_20261030_main import physical_metrics

ROOT = saved.hybrid.ROOT
TABLE = ROOT / "results/p1_uphill_pairing_table_20261031_RUN01.csv"
TABLE_SHA = "30c312e291b65f3219e98bf1d10ad327654923d0909139601a3afb6005056189"
CP5 = ROOT / "results/p1_hybrid_cp5_physical_saved_20261031_RUN01.json"
CP5_SHA = "70068a853c359b37d74dcac17f0121e9978b7227e4c85373582cb74a3b9764ff"
KEYS = ("object_position_error", "object_angle_error_wrapped", "pusher_position_error",
        "native_joint_xy_error", "native_angle_term")
need, sha = saved.require, saved.hybrid.sha


def difference(a, b):
    if a is None or b is None:
        return None
    return {k: a[k] - b[k] for k in KEYS}


def csv_metrics(row, prefix):
    values = [row[prefix + key] for key in KEYS]
    if all(value == "" for value in values):
        return None
    need(all(value != "" for value in values), "partial metric")
    result = {key: float(value) for key, value in zip(KEYS, values, strict=True)}
    need(all(math.isfinite(value) for value in result.values()), "finite metrics")
    return result


def build(directory, report_sha, prepared, prepare_sha):
    verified = saved.consume(directory, report_sha, prepared, prepare_sha)
    need(sha(TABLE) == TABLE_SHA, "historical table hash")
    with TABLE.open(newline="") as handle:
        old = list(csv.DictReader(handle))
    p = saved.hybrid.read(prepared)
    need(sha(CP5) == CP5_SHA, "independent suffix physical receipt")
    own = saved.hybrid.read(CP5)
    need(own["status"] == "saved_checks_passed" and own["fixed_rows"] == 2 and
         len(own["rows"]) == 2, "independent suffix roster")
    run_report = saved.hybrid.read(Path(directory) / "run_report.json")
    output = []
    for anchor, bank, shard, record, own_row in zip(
            p["anchors"], p["banks"], verified["rows"], run_report["shards"], own["rows"], strict=True):
        goal = anchor["goal"]
        need(own_row["index"] == anchor["index"] and own_row["root_slot"] == anchor["root_slot"]
             and own_row["condition"] == anchor["condition"], "independent suffix identity")
        own_endpoint = physical_metrics(own_row["completed_endpoint"], goal)
        root = physical_metrics(anchor["root_state"], goal)
        prefix = physical_metrics(anchor["checkpoint"]["state"], goal)
        endpoints = [physical_metrics(state, goal) for state in shard["endpoints"]]
        report_path = Path(directory) / f"anchor_{anchor['index']:02d}" / "report.json"
        need(sha(report_path) == record["sha256"], "shard re-read binding")
        physical = saved.hybrid.read(report_path)["rows"]
        need(bank[0]["name"] == "zero", "zero suffix index")
        for probe, status, endpoint, real in zip(bank, shard["statuses"], endpoints, physical, strict=True):
            need(real["index"] == probe["index"] and real["status"] == status, "branch re-read")
            last = physical_metrics(real["responses"][-1]["state"], goal) if real["responses"] else None
            raw = bytes.fromhex(probe["raw10_float32_hex"])
            digest = hashlib.sha256(raw).hexdigest()
            need(digest == probe["raw10_sha256"], "probe digest")
            controls = {}
            for role in ("zero", "greedy"):
                matches = [row for row in old if row["logical_pair"] == str(anchor["logical_pair"])
                           and row["role"] == role and row["probe_sha256"] == digest]
                need(len(matches) == 1, "unique same-action control")
                row = matches[0]
                need(all(row[k] == str(anchor[k]) for k in
                         ("root_slot", "condition", "arm", "task_id", "physical_root_sha256"))
                     and row["probe"] == probe["name"], "control root task condition")
                control_endpoint = csv_metrics(row, "common_endpoint_")
                control_response = csv_metrics(row, "common_delta_own_zero_")
                need((control_endpoint is not None) == (row["common_status"] == "probe_completed"),
                     "control censor semantics")
                controls[role] = {
                    "status": row["common_status"],
                    "prefix_metrics": csv_metrics(row, "prefix_"),
                    "endpoint_metrics": control_endpoint,
                    "response_from_own_zero": control_response,
                    "hybrid_minus_control_endpoint": difference(endpoint, control_endpoint),
                    "hybrid_minus_control_response": difference(
                        difference(endpoint, endpoints[0]), control_response),
                    "own_optimized_suffix_status": row["multistart_status"],
                    "own_optimized_suffix_success": row["multistart_last_native_success"],
                    "own_optimized_suffix_endpoint_metrics": csv_metrics(row, "multistart_endpoint_"),
                }
            output.append({"root_slot": anchor["root_slot"], "task_id": anchor["task_id"],
                           "physical_root_sha256": anchor["physical_root_sha256"],
                           "condition": anchor["condition"], "probe": probe["name"],
                           "probe_sha256": digest, "hybrid_status": status,
                           "hybrid_returned_steps": len(real["responses"]),
                           "hybrid_last_accepted_native_success": (
                               last["native_success"] if last is not None else None),
                           "root_metrics": root, "hybrid_prefix_metrics": prefix,
                           "hybrid_prefix_change": difference(prefix, root),
                           "hybrid_endpoint_metrics": endpoint,
                           "hybrid_response_from_own_zero": difference(endpoint, endpoints[0]),
                           "hybrid_own_optimized_suffix": {
                               "status": own_row["status"], "returned_suffix": own_row["returned_suffix"],
                               "endpoint_metrics": own_endpoint,
                               "last_accepted_native_success": own_row["last_accepted_state_metrics"]["native_success"]
                                   if own_row["last_accepted_state_metrics"] else None},
                           "controls": controls})
    need(len(output) == 84 and sha(TABLE) == TABLE_SHA, "fixed rows and table drift")
    return {"schema": "p1_hybrid_h10_paired_20261031", "rows": output,
            "counts": verified["counts"], "report_sha256": report_sha, "prepare_sha256": prepare_sha,
            "historical_table_sha256": TABLE_SHA, "independent_suffix_receipt_sha256": CP5_SHA,
            "P1_admission": False,
            "independent_confirmation": False,
            "limitations": "Two known roots; common suffixes share trajectories, not independent samples. "
                           "Common suffix response is not independently optimized planning ability."}


def main():
    parser = argparse.ArgumentParser()
    for option in ("directory", "report-sha256", "prepared", "prepare-sha256", "output"):
        parser.add_argument("--" + option, required=True)
    args = parser.parse_args()
    target = Path(args.output)
    need(target.is_absolute() and target.parent == ROOT / "results"
         and target.name.startswith("p1_hybrid_h10_paired_") and not target.exists(),
         "exclusive output")
    result = build(args.directory, args.report_sha256, args.prepared, args.prepare_sha256)
    with target.open("x") as handle:
        json.dump(result, handle, sort_keys=True, allow_nan=False)
    print(json.dumps({"rows": len(result["rows"]), "sha256": sha(target), "counts": result["counts"]}))


if __name__ == "__main__":
    main()
