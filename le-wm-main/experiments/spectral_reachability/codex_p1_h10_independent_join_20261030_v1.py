"""Join one independent cp5 suffix with 42 shared suffixes per fixed anchor.

This is a source/identity join, not an independent verification of H10 physics.
It never repeats an independent endpoint as 42 independent observations.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from . import codex_p1_h10_paired_table_20261030_main as main_table
from . import codex_p1_cp5_physical_24_20261030_v1 as physical

ROOT = physical.ROOT
NAME = "codex_p1_h10_independent_join_20261030_v1"
TABLE = ROOT/"results/codex_p1_h10_paired_table_20261030_main_FULL01.json"
H10_PREP = ROOT/"results/codex_p1_three_prefix_h10_runner_20261030_prepare_CODEX01.json"
H10_SHA = "ee136f0a7537c8328aa4ec7417e6d4d0e0c1324c50dc5f59bc4e0c871f020f30"
H10_RUN = ROOT/"results/codex_p1_three_prefix_h10_runner_20261030_run_MAIN02"
PHYS_PREP = ROOT/"results/codex_p1_cp5_physical_24_20261030_v1_prepare_CODEX01.json"
PHYS_PREP_SHA = "367091192c51c3be2565ee8c7d811cde164c155a4720c0c19531836c8b3fb737"
PHYS_DIR = ROOT/"results/codex_p1_cp5_physical_24_20261030_v1_run_CODEX01"
PHYS_RUN_SHA = "c11f4e79f999ee9e89ea94ef05cb093a3e32ed1beb1cfe5c9bbfb33d7f29df6a"
RECEIPT = ROOT/"results/codex_p1_cp5_physical_saved_20261030_v1_CODEX01.json"
RECEIPT_SHA = "0585d0ecb6299d50491f330da46d8011e8a2669979ce6347617027b67e4984b9"
PLAN_RECEIPT_SHA = physical.RECEIPT_SHA
need = physical.need
sha = physical.sha


def read(path, expected):
    path = Path(path)
    need(path.is_file() and not path.is_symlink() and sha(path) == expected, "missing or changed source: "+str(path))
    data = json.loads(path.read_bytes())
    need(sha(path) == expected, "source drift: "+str(path))
    return data


def cp5(row, anchor, directory):
    need(row["start_verified"] is True and row["start_state"] == anchor["checkpoint"]["state"], "cp5 state")
    matches = [v for v in row["checkpoints"] if v["step"] == 5]
    need(len(matches) == 1, "unique cp5 checkpoint")
    checkpoint = matches[0]
    for key in ("step", "state", "snapshot", "file_sha256", "pixel_sha256"):
        need(checkpoint[key] == anchor["checkpoint"][key], "cp5 "+key)
    name = checkpoint["png"]
    need(type(name) is str and Path(name).name == name, "cp5 file name")
    need(sha(directory/name) == checkpoint["file_sha256"], "cp5 actual frame")


def branch_counts(receipts, row, anchor):
    counts = {key: {"attempted": 0, "completed": 0} for key in physical.CAPS}
    for rec in receipts:
        if rec.get("branch") != row["index"]:
            continue
        need(rec.get("anchor_index") == anchor["index"] and rec.get("condition") == anchor["condition"]
             and rec.get("arm") == anchor["arm"] and rec.get("candidate_role") == anchor["role"]
             and rec.get("probe") == row["probe"] and rec["resource"] in counts
             and rec["status"] in ("returned", "failed", "pending"), "common resource identity")
        item = counts[rec["resource"]]
        item["attempted"] += 1
        item["completed"] += rec["status"] == "returned"
    need(counts["public_step"]["completed"] == row["warmup_returned"] + row["own_returned"]
         and counts["constructor"]["attempted"] == 1 and counts["close"]["attempted"] == 1,
         "common per-branch cost")
    return counts


def join(table, prepared, individual, receipt, common_shards, individual_shards, planner_rows):
    original = prepared["original"]
    anchors, banks = original["anchors"], original["probe_banks"]
    need(table["schema"] == main_table.NAME and table["prepare_sha256"] == H10_SHA
         and table["P1_admission"] is False and table["equal_budget_certified"] is False
         and len(table["rows"]) == 1008 and len(anchors) == 24 and len(banks) == 8
         and len(common_shards) == len(individual_shards) == len(planner_rows) == 24
         and len(individual["roster"]) == len(receipt["rows"]) == 24
         and individual["h10_prepare_sha256"] == physical.h10.INPUT_SHA
         and receipt["status"] == "saved_checks_passed" and receipt["fixed_rows"] == 24,
         "complete fixed sources")
    entries = []
    # Table order is logical pair, common probe, prefix role.
    for index, anchor in enumerate(anchors):
        fixed, own = individual["roster"][index], receipt["rows"][index]
        shard, ind_shard, plan = common_shards[index], individual_shards[index], planner_rows[index]
        identity = ("index", "logical_pair", "root_slot", "condition", "arm", "role", "task_id",
                    "prefix5_float32_hex")
        need(all(anchor[k] == fixed[k] for k in identity) and own["index"] == index
             and all(own[k] == fixed[k] for k in ("root_slot", "condition", "arm", "role"))
             and fixed["cp5_pixel_sha256"] == anchor["checkpoint"]["pixel_sha256"]
             and fixed["cp5_file_sha256"] == anchor["checkpoint"]["file_sha256"], "independent anchor identity")
        need(own["status"] in ("probe_completed", "native_stopped", "censored_out_of_domain")
             and ind_shard["row"]["status"] == own["status"] and ind_shard["counts"] == own["counts"]
             and ind_shard["raw10_sha256"] == fixed["raw10_sha256"], "independent physical identity")
        actual_raw = bytes.fromhex(fixed["raw10_float32_hex"])
        need(len(actual_raw) == 80 and hashlib.sha256(actual_raw).hexdigest() == fixed["raw10_sha256"],
             "independent action hash")
        need(plan["kind"] == "row_complete" and plan["index"] == index
             and plan["result"]["index"] == index, "planner row identity")
        import numpy as np
        raw = np.asarray(plan["result"]["actions"], dtype="<f4")
        need(raw.shape == (1, 2, 5, 2) and raw.tobytes() == actual_raw, "planned/executed action bytes")
        need(shard["status"] == "recorded_not_consumed" and shard["anchor"] == index
             and len(shard["rows"]) == 42, "common shard")
        common = []
        for j, probe in enumerate(banks[anchor["logical_pair"]]):
            row = shard["rows"][j]
            tab = table["rows"][anchor["logical_pair"]*126 + j*3 + index%3]
            need(probe["index"] == j and row["index"] == j and row["probe"] == probe["name"]
                 and all(tab[k] == anchor[k] for k in ("root_slot", "condition", "arm", "role", "logical_pair"))
                 and tab["probe"] == probe["name"] and tab["probe_sha256"] == probe["raw10_sha256"]
                 and tab["status"] == row["status"] and tab["independently_optimized_suffix"] is None,
                 "common row identity/action/status")
            need(hashlib.sha256(bytes.fromhex(probe["raw10_float32_hex"])).hexdigest() ==
                 probe["raw10_sha256"], "common raw10 digest")
            cp5(row, anchor, H10_RUN/f"anchor_{index:02d}")
            common.append({"probe_index": j, "probe": probe["name"], "raw10_sha256": probe["raw10_sha256"],
                           "status": tab["status"], "observation": tab["observation"],
                           "endpoint_metrics": tab["common_suffix_endpoint_native_metrics"],
                           "actual_physical_resources": branch_counts(shard["receipts"], row, anchor)})
        cp5(ind_shard["row"], anchor, PHYS_DIR/f"row_{index:02d}")
        entries.append({"identity": {"index": index, "logical_pair": anchor["logical_pair"],
                       "root_slot": anchor["root_slot"], "physical_root_sha256": anchor["physical_root_sha256"],
                       "condition": anchor["condition"], "arm": anchor["arm"], "role": anchor["role"],
                       "task_id": anchor["task_id"], "prefix5_float32_hex": anchor["prefix5_float32_hex"],
                       "cp5_file_sha256": fixed["cp5_file_sha256"], "cp5_pixel_sha256": fixed["cp5_pixel_sha256"],
                       "cp5_state": anchor["checkpoint"]["state"],
                       "cp5_snapshot_sha256": physical.plans.run.spec.digest(anchor["checkpoint"]["snapshot"])},
                       "independent_suffix": {"method": "independent_cp5_cuda_optimized",
                           "status": own["status"], "raw10_sha256": fixed["raw10_sha256"],
                           "raw10_float32_hex": fixed["raw10_float32_hex"],
                           "planning_row_sha256": fixed["plan_row_sha256"],
                           "completed_endpoint": own["completed_endpoint"],
                           "endpoint_metrics": main_table.physical_metrics(own["completed_endpoint"], anchor["goal"]),
                           "last_accepted_state_metrics": own["last_accepted_state_metrics"],
                           "actual_physical_resources": own["counts"],
                           "actual_planning_resources": plan["result"]["ledger_receipt"]},
                       "common_suffixes": common})
    need(len(entries) == 24 and all(len(e["common_suffixes"]) == 42 for e in entries), "full denominator")
    return entries


def build(table_sha):
    code_paths = [Path(__file__).resolve(), Path(main_table.__file__).resolve()]
    code_hashes = {str(path): sha(path) for path in code_paths}
    table = read(TABLE, table_sha)
    prepared = read(H10_PREP, H10_SHA)
    individual = read(PHYS_PREP, PHYS_PREP_SHA)
    receipt = read(RECEIPT, RECEIPT_SHA)
    planner_receipt = read(physical.RECEIPT, PLAN_RECEIPT_SHA)
    common_report = read(H10_RUN/"run_report.json", table["run_report_sha256"])
    own_report = read(PHYS_DIR/"run_report.json", PHYS_RUN_SHA)
    need(common_report["status"] == "recorded_not_consumed" and len(common_report["shards"]) == 24
         and own_report["status"] == "recorded_not_consumed" and len(own_report["shards"]) == 24
         and own_report["prepare_sha256"] == PHYS_PREP_SHA and len(planner_receipt["rows"]) == 24,
         "anchored runs")
    common_shards, individual_shards, planner_rows = [], [], []
    for i in range(24):
        cs, ps, pr = common_report["shards"][i], own_report["shards"][i], planner_receipt["rows"][i]
        need(cs["anchor"] == ps["index"] == pr["index"] == i, "ordered report shards")
        common_shards.append(read(H10_RUN/f"anchor_{i:02d}"/"report.json", cs["report_sha256"]))
        individual_shards.append(read(PHYS_DIR/f"row_{i:02d}"/"report.json", ps["report_sha256"]))
        need(pr["row_sha256"] == individual["roster"][i]["plan_row_sha256"], "planning source identity")
        planner_rows.append(read(physical.PLAN_DIR/f"row_{i:02d}_result.json", pr["row_sha256"]))
    rows = join(table, prepared, individual, receipt, common_shards, individual_shards, planner_rows)
    need(sha(TABLE) == table_sha and all(sha(path) == digest for path, digest in code_hashes.items()),
         "table or join code changed")
    return {"schema": NAME, "status": "source_bound_join_not_new_h10_verification",
            "input_sha256": {"common_table": table_sha, "h10_prepare": H10_SHA,
                             "h10_run_report": table["run_report_sha256"],
                             "independent_prepare": PHYS_PREP_SHA, "independent_run_report": PHYS_RUN_SHA,
                             "independent_saved_consumer": RECEIPT_SHA, "planner_saved_receipt": PLAN_RECEIPT_SHA},
            "join_source_sha256": code_hashes,
            "rows": rows, "independent_denominator": 24, "common_denominator": 1008,
            "common_suffix_method": "fixed shared physical probe, not independent optimization",
            "common_preparation_and_total_cost": "unknown_not_zero",
            "independent_planning_accounting": "recorded dispatched boundaries, not complete total cost",
            "limitations": ["known roots and duplicate trajectories", "OS planning isolation uncertified",
                            "join does not replace main thread full H10 verification",
                            "no evidence of equal total budgets or independent confirmation"],
            "P1_admission": False, "equal_budget_certified": False, "independent_confirmation": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table-sha256", required=True)
    parser.add_argument("--output", required=True)
    a = parser.parse_args()
    output = Path(a.output)
    need(output.is_absolute() and output.parent == ROOT/"results" and
         output.name.startswith(NAME+"_") and not os.path.lexists(output), "exclusive output")
    result = build(a.table_sha256)
    digest = physical.h10._write_new(output, result)
    print(json.dumps({"sha256": digest, "anchors": len(result["rows"]),
                      "status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
