"""Join saved midpoint-feedback outcomes on certified historical control identities."""
import argparse
import json
from pathlib import Path

from . import p1_response_h10_control_binding_20261031 as binding
from . import p1_response_h10_paired_table_20261031 as table
from . import p1_response_midpoint_online_20261031 as online

ROOT, need, sha = binding.panel.ROOT, binding.need, binding.sha
NAME = "p1_response_midpoint_controls_20261031"
NEW_RECEIPT = ROOT / "results/p1_response_midpoint_online_consume_20261031_RUN01.json"
OLD_RECEIPT = ROOT / "results/p1_uphill_midpoint_feedback_physical_20261031_main_consume_RUN01.json"
NEW_REPORT = ROOT / "results/p1_response_midpoint_online_20261031_RUN01/run_report.json"
OLD_PLAN = ROOT / "results/p1_uphill_midpoint_feedback_plan_20261031_main_RUN01.json"
SOURCES = (
    (binding.NEW, binding.NEW_SHA), (binding.OLD, binding.OLD_SHA),
    (binding.PAIR, binding.PAIR_SHA),
    (table.NEW_PLAN, table.NEW_PLAN_SHA), (table.OLD_PLAN, table.OLD_PLAN_SHA),
    (NEW_RECEIPT, "496975f25116cbb3ef1b615b72d16e02302ad18c33d8d412f73c47c3fea984c3"),
    (OLD_RECEIPT, "b1e2792472db637e826e120fe170638bdb7d865a3fd22dab2949fc8517ef1c9e"),
    (NEW_REPORT, "9811cc9b32690dcfd052194b04106641acf10cc13a6a7168e3e41b6497c94409"),
    (OLD_PLAN, "8ac6af3059fe7ff70158fdc861f726ec9927f0c0dbc56bdc07c312b8b1ac066a"),
)


def outcome(row, goal):
    status = row["status"]
    need(status in table.STATUSES and
         ((row["completed_endpoint"] is not None) == (status == "probe_completed")),
         "endpoint censoring")
    return {
        "status": status,
        "returned_suffix": row["returned_suffix"],
        "fixed_horizon_metrics": table.physical_metrics(row["completed_endpoint"], goal),
        "last_accepted_metrics": row["last_accepted_state_metrics"],
    }


def build(new, old, pairing, new_plan, old_plan, new_receipt, old_receipt, report, feedback_plan):
    binding.verify(new, old, pairing)
    table.verify_planning_protocol(new, new_plan, old_plan)
    need(new_receipt["status"] == "saved_checks_passed" and new_receipt["fixed_rows"] == 16
         and len(new_receipt["rows"]) == 13 and old_receipt["fixed_rows"] == 24
         and len(old_receipt["rows"]) == len(feedback_plan["rows"]) == 24,
         "receipt roster")
    need(feedback_plan["status"] == "complete"
         and report["model_before"] == report["model_after"] == feedback_plan["model_before"]
         == feedback_plan["model_after"], "same frozen model")
    calls = {"encoder": 2, "predict": 22, "action_encoder": 22, "backward": 20}
    need(feedback_plan["per_row_calls_code_path"] == calls
         and len(report["online_events"]) == 13
         and all(event["calls_code_path"] == calls for event in report["online_events"]),
         "midpoint compute")
    current = {row["index"]: row for row in new_receipt["rows"]}
    need(len(current) == 13, "unique new indices")
    rows = []
    for anchor in new["anchors"]:
        index, pair = anchor["index"], anchor["logical_pair"]
        if index not in current:
            continue
        row = current[index]
        need(all(row[k] == anchor[k] for k in ("index", "root_slot", "condition", "arm", "role")),
             "new identity")
        controls = {}
        for role, offset in (("zero", 1), ("greedy", 2)):
            old_index = 3 * pair + offset
            control = old_receipt["rows"][old_index]
            plan = feedback_plan["rows"][old_index]
            historical_anchor = old["anchors"][old_index]
            need(control["index"] == plan["index"] == historical_anchor["index"] == old_index
                 and plan["logical_pair"] == pair and control["role"] == plan["role"]
                 == historical_anchor["role"] == role
                 and all(control[k] == anchor[k] for k in ("root_slot", "condition", "arm")),
                 "historical identity")
            control_result = outcome(control, anchor["goal"])
            current_result = outcome(row, anchor["goal"])
            controls[role] = {
                "index": old_index,
                "prefix_metrics": table.physical_metrics(historical_anchor["checkpoint"]["state"], anchor["goal"]),
                "outcome": control_result,
                "new_minus_control_fixed_endpoint": table.paired_difference(
                    current_result["fixed_horizon_metrics"], control_result["fixed_horizon_metrics"]),
            }
        rows.append({
            "index": index, "logical_pair": pair, "role": anchor["role"],
            "root_slot": anchor["root_slot"], "condition": anchor["condition"], "arm": anchor["arm"],
            "root_metrics": table.physical_metrics(anchor["root_state"], anchor["goal"]),
            "prefix_metrics": table.physical_metrics(anchor["checkpoint"]["state"], anchor["goal"]),
            "outcome": outcome(row, anchor["goal"]), "controls": controls,
        })
    need(len(rows) == 13, "joined roster")
    return {"schema": NAME, "rows": rows, "fixed_rows": 16,
            "excluded": [i for i in range(16) if i not in current],
            "P1_admission": False, "independent_confirmation": False, "equal_budget_certified": False,
            "limitations": "Known roots and repeated tasks; historical controls use offline midpoint "
            "planning with physical replay. Online new candidate and historical controls have "
            "matched suffix optimizer contracts, not certified equal full-method budgets. "
            "No fixed-horizon endpoint for early stop or censoring."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    need(output.is_absolute() and output.parent == ROOT / "results"
         and output.name.startswith(NAME + "_") and not output.exists()
         and not output.is_symlink(), "exclusive output")
    need(all(sha(path) == digest for path, digest in SOURCES), "source hashes")
    read = binding.panel.physical.bundle.physical.prior.native.read
    values = [read(path, cap=64 * 1024**2) for path, _ in SOURCES]
    result = build(*values)
    need(all(sha(path) == digest for path, digest in SOURCES), "source drift")
    result["source_sha256"] = {str(path): digest for path, digest in SOURCES}
    result["analysis_sha256"] = sha(__file__)
    digest = binding.panel.physical.bundle.physical.write(output, result)
    print(json.dumps({"rows": len(result["rows"]), "sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
