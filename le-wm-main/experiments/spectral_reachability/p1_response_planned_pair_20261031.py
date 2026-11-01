"""Pair already-executed RGB-feedback suffixes, without opening the running H10 panel."""
import argparse
import json
from pathlib import Path

from . import p1_response_h10_paired_table_20261031 as table

need, sha = table.need, table.sha
NAME = "p1_response_planned_pair_20261031"
PREP = table.control.NEW
PREP_SHA = table.control.NEW_SHA


def analyze(prepared, old, old_pairing, planned, new_plan, old_plan):
    table.control.verify(prepared, old, old_pairing)
    table.verify_planning_protocol(prepared, new_plan, old_plan)
    need(planned["status"] == "saved_checks_passed" and planned["fixed_rows"] == 16
         and planned["censored_before_cp5"] == prepared["censored_before_cp5"]
         and len(planned["rows"]) == len(prepared["anchors"]) == 14, "planning roster")
    rows = []
    for anchor, execution in zip(prepared["anchors"], planned["rows"], strict=True):
        need(all(anchor[k] == execution[k] for k in ("index", "role", "root_slot", "condition", "arm")),
             "execution identity")
        status, state = execution["status"], execution["completed_endpoint"]
        need(status in table.STATUSES and ((state is not None) == (status == "probe_completed")),
             "censored endpoint")
        pair = anchor["logical_pair"]
        root = table.physical_metrics(anchor["root_state"], anchor["goal"])
        prefix = table.physical_metrics(anchor["checkpoint"]["state"], anchor["goal"])
        endpoint = table.physical_metrics(state, anchor["goal"])
        controls = {}
        for role, offset in (("zero", 1), ("greedy", 2)):
            source = old["anchors"][3 * pair + offset]
            other = old_pairing["anchors"][3 * pair + offset]
            need(source["index"] == other["index"] and other["identity"]["role"] == role,
                 "control identity")
            control_plan = other["common_multistart_adam"]
            need((control_plan["endpoint"] is not None) == (control_plan["status"] == "probe_completed"),
                 "control endpoint censoring")
            controls[role] = {
                "prefix_metrics": table.physical_metrics(source["checkpoint"]["state"], anchor["goal"]),
                "status": control_plan["status"], "endpoint_metrics": control_plan["endpoint"],
                "endpoint_minus_control": table.paired_difference(endpoint, control_plan["endpoint"]),
            }
        rows.append({
            "anchor": anchor["index"], "logical_pair": pair, "root_slot": anchor["root_slot"],
            "condition": anchor["condition"], "arm": anchor["arm"], "role": anchor["role"],
            "root_metrics": root, "prefix_metrics": prefix,
            "prefix_minus_root": table.paired_difference(prefix, root), "status": status,
            "accepted_suffix_steps": execution["returned_suffix"], "endpoint_metrics": endpoint,
            "controls": controls,
        })
    return {
        "schema": NAME, "rows": rows, "censored_before_cp5": prepared["censored_before_cp5"],
        "fixed_prefixes": 16, "P1_admission": False, "equal_budget_certified": False,
        "independent_confirmation": False,
        "limitations": "Exploratory known roots and dependent arms; the second diverse candidate "
                       "is not a certified deployment winner. Same suffix optimizer and starts do "
                       "not account for prefix generation or prove total-budget fairness. "
                       "Fixed-horizon endpoints are not imputed after early stop or censoring.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    need(output.is_absolute() and output.parent == table.panel.ROOT / "results"
         and output.name.startswith(NAME + "_") and not output.exists()
         and not output.is_symlink(), "exclusive output")
    sources = ((PREP, PREP_SHA), (table.control.OLD, table.control.OLD_SHA),
               (table.control.PAIR, table.control.PAIR_SHA), (table.PLANNED, table.PLANNED_SHA),
               (table.NEW_PLAN, table.NEW_PLAN_SHA), (table.OLD_PLAN, table.OLD_PLAN_SHA))
    need(all(sha(path) == digest for path, digest in sources), "source hash")
    read = table.panel.physical.bundle.physical.prior.native.read
    values = [read(path, cap=64 * 1024**2) for path, _ in sources]
    result = analyze(*values)
    need(all(sha(path) == digest for path, digest in sources), "source drift")
    result["source_sha256"] = {str(path): digest for path, digest in sources}
    result["analysis_sha256"] = sha(__file__)
    digest = table.panel.physical.bundle.physical.write(output, result)
    print(json.dumps({"rows": len(result["rows"]), "sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
