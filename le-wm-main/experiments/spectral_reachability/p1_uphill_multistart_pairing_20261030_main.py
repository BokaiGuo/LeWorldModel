"""Join fixed P1 plan and verified physical results without endpoint imputation."""
import argparse
import hashlib
import json
from pathlib import Path

from . import codex_p1_h10_paired_table_20261030_main as metric
from . import p1_uphill_multistart_physical_20261030_main as physical

ROOT = physical.physical.ROOT
NAME = "p1_uphill_multistart_pairing_20261030_main"
CONSUME = ROOT / "results/p1_uphill_multistart_physical_20261030_main_consume_RUN01.json"
CONSUME_SHA = "575ee720d8282d349ae06f8f06e27e4caa592e65b228637722e9e4ca6183d870"
ANCHORS = ROOT / "results/p1_uphill_h10_20261030_prepare_RUN01.json"
ANCHORS_SHA = "890e38ecf7758ae739c25e9d355533e4d3b31be91a8c063c9625d1886c77d56d"
PREPARED = ROOT / "results/p1_uphill_multistart_physical_20261030_main_prepare_RUN01.json"
PREPARED_SHA = "c58e71cf4a5f42d4ab5668993e95869d64718a3b3d5ed37172816dc980e82ff8"
need = physical.need
ROLES = ("selected", "zero", "greedy")


def read(path, digest):
    need(physical.sha(path) == digest, "input hash")
    return json.loads(path.read_bytes())


def join(anchors, prepared, plan, observed):
    need(len(anchors["anchors"]) == len(prepared["roster"]) == len(plan["rows"])
         == len(observed["rows"]) == 24, "complete roster")
    rows = []
    for i, (anchor, item, proposal, execution) in enumerate(zip(
            anchors["anchors"], prepared["roster"], plan["rows"], observed["rows"], strict=True)):
        need(anchor["index"] == item["index"] == proposal["index"] == execution["index"] == i
             and anchor["role"] == item["role"] == proposal["role"] == execution["role"]
             and anchor["condition"] == item["condition"] == execution["condition"]
             and anchor["root_slot"] == item["root_slot"] == execution["root_slot"]
             and anchor["logical_pair"] == item["logical_pair"] == proposal["logical_pair"]
             and anchor["prefix5_sha256"] == hashlib.sha256(
                 bytes.fromhex(item["prefix5_float32_hex"])).hexdigest()
             and anchor["checkpoint"]["pixel_sha256"] == item["cp5_pixel_sha256"]
             and item["raw10_sha256"] == proposal["selected_action_sha256"], "identity binding")
        need(proposal["selected_index"] == physical.planner.adam.prior.pick(proposal["candidate_scores"]),
             "model winner")
        root = metric.physical_metrics(anchor["root_state"], anchor["goal"])
        prefix = metric.physical_metrics(anchor["checkpoint"]["state"], anchor["goal"])
        endpoint = metric.physical_metrics(execution["completed_endpoint"], anchor["goal"])
        need((endpoint is not None) == (execution["status"] == "probe_completed"), "endpoint censoring")
        last = execution["last_accepted_state_metrics"]
        rows.append({"index": i, "logical_pair": anchor["logical_pair"], "root_slot": anchor["root_slot"],
                     "physical_root_sha256": anchor["physical_root_sha256"], "condition": anchor["condition"],
                     "arm": anchor["arm"], "role": anchor["role"], "task_id": anchor["task_id"],
                     "root": root, "prefix": prefix, "prefix_minus_root": metric.paired_difference(prefix, root),
                     "model_best_score": proposal["candidate_scores"][proposal["selected_index"]],
                     "model_all_scores": proposal["candidate_scores"],
                     "chosen_start": proposal["selected_index"], "status": execution["status"],
                     "returned_suffix": execution["returned_suffix"], "endpoint": endpoint,
                     "last_native_success": None if last is None else last["native_success"],
                     "last_accepted_is_fixed_horizon_endpoint": endpoint is not None,
                     "calls": proposal["calls"], "planning_wall_seconds": proposal["planning_wall_seconds"]})
    for group in range(8):
        trio = rows[3*group:3*group+3]
        need([r["role"] for r in trio] == list(ROLES)
             and all(r["logical_pair"] == group and r["task_id"] == trio[0]["task_id"]
                     and r["physical_root_sha256"] == trio[0]["physical_root_sha256"] for r in trio),
             "paired task identity")
        for row in trio:
            row["endpoint_minus_zero_prefix"] = metric.paired_difference(row["endpoint"], trio[1]["endpoint"])
            row["endpoint_minus_greedy_prefix"] = metric.paired_difference(row["endpoint"], trio[2]["endpoint"])
            row["model_score_minus_zero_prefix"] = row["model_best_score"] - trio[1]["model_best_score"]
    return {"schema": NAME, "rows": rows, "P1_admission": False, "independent_confirmation": False,
            "equal_budget_certified": False,
            "limitations": "Known roots, dependent arms; model predictions are not physical outcomes; "
                           "early stopped fixed-horizon endpoints are null; no causal mechanism attribution."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    need(output.is_absolute() and output.parent == ROOT / "results"
         and output.name.startswith(NAME + "_") and not output.exists() and not output.is_symlink(), "output")
    anchors = read(ANCHORS, ANCHORS_SHA)
    prepared = read(PREPARED, PREPARED_SHA)
    plan = read(physical.PLAN, physical.PLAN_SHA)
    observed = read(CONSUME, CONSUME_SHA)
    value = join(anchors, prepared, plan, observed)
    value["source_hashes"] = {str(p): h for p, h in ((ANCHORS, ANCHORS_SHA), (PREPARED, PREPARED_SHA),
                                  (physical.PLAN, physical.PLAN_SHA), (CONSUME, CONSUME_SHA))}
    need(all(physical.sha(p) == h for p, h in value["source_hashes"].items()), "source drift")
    value["analysis_sha256"] = physical.sha(Path(__file__).resolve())
    sha = physical.physical.design.run._write_new(output, value)
    print(json.dumps({"rows": len(value["rows"]), "sha256": sha}), flush=True)


if __name__ == "__main__":
    main()
