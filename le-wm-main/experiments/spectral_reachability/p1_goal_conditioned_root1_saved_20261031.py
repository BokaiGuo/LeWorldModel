"""Saved ledger, selection and executed-action checks for known root 1."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import codex_four_arm_consumer_20261030 as consumer
from . import p1_four_arm_winner_budget_saved_check_20261031 as prior
from . import p1_goal_conditioned_spectral_20261031 as control

ROOT, need, sha = prior.ROOT, prior.need, prior.sha
NAME = "p1_goal_conditioned_root1_saved_20261031"


def check(directory, expected):
    directory = Path(directory)
    need(directory.is_absolute() and directory.parent == ROOT / "results" and
         directory.is_dir() and not directory.is_symlink(), "directory")
    path = directory / "report.json"
    need(sha(path) == expected, "report hash")
    report = json.loads(path.read_bytes())
    need(report["status"] == "recorded_known_root_not_admitted" and not report["errors"]
         and report["P1_admission"] is False and report["fairness_proven"] is False
         and report["host"]["device"] == "cuda:0"
         and report["model_before"] == report["model_after"]
         and sha(directory / "journal.jsonl") == report["journal_sha256"], "report/journal")
    need([r["arm"] for r in report["rows"]] == ["structured"], "arm roster")
    audits = []
    for row in report["rows"]:
        arm = row["arm"]
        need(row["source_slot"] == 1 and row["initial_exact"] and row["status"] == "completed20"
             and row["control_returned"] == 20, "physical row")
        actions = np.asarray(row["executed_raw"], np.float32)
        need(actions.shape == (20, 2) and np.isfinite(actions).all(), "physical actions")
        nodes = report["nodes"][row["episode_id"]]
        need(len(nodes) == 4, "node count")
        for index, node in enumerate(nodes):
            boundary = index * 5
            need(node["status"] == "sealed" and node["boundary"] == boundary, "node identity")
            receipt = node["receipt"]
            audit = json.loads((directory / f"s1_{arm}_b{boundary:02}_hook_audit.json").read_bytes())
            need(audit == receipt["module_hook_audit"] and audit["arm"] == arm
                 and audit["root_slot"] == 1 and audit["boundary"] == boundary
                 and audit["full_cost_certified"] is False and audit["fairness_proven"] is False,
                 "hook receipt")
            ns = directory / f"s1_{arm}_b{boundary:02}_winner"
            ledger = receipt["ledger"]
            ledger_path = Path(ledger["journal"])
            need(ledger_path == ns / "ledger/events.jsonl" and sha(ledger_path) == ledger["sha256"],
                 "ledger identity")
            account = consumer.consume(str(ledger_path), ledger["sha256"], receipt["contract_sha256"],
                                       arm, "node", arm)
            costs = control.cost(arm)
            need(account == receipt["account"] and account["charged_upper_bounds"] == costs
                 and audit["ledger_charged_upper_bounds"] == costs
                 and audit["module_forward_pre_hooks"] == {
                     "predict_calls": costs["predict_calls"], "action_encode_calls": costs["action_encode_calls"]},
                 "hook/ledger")
            need(audit["autograd_api_counts"] == {
                "grad_attempted": costs["backward_calls"],
                "grad_completed": costs["backward_calls"], "backward_attempted": 0}
                and audit["backward_independently_measured"] == "python_api_only",
                "autograd API counts")
            trace = Path(receipt["trace"])
            need(trace == ns / "candidates.jsonl" and sha(trace) == receipt["trace_sha256"], "trace identity")
            lines = [json.loads(line) for line in trace.read_bytes().splitlines()]
            relevant = [x for x in lines if x["kind"] in
                        ("node_header", "optimized_candidates", "winner_refinement", "final_selection")]
            values = {x["kind"]: x["payload"] for x in relevant}
            expected_kinds = {"node_header", "optimized_candidates", "final_selection"}
            if arm in control.EXTRA_ARMS:
                expected_kinds.add("winner_refinement")
            need(len(relevant) == len(expected_kinds) and set(values) == expected_kinds
                 and values["node_header"]["extra_policy"] ==
                 ("model_winner" if arm in control.EXTRA_ARMS else "none"), "trace entries")
            candidates = np.asarray(values["optimized_candidates"]["raw"], np.float32)
            choice = values["final_selection"]["index"]
            need(candidates.shape == (8, 3, 5, 2) and type(choice) is int and 0 <= choice < 8,
                 "candidate shape and index")
            if arm in control.EXTRA_ARMS:
                refinement = values["winner_refinement"]
                need(refinement["extra_rounds"] == control.EXTRA_ROUNDS
                     and refinement["index"] == choice
                     and refinement["accepted"] == (refinement["after_score"] < refinement["before_score"])
                     and candidates[choice].tobytes() == np.asarray(refinement["before_raw"], np.float32).tobytes(),
                     "refinement identity")
                if refinement["accepted"]:
                    candidates[choice] = np.asarray(refinement["after_raw"], np.float32)
                need(values["final_selection"]["scores"][choice] ==
                     (refinement["after_score"] if refinement["accepted"] else refinement["before_score"]),
                     "refinement score")
            commands = np.asarray(receipt["commands"], np.float32)
            need(choice == receipt["selected_index"] and commands.shape == (5, 2)
                 and commands.tobytes() == candidates[choice, 0].tobytes()
                 and commands.tobytes() == actions[boundary:boundary + 5].tobytes()
                 and values["final_selection"]["scores"] == receipt["scores"], "action binding")
            audits.append(audit)
    need(len(audits) == 4 and sha(path) == expected, "complete/no drift")
    return {"schema": NAME, "report_sha256": expected, "nodes": len(audits), "audits": audits,
            "physical_event_chain_independently_checked": False, "full_cost_certified": False,
            "fairness_proven": False, "P1_admission": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--report-sha", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    need(output.is_absolute() and output.parent == ROOT / "results"
         and output.name.startswith(NAME + "_") and not output.exists() and not output.is_symlink(),
         "exclusive output")
    result = check(args.directory, args.report_sha)
    result["code_sha256"] = sha(Path(__file__))
    encoded = json.dumps(result, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    with output.open("xb") as stream:
        need(stream.write(encoded) == len(encoded), "short write")
    print(json.dumps({"nodes": result["nodes"], "sha256": sha(output)}))


if __name__ == "__main__":
    main()
