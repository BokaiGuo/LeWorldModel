"""Verify saved hook receipts, ledger, trace selection, and recorded commands."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import codex_four_arm_consumer_20261030 as consumer

NAME = "p1_four_arm_hooked_saved_check_20261030_main"
ROOT = Path(__file__).resolve().parents[2]
ARMS = ("structured", "native", "random", "immediate_progress")


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(directory, expected):
    directory = Path(directory)
    need(directory.is_absolute() and directory.parent == ROOT / "results"
         and directory.is_dir() and not directory.is_symlink(), "directory")
    path = directory / "report.json"
    need(sha(path) == expected, "report hash")
    report = json.loads(path.read_bytes())
    need(report["status"] == "recorded_known_root_not_admitted" and not report["errors"]
         and report["P1_admission"] is False and report["fairness_proven"] is False
         and sha(directory / "journal.jsonl") == report["journal_sha256"], "report status or journal")
    need([r["arm"] for r in report["rows"]] == list(ARMS), "arm roster")
    audits = []
    for row in report["rows"]:
        arm = row["arm"]
        need(row["source_slot"] == 0 and row["initial_exact"] and row["status"] == "completed20"
             and row["control_returned"] == 20, "physical row")
        actions = np.asarray(row["executed_raw"], np.float32)
        need(actions.shape == (20, 2) and np.isfinite(actions).all(), "physical actions")
        nodes = report["nodes"][row["episode_id"]]
        need(len(nodes) == 4, "node count")
        for index, node in enumerate(nodes):
            boundary = index * 5
            need(node["status"] == "sealed" and node["boundary"] == boundary, "node identity")
            receipt = node["receipt"]
            audit_path = directory / f"s0_{arm}_b{boundary:02}_hook_audit.json"
            audit = json.loads(audit_path.read_bytes())
            need(audit == receipt["module_hook_audit"] and audit["arm"] == arm
                 and audit["boundary"] == boundary and audit["full_cost_certified"] is False
                 and audit["fairness_proven"] is False
                 and math.isfinite(audit["synchronized_wall_seconds_including_audit"])
                 and audit["synchronized_wall_seconds_including_audit"] > 0, "hook receipt")
            account = receipt["account"]
            ledger = receipt["ledger"]
            ledger_path = Path(ledger["journal"])
            ns = directory / f"s0_{arm}_b{boundary:02}_common"
            need(ledger_path == ns / "ledger/events.jsonl" and sha(ledger_path) == ledger["sha256"],
                 "ledger path or hash")
            checked = consumer.consume(str(ledger_path), ledger["sha256"],
                                       receipt["contract_sha256"], arm, "node", arm)
            need(checked == account and audit["ledger_charged_upper_bounds"] == account["charged_upper_bounds"]
                 and audit["module_forward_pre_hooks"] == {
                     "predict_calls": account["charged_upper_bounds"]["predict_calls"],
                     "action_encode_calls": account["charged_upper_bounds"]["action_encode_calls"]}, "hook/ledger")
            trace = Path(receipt["trace"])
            need(trace == ns / "candidates.jsonl" and sha(trace) == receipt["trace_sha256"], "trace hash")
            lines = [json.loads(line) for line in trace.read_bytes().splitlines()]
            values = {line["kind"]: line["payload"] for line in lines
                      if line["kind"] in ("optimized_candidates", "final_selection")}
            need(set(values) == {"optimized_candidates", "final_selection"}, "trace entries")
            candidates = np.asarray(values["optimized_candidates"]["raw"], np.float32)
            choice = values["final_selection"]["index"]
            commands = np.asarray(receipt["commands"], np.float32)
            need(candidates.shape == (8, 3, 5, 2) and type(choice) is int and 0 <= choice < 8
                 and choice == receipt["selected_index"] and commands.shape == (5, 2)
                 and commands.tobytes() == candidates[choice, 0].tobytes()
                 and commands.tobytes() == actions[boundary:boundary + 5].tobytes()
                 and values["final_selection"]["scores"] == receipt["scores"], "executed action binding")
            audits.append(audit)
    need(len(audits) == 16 and sha(path) == expected, "complete roster and no drift")
    return {"schema": NAME, "report_sha256": expected, "nodes": 16, "audits": audits,
            "scope": "saved hook receipt, ledger, trace hash and recorded action; not runtime attestation",
            "physical_event_chain_independently_verified": False, "backward_independently_measured": False,
            "full_cost_certified": False, "fairness_proven": False, "P1_admission": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--report-sha", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    need(output.is_absolute() and output.parent == ROOT / "results"
         and output.name.startswith(NAME + "_") and not output.exists(), "exclusive output")
    result = check(args.directory, args.report_sha)
    result["code_sha256"] = sha(Path(__file__))
    raw = json.dumps(result, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    with output.open("xb") as f:
        need(f.write(raw) == len(raw), "short write")
    print(json.dumps({"nodes": result["nodes"], "sha256": hashlib.sha256(raw).hexdigest()}))


if __name__ == "__main__":
    main()
