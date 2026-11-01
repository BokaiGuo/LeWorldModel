"""CPU-only saved action-seal and physical-chain audit of one root1 run."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import codex_p1_two_root_saved_audit_20261101_exclusive as a
from . import p1_carry_binding_20261101 as binding


def audit(folder, report_sha):
    folder = Path(folder).resolve(strict=True)
    report_path = folder / "report.json"
    a.require(a.filehash(report_path) == report_sha, "report hash")
    report = json.loads(report_path.read_bytes())
    a.require(report["status"] == "recorded_known_root_not_admitted" and
              report["P1_admission"] is False and report["fairness_proven"] is False
              and not report["errors"] and not report["pending"], "report status")
    a.require(a.filehash(report["prepared"]) == report["prepare_sha256"], "prepare hash")
    prepared = json.loads(Path(report["prepared"]).read_bytes())
    a.require(prepared["cp10_identity"] == report["cp10_goal_binding"]
              and prepared["protocol"] == report["protocol"]
              and report["model_before"] == report["model_after"], "identity")
    rows = report["rows"]
    a.require(len(rows) == 1 and rows[0]["source_slot"] == 1 and
              rows[0]["arm"] == "structured" and rows[0]["initial_exact"], "root1 row")
    row = rows[0]
    physical_path = folder / "journal.jsonl"
    physical = a.chain(physical_path, "physical")
    a.require(a.filehash(physical_path) == report["journal_sha256"] and
              physical[-1]["event_sha256"] == report["journal_chain_tail"], "physical chain")
    actions = row["executed_raw"]
    dispatch = [e for e in physical if e["kind"] == "authorized_dispatch"]
    returns = [e for e in physical if e["kind"] == "control_return"]
    a.require(len(actions) == row["control_returned"] == len(dispatch) == len(returns), "action count")
    sealed = {}
    for node in report["nodes"][row["episode_id"]]:
        if node["status"] == "not_started":
            continue
        a.require(node["status"] == "sealed", "node not sealed")
        b, receipt = node["boundary"], node["receipt"]
        envelope = json.loads((folder / f"{row['episode_id']}_b{b:02}_seal.json").read_bytes())
        a.require(envelope["receipt"] == receipt, "saved node seal")
        trace = Path(receipt["trace"])
        a.require(trace.resolve().is_relative_to(folder) and a.filehash(trace) == receipt["trace_sha256"],
                  "trace identity")
        events = a.chain(trace, "ledger")
        selected = [e["payload"] for e in events if e["kind"] == "selected_full_plan"]
        a.require(len(selected) == 1, "selected plan event")
        plan = np.asarray(receipt["selected_full_plan"], np.float32)
        a.require(binding.BoundPlan.seal(row["episode_id"], b, plan).plan_sha256
                  == receipt["selected_full_plan_sha256"] == selected[0]["plan_sha256"]
                  and np.array_equal(plan, np.asarray(selected[0]["raw"], np.float32))
                  and np.array_equal(plan[0], np.asarray(receipt["commands"], np.float32)), "plan binding")
        seals = [e for e in physical if e["kind"] == "boundary_sealed" and e["boundary"] == b]
        a.require(len(seals) == 1 and seals[0]["plan_sha256"] == a.digest(receipt),
                  "physical seal")
        sealed[b] = (receipt["commands"], seals[0]["index"])
    for index, action in enumerate(actions):
        b = index // 5 * 5
        a.require(b in sealed and dispatch[index]["step"] == returns[index]["step"] == index + 1
                  and sealed[b][1] < dispatch[index]["index"] < returns[index]["index"]
                  and dispatch[index]["command"] == returns[index]["command"] == action == sealed[b][0][index % 5],
                  "sealed action or event order")
    return {"status": "saved_action_seals_consistent", "report_sha256": report_sha,
            "physical_journal_sha256": report["journal_sha256"], "steps": len(actions),
            "sealed_boundaries": sorted(sealed), "row_status": row["status"],
            "P1_admission": False, "fairness_proven": False,
            "boundary": "saved-chain consistency only; does not independently verify total cost or causal outcome"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args.directory, args.report_sha256)
    with Path(args.output).open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
