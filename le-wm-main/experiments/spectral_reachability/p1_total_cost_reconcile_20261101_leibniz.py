"""Reconcile saved physical/visual operations and node reservations, not GPU cost."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .p1_common_cp10_pair_audit_20261101_leibniz import (
    RESULTS, chain, digest, filehash, ledger_check, require,
)

ROSTER = {
    "root0_structured": ("codex_p1_common_structured_cp10_host_20261101_exclusive_RUN_CODEX01", "6c49744a051ab327ca91c22f0aa5c19261eacff51c472fdf0b1cd4915d80f4cd"),
    "root0_cem": ("p1_common_budget_cem_host_20261101_leibniz_root0_RUN_LEIBNIZ01", "28d1df63a0aa883256229d1a7fb5a9d2b78e96086f79b29ef9205703dee8a6c2"),
    "root1_cem": ("p1_common_budget_cem_host_20261101_leibniz_root1_RUN_MAIN02", "b9f2d1a5a5b7653ec0ddaaebe2608bd72efd1a886d72441d4a4cb1b9de503479"),
}


def physical_census(events, report):
    """Check every operation pair and preserve attempted vs returned, including nested ops."""
    counts = {k: {"attempted": 0, "completed": 0} for k in report["counts"]}
    pending, serial = {}, -1
    for e in events:
        if e["kind"] not in ("operation_attempted", "operation_completed"):
            continue
        op = {k: e[k] for k in ("operation_id", "episode_id", "resource", "units", "region")}
        ident, resource, units = op["operation_id"], op["resource"], op["units"]
        require(resource in counts and type(units) is int and units > 0, "operation resource/units")
        if e["kind"] == "operation_attempted":
            require(type(ident) is int and ident > serial, "duplicate/nonmonotonic operation")
            serial = ident
            pending[ident] = op
            counts[resource]["attempted"] += units
        else:
            require(pending.pop(ident, None) == op, "operation return mismatch")
            counts[resource]["completed"] += units
    require(events[-1]["kind"] == "run_finished" and events[-1]["counts"] == counts
            and counts == report["counts"]
            and {str(k): v for k, v in pending.items()} == {str(k): v for k, v in report["pending"].items()},
            "physical census/terminal mismatch")
    return counts


def reconcile(name, meta):
    folder = RESULTS / meta[0]
    report_path = folder / "report.json"
    require(filehash(report_path) == meta[1], "pinned report drift")
    report = json.loads(report_path.read_bytes())
    prep_path = Path(report["prepared"])
    require(prep_path.parent == RESULTS and filehash(prep_path) == report["prepare_sha256"], "prepare binding")
    prepared = json.loads(prep_path.read_bytes())
    require(report["protocol"] == prepared["protocol"], "protocol binding")
    for path, sha in prepared["sources"].items():
        require(filehash(path) == sha, f"source drift: {path}")
    journal = folder / "journal.jsonl"
    require(filehash(journal) == report["journal_sha256"], "physical journal hash")
    events = chain(journal, "physical")
    require(events[-1]["event_sha256"] == report["journal_chain_tail"]
            and events[-1]["nodes_sha256"] == digest(report["nodes"]), "physical terminal binding")
    physical = physical_census(events, report)
    require(len(report["rows"]) == 1 and report["model_before"] == report["model_after"], "row/model")
    row = report["rows"][0]
    nodes = report["nodes"][row["episode_id"]]
    require([n["boundary"] for n in nodes] == [0, 5, 10, 15], "node roster")
    charges, started = Counter(), []
    for node in nodes:
        if node["status"] == "not_started":
            require(node["receipt"] is None, "unstarted receipt")
            continue
        require(node["status"] == "sealed", "unsealed node")
        started.append(node["boundary"])
        charges.update(ledger_check(node, report["protocol"]["common_node_caps"],
                                   report["cp10_goal_binding"],
                                   report["model_before"]["original_state_dict"], folder))
    require(started == list(range(0, row["control_returned"], 5)), "node/control roster")
    require(physical["encoder"]["completed"] == len(row["encodings"]), "RGB encode count")
    return {
        "report_sha256": meta[1], "sources_checked": len(prepared["sources"]),
        "root_goal": report["cp10_goal_binding"], "model": report["model_before"],
        "control_returned": row["control_returned"], "episode_status": row["status"],
        "started_boundaries": started, "physical_visual_attempted_completed": physical,
        "node_charged_upper_bounds": dict(charges),
        "node_account_status": "closed_unverified",
        "upstream_target_model_loading_prepare": None,
        "gpu_time_flops_and_coverage": None,
        "full_cost_certified": False, "fairness_proven": False,
    }


def run():
    arms = {name: reconcile(name, meta) for name, meta in ROSTER.items()}
    a, b = arms["root0_structured"], arms["root0_cem"]
    require(a["root_goal"] == b["root_goal"] and a["model"] == b["model"], "root0 pair identity")
    require(arms["root1_cem"]["root_goal"] != a["root_goal"], "root1 is not root0")
    return {"schema": "p1_total_cost_reconcile_20261101_leibniz.v1",
            "arms": arms, "root1_structured": None,
            "missing_reason": "no pinned common-budget root1 structured report",
            "interpretation": "separate operation counts and reservations; not additive compute units; upstream unknown",
            "full_cost_certified": False, "fairness_proven": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).absolute()
    require(output.parent == RESULTS and output.name.startswith("p1_total_cost_reconcile_20261101_leibniz_")
            and output.suffix == ".json", "exclusive output path")
    result = run()
    data = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    with output.open("xb") as f:
        f.write(data)
    print(json.dumps({"path": str(output), "sha256": hashlib.sha256(data).hexdigest(),
                      "full_cost_certified": False}))


if __name__ == "__main__":
    main()
