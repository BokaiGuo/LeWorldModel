"""Join two pinned, pre-existing call inventories; this does not certify history."""
import argparse
import json
import os
from pathlib import Path

from experiments.spectral_reachability import agent_freshroot_authority_20261028_v1 as authority

NAME = "p1_call_evidence_join_20261101_hypatia"
RECOVERY = authority.ROOT / "results/codex_reset_ledger_recovery_map_20261030_LOCAL04.json"
RECOVERY_SHA = "d101bc2cad391b004998324ba31f8cc8bca3a02ef10c74a6b5f518851dce0582"


def join(request, recovery):
    authority.need(request["known_dispatch_count"] == 10 and len(request["sources"]) == 8,
                   "request denominator")
    calls = recovery["calls"]
    authority.need(recovery["known_calls"] == 10 and len(calls) == 10, "recovery denominator")
    by_id = {c["call_id"]: c for c in calls}
    authority.need(len(by_id) == 10, "duplicate recovery call")
    rows = []
    for source in request["sources"]:
        for dispatch in source["dispatches"]:
            call = by_id.pop(dispatch["call_id"], None)
            authority.need(call is not None, "missing or duplicate request call")
            report = call["reported_family_source"]
            authority.need(call["source_id"] == source["source_id"]
                           and report["path"] == source["report"]["path"]
                           and report["sha256"] == source["report"]["sha256"]
                           and call["dispatch"]["line"] == dispatch["line"], "cross-source mismatch")
            rows.append({
                "source_id": source["source_id"], "call_id": dispatch["call_id"],
                "dispatch": dispatch, "report_family": source["report"],
                "report_binding": report["binding"],
                "process_status": call["process_status"], "exit_code": call["exit_code"],
                "return_receipts": call["return_receipts"],
                "missing_evidence": call["missing_evidence"],
                "reset_attempt_ledger": call["reset_attempt_ledger"],
                "loaded_closure_at_run": call["loaded_closure_at_run"],
                "restart_and_scope_complete": call["restart_and_scope_complete"],
            })
    authority.need(not by_id and len(rows) == 10, "unmatched calls")
    authority.need([r["source_id"] for r in rows].count("B") == 3, "B calls")
    return {
        "schema": NAME + ".v1", "request_semantic_sha256": authority.sha(authority.canonical(request)),
        "recovery_sha256": RECOVERY_SHA, "calls": rows,
        "scope": "join of already recorded dispatches, not a search for all calls",
        "limitations": ["family report join is not a run-to-loaded-code binding",
                        "process evidence does not establish reset count or restart completeness",
                        "training exposure and global historical coverage remain unknown"],
        "history_complete": False, "training_disjoint": "unknown",
        "freshness_certified": False, "source_run_authorized": False, "P1_admitted": False,
    }


def generate():
    reader = authority.Reader()
    catalog = authority.parse(reader.read(str(authority.CATALOG), authority.CATALOG_SHA, authority.CAP))
    recovery = authority.parse(reader.read(str(RECOVERY), RECOVERY_SHA, authority.CAP))
    return join(authority.requests(catalog), recovery)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    out = parser.parse_args().output
    authority.need(out.is_absolute() and out.parent.resolve() == out.parent
                   and out.name.startswith(NAME + "_") and out.suffix == ".json", "output namespace")
    data = (json.dumps(generate(), sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with out.open("xb") as stream:
        authority.need(stream.write(data) == len(data), "short write")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"output": str(out), "sha256": authority.sha(data), "freshness_certified": False}))


if __name__ == "__main__":
    main()
