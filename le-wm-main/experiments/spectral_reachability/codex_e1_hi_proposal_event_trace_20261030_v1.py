"""Saved E1/Hi event projection, not a causal or complete proposal cost ledger."""
import hashlib
import json
from pathlib import Path
import sys

from . import codex_e1_sealed_event_axis_audit_20261030_v1 as util

BASE = Path(__file__).resolve().parents[2] / "results"
REPORT = BASE / "agent_hilewm_rgb_high_low_20261028_v1_cuda_MAIN01/report.json"
RECEIPT = BASE / "agent_hilewm_rgb_high_low_20261028_v1_consume_MAIN01.json"
TARGETS = BASE / "agent_e1_six_slot_common_prepare_20261029_v1_prepare_MAIN01.json"
OUTPUT = BASE / "codex_e1_hi_proposal_event_trace_20261030_v1_LOCAL01.json"
REPORT_SHA = "083d0d9aa190b561e6ccf865b09d7224b97d404b16ba38c792bd8af386f5622a"
RECEIPT_SHA = "40f28414682c9540e9e195b361a36dfc514f816358adbeacb379fedc460e4ef0"
TARGET_SHA = "cd96f6e28b25d3b7e3b6649c760b2f111747aec5c0ce9c8a41741318209df8a0"
MANIFEST_SHA = "a3255d38e9cf718f9560c1b9e381abe38750edd9e91d3f04e5ad332cf414d7e9"
ORDER = ((0, "off"), (0, "on"), (1, "off"), (1, "on"))


def need(ok, text):
    if not ok:
        raise ValueError(text)


def analyze(report, receipt, targets):
    need(report["schema"] == "agent_hilewm_rgb_high_low_20261028_v1.report" and
         report["completed"] is True and report["errors"] == [] and
         report["scientific_admission"] is False and report["manifest_sha256"] == MANIFEST_SHA,
         "report identity")
    need(receipt["schema"] == "agent_hilewm_rgb_high_low_20261028_v1.consume" and
         receipt["report_sha256"] == REPORT_SHA and receipt["manifest_sha256"] == MANIFEST_SHA and
         receipt["scientific_admission"] is False and
         receipt["resources"] == report["resources"], "receipt binding")
    need(targets["schema"] == "agent_e1_six_slot_common_prepare_20261029_v1.prepare",
         "target schema")
    rows = report["rows"]
    need(type(rows) is list and len(rows) == 4 and len(receipt["rows"]) == 4, "four row denominator")
    totals, projections = {}, []
    for i, row in enumerate(rows):
        root, condition = ORDER[i]
        need(type(row["index"]) is int and row["index"] == i and type(row["root_slot"]) is int
             and row["root_slot"] == root and row["condition"] == condition and row["status"] == "returned",
             "row identity")
        rr = receipt["rows"][i]
        need(rr["index"] == i and rr["root_slot"] == root and rr["condition"] == condition, "receipt row")
        events = row["events"]
        need(type(events) is list and bool(events), "events missing")
        per_kind = {}
        for j, event in enumerate(events):
            need(type(event["index"]) is int and event["index"] == j and event["returned"] is True,
                 "event index or incomplete return")
            parent = event["parent"]
            need(parent is None or (type(parent) is int and 0 <= parent < j), "event parent")
            kind = event["kind"]
            need(type(kind) is str and kind in report["resources"]["modules"], "unknown module")
            layouts = event["layouts"]
            need(type(layouts) is list and len(layouts) > 0, "layout missing")
            shape = layouts[0]["shape"]
            need(type(shape) is list and len(shape) > 0 and
                 all(type(d) is int and d > 0 for d in shape), "invalid shape")
            c = per_kind.setdefault(kind, {"attempted": 0, "returned": 0, "input_rows": 0,
                                           "input_context_tokens": 0, "event_indexes": []})
            c["attempted"] += 1
            c["returned"] += 1
            c["input_rows"] += shape[0]
            c["input_context_tokens"] += shape[0] * (shape[1] if len(shape) == 3 else 1)
            c["event_indexes"].append(j)
        for kind, count in per_kind.items():
            for field in ("attempted", "returned", "input_rows", "input_context_tokens"):
                totals.setdefault(kind, {}).setdefault(field, 0)
                totals[kind][field] += count[field]
            indices = count.pop("event_indexes")
            body = {"schema": "codex.e1.hi_row_events.v1", "report_sha256": REPORT_SHA,
                    "row": i, "kind": kind,
                    "events": [events[x] for x in indices]}
            count["event_set_sha256"] = hashlib.sha256(util.canonical(body)).hexdigest()
        projections.append({"row": i, "root": root, "condition": condition, "modules": per_kind,
                            "allocation": "row_execution_only_not_target_causal_cost"})
    need(totals == report["resources"]["modules"], "row sums differ from pinned receipt")
    need(len(targets["rows"]) == 6, "target denominator")
    for target_index, hi_index in ((1, 0), (4, 2)):
        t, rr = targets["rows"][target_index], receipt["rows"][hi_index]
        need(t["arm"] == "existing_learned_prior" and t["root_slot"] == rr["root_slot"] and
             t["origin"]["kind"] == "inherited_hi_FIRST" and
             t["origin"]["report_row"] == hi_index and rr["target_delivered_exact"] is True and
             t["target_data_sha256"] == rr["high_target_sha256"] == rr["low_goal_sha256"] and
             t["blind_sha256"] == rr["RGB_source"]["blind_sha256"], "FIRST provenance mismatch")
    return {"schema": "codex.e1.hi_proposal_event_trace.v1", "status": "incomplete",
            "report_sha256": REPORT_SHA, "receipt_sha256": RECEIPT_SHA, "targets_sha256": TARGET_SHA,
            "row_execution_projections": projections,
            "unallocated_aggregate_modules": report["resources"]["modules"],
            "blockers": ["row_events_not_proven_minimal_or_complete_target_cost",
                         "shared_and_ancestor_cost_allocation_unknown",
                         "arbitrary_and_compiled_proposal_event_closure_not_verified",
                         "full_end_to_end_cost_not_verified", "flops_unknown"],
            "event_semantics": "events embedded in pinned report; digest is newly derived, not producer signed",
            "equal_budget_certified": False, "ranking_authorized": False, "scientific_admission": False}


def load():
    return tuple(util.parse(util.read_bound(path, digest, 4 * 1024 * 1024)) for path, digest in
                 ((REPORT, REPORT_SHA), (RECEIPT, RECEIPT_SHA), (TARGETS, TARGET_SHA)))


def main():
    try:
        need(len(sys.argv) == 1, "no arguments")
        result = analyze(*load())
        raw = json.dumps(result, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
        with OUTPUT.open("xb") as stream:
            stream.write(raw)
        print(json.dumps({"path": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest(),
                          "status": result["status"]}))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc)}), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
