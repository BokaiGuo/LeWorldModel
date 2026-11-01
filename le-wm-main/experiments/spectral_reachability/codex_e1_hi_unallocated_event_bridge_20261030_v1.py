"""Attach verified Hi source-row observations without allocating proposal costs."""
import copy
import json
import sys
import zlib

from . import codex_e1_ancestor_integrated_ledger_20261030_v1 as integrated
from . import codex_e1_hi_proposal_event_trace_20261030_v1 as hi

util = hi.util
BASE_SHA = "0bd6601848d43ed8c5e0763f98e7858a7d6909347046b4a277497eadc0a5988b"
TRACE_SHA = "b4382de707531a59e36bc287f8932656f2cb3186cf552e13e96d5f0f68d1d349"
OUTPUT = hi.BASE / "codex_e1_hi_unallocated_event_bridge_20261030_v1_LOCAL01.json"
# Mapping describes hook events, not complete per-target costs.
HOOKS = {"action_encoder": "action_encoder_calls", "encoder": "image_encoder_calls",
         "high_cost": "high_cost_calls", "high_predictor": "high_predictor_calls",
         "low_predictor": "low_predictor_calls"}


def join(base, trace, targets):
    """Join previously verified data; raw-source verification belongs to load_verified."""
    hi.need(base["schema"] == "codex.e1.ancestor_integrated_ledger.v1" and
            base["status"] == "blocked_incomplete" and
            trace["schema"] == "codex.e1.hi_proposal_event_trace.v1" and
            trace["status"] == "incomplete" and trace["report_sha256"] == hi.REPORT_SHA and
            trace["receipt_sha256"] == hi.RECEIPT_SHA and
            trace["targets_sha256"] == hi.TARGET_SHA, "source identity")
    slots, rows = base["slots"], trace["row_execution_projections"]
    hi.need(len(slots) == len(targets["rows"]) == 6 and len(rows) == 4, "denominator")
    out = copy.deepcopy(base)
    for root, slot_index, row_index in ((0, 1, 0), (1, 4, 2)):
        slot = out["slots"][slot_index]
        item, target, row = slot["ledger_input"], targets["rows"][slot_index], rows[row_index]
        hi.need(slot["slot"] == slot_index and slot["root"] == root and
                item["arm"] == target["arm"] == "existing_learned_prior" and
                target["root_slot"] == root and target["origin"]["kind"] == "inherited_hi_FIRST" and
                target["origin"]["report_row"] == row_index and
                item["target_sha256"] == target["target_data_sha256"] and
                item["task_sha256"] == target["task_sha256"] and
                row["row"] == row_index and row["root"] == root and row["condition"] == "off" and
                row["allocation"] == "row_execution_only_not_target_causal_cost", "row attribution")
        proposal = item["phases"]["proposal"]
        hi.need(proposal["scope"] == "unknown" and proposal["receipt_sha256"] is None and
                all(x is None for x in proposal["measurements"].values()), "proposal already populated")
        axes = {}
        for module, axis in HOOKS.items():
            event = row["modules"][module]
            hi.need(type(event["returned"]) is int and event["returned"] > 0 and
                    event["attempted"] == event["returned"], "invalid recorded event count")
            integrated.ledger.digest(event["event_set_sha256"])
            axes[axis] = {"observed_returned_events": event["returned"],
                          "event_set_sha256": event["event_set_sha256"]}
        pred = row["modules"]["high_predictor"]
        hi.need(type(pred["input_rows"]) is int and pred["input_rows"] > 0, "invalid input rows")
        axes["high_input_rows"] = {"observed_returned_input_rows": pred["input_rows"],
                                    "event_set_sha256": pred["event_set_sha256"]}
        slot["unallocated_hi_source_row"] = {
            "row": row_index, "root": root, "condition": "off",
            "report_sha256": hi.REPORT_SHA, "receipt_sha256": hi.RECEIPT_SHA,
            "trace_sha256": TRACE_SHA, "axes": axes,
            "allocation": "row_execution_only_not_target_causal_cost",
            "failed_attempts": "unknown", "shared_costs": "unknown",
            "event_coverage": "unknown", "cross_stage_overlap": "unknown"}
        slot["ledger_preflight"] = integrated.ledger.inspect(item)
        hi.need(slot["ledger_preflight"]["structural_status"] == "incomplete" and
                "proposal:receipt_missing" in slot["ledger_preflight"]["blockers"],
                "incomplete proposal not blocked")
    out["schema"] = "codex.e1.hi_unallocated_event_bridge.v1"
    out["sources"]["hi_report_sha256"] = hi.REPORT_SHA
    out["sources"]["hi_receipt_sha256"] = hi.RECEIPT_SHA
    out["sources"]["hi_trace_sha256"] = TRACE_SHA
    out["unresolved"] += ["hi_target_causal_allocation", "hi_failed_attempt_coverage",
                          "hi_shared_and_cross_stage_allocation"]
    out["boundary"] = ("Hi row observations unallocated; proposal ledger unknown; "
                       "no totals, no parity, no P1/GPU/physics")
    out["equal_budget_certified"] = out["ranking_authorized"] = out["scientific_admission"] = False
    return out


def load_verified():
    base_saved = util.parse(util.read_bound(integrated.OUTPUT, BASE_SHA, 4 * 1024 * 1024))
    base = integrated.join(*integrated.load_verified())
    hi.need(base == base_saved, "integrated ledger recomputation mismatch")
    saved = util.parse(util.read_bound(hi.OUTPUT, TRACE_SHA, 4 * 1024 * 1024))
    report, receipt, targets = hi.load()
    trace = hi.analyze(report, receipt, targets)
    hi.need(trace == saved, "Hi event trace recomputation mismatch")
    return base, trace, targets


def main():
    try:
        hi.need(len(sys.argv) == 1, "no arguments")
        raw = json.dumps(join(*load_verified()), sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
        with OUTPUT.open("xb") as stream:
            stream.write(raw)
        print(json.dumps({"path": str(OUTPUT), "sha256": util.sha(raw), "status": "blocked_incomplete"}))
    except (ValueError, KeyError, TypeError, OSError, zlib.error) as exc:
        print(json.dumps({"status": "blocked_invalid", "reason": str(exc),
                          "equal_budget_certified": False, "ranking_authorized": False}), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
