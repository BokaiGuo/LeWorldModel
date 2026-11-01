"""Join pinned E1 ledger with partial compiled ancestor observations, fail closed."""
import copy
import json
import sys
import zlib

from . import codex_e1_compiled_ancestor_ledger_20261030_v1 as ancestor
from . import codex_e1_event_ledger_consumer_20261030_v1 as consumer

audit = consumer.audit
ledger = consumer.ledger
SAVED = ancestor.OUTPUT
SAVED_SHA = "6c0149485e6f5238ea932b6eef7fd53d6e3763fd38863300a81a0d66102317fd"
OUTPUT = ancestor.BASE / "codex_e1_ancestor_integrated_ledger_20261030_v1_LOCAL01.json"
MAP = {"predict_API": "predict_api_calls", "predictor_samples": "predictor_samples",
       "action_encoder": "action_encoder_calls", "pred_proj": "pred_proj_calls"}


def join(base, observed):
    """Join already verified in-memory inputs; caller must verify their provenance."""
    consumer.require(base["schema"] == "codex.e1.event_ledger_consumer.v1" and
                     base["status"] == "blocked_incomplete" and
                     observed["schema"] == "codex.e1.compiled_ancestor_ledger.v1" and
                     observed["status"] == "incomplete" and
                     observed["event_sha256"] == ancestor.EVENT_SHA and
                     observed["new_prepare_sha256"] == consumer.TARGETS_SHA,
                     "input identity mismatch")
    consumer.require(len(base["slots"]) == 6 and len(observed["rows"]) == 2,
                     "denominator mismatch")
    result = copy.deepcopy(base)
    rows = {}
    for row in observed["rows"]:
        root = row["root"]
        consumer.require(type(root) is int and root in (0, 1) and root not in rows and
                         row["arm"] == "dynamics_compiled_witness" and
                         set(row["event_axes"]) == set(ancestor.RESOURCES) | {"predictor_samples"},
                         "invalid ancestor row")
        rows[root] = row
    matched = set()
    for index, slot in enumerate(result["slots"]):
        item = slot["ledger_input"]
        consumer.require(slot["slot"] == index and slot["root"] == index // 3 and
                         item["arm"] == audit.ARMS[index % 3], "slot identity mismatch")
        inherited = item["phases"]["inherited"]
        consumer.require(inherited["scope"] == "unknown" and inherited["receipt_sha256"] is None and
                         all(value is None for value in inherited["measurements"].values()),
                         "inherited phase is not empty")
        if item["arm"] == "dynamics_compiled_witness":
            root = slot["root"]
            row = rows[root]
            consumer.require(item["task_sha256"] == row["task_sha256"] and
                             item["target_sha256"] == row["target_sha256"], "target identity mismatch")
            for source_axis, destination_axis in MAP.items():
                inherited["measurements"][destination_axis] = copy.deepcopy(row["event_axes"][source_axis])
            slot["partial_inherited_source"] = {
                "scope": "compiled_beam_search_and_replay_only",
                "journal_sha256": ancestor.EVENT_SHA,
                "saved_audit_sha256": SAVED_SHA,
                "selection_sha256": row["selection_sha256"],
                "predictor_module_calls": copy.deepcopy(row["event_axes"]["predictor"]),
                "shared_costs": "unknown",
                "cross_stage_overlap": "unknown",
            }
            matched.add(root)
        slot["ledger_preflight"] = ledger.inspect(item)
        consumer.require(slot["ledger_preflight"]["structural_status"] == "incomplete" and
                         "inherited:receipt_missing" in slot["ledger_preflight"]["blockers"] and
                         "inherited:not_attributed_per_arm" in slot["ledger_preflight"]["blockers"] and
                         "proposal:receipt_missing" in slot["ledger_preflight"]["blockers"],
                         "missing cost not blocked")
    consumer.require(matched == {0, 1}, "compiled root coverage mismatch")
    result["schema"] = "codex.e1.ancestor_integrated_ledger.v1"
    result["sources"]["compiled_ancestor_saved_audit_sha256"] = SAVED_SHA
    result["sources"]["compiled_ancestor_journal_sha256"] = ancestor.EVENT_SHA
    result["unresolved"] += ["compiled_ancestor_shared_and_earlier_costs",
                             "cross_stage_event_overlap_and_allocation",
                             "inherited_phase_receipt_and_complete_coverage"]
    result["boundary"] = ("partial compiled ancestor event observations in inherited phase; "
                          "scope unknown, no totals, no parity, no P1/GPU/physics")
    result["equal_budget_certified"] = result["ranking_authorized"] = result["scientific_admission"] = False
    return result


def load_verified():
    a = audit
    targets = a.parse(a.read_bound(consumer.TARGETS, consumer.TARGETS_SHA, 4 * 1024 * 1024))
    prepare = a.parse(a.read_bound(consumer.PREPARE, consumer.PREPARE_SHA, 4 * 1024 * 1024))
    report = a.parse(a.read_bound(a.REPORT, a.REPORT_SHA, 4 * 1024 * 1024))
    receipt = a.parse(a.read_bound(a.RECEIPT, a.RECEIPT_SHA, 4 * 1024 * 1024))
    records, terminal = a.decode(a.read_bound(a.EVENTS, a.EVENT_SHA, 16 * 1024 * 1024))
    base = consumer.consume(targets, prepare, report, a.audit(records, terminal, report, receipt))
    saved = a.parse(a.read_bound(SAVED, SAVED_SHA, 4 * 1024 * 1024))
    recomputed = ancestor.analyze(*ancestor.load())
    consumer.require(saved == recomputed, "saved ancestor audit differs from recomputation")
    return base, recomputed


def main():
    try:
        consumer.require(len(sys.argv) == 1, "no arguments")
        data = json.dumps(join(*load_verified()), sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
        with OUTPUT.open("xb") as stream:
            stream.write(data)
        print(json.dumps({"path": str(OUTPUT), "sha256": audit.sha(data), "status": "blocked_incomplete"}))
    except (ValueError, KeyError, TypeError, OSError, zlib.error) as exc:
        print(json.dumps({"status": "blocked_invalid", "reason": str(exc),
                          "equal_budget_certified": False, "ranking_authorized": False}), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
