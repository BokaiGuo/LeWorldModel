"""Consume pinned E1 hook journal into incomplete per-slot ledgers; no P1 access."""
import base64
import binascii
import json
from pathlib import Path
import sys
import zlib

from . import codex_e1_end_to_end_ledger_interface_20261030_v1 as ledger
from . import codex_e1_sealed_event_axis_audit_20261030_v1 as audit

BASE = Path(__file__).resolve().parents[2] / "results"
TARGETS = BASE / "agent_e1_six_slot_common_prepare_20261029_v1_prepare_MAIN01.json"
PREPARE = BASE / "agent_e1_six_slot_common_cuda_20261029_v2_prepare_MAIN01.json"
TARGETS_SHA = "cd96f6e28b25d3b7e3b6649c760b2f111747aec5c0ce9c8a41741318209df8a0"
PREPARE_SHA = "df548a6a9d8356f654e6aac823e7dd90383570da56631c38155b539bad8ae0dc"
AXES = ("predict_api_calls", "predictor_samples", "action_encoder_calls", "pred_proj_calls")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def consume(targets, prepare, report, event_report):
    """Join checked structures; public entry point recomputes event_report from fixed bytes."""
    require(targets["schema"] == "agent_e1_six_slot_common_prepare_20261029_v1.prepare" and
            prepare["schema"] == "agent_e1_six_slot_common_cuda_20261029_v2.prepare" and
            prepare["six_sha256"] == TARGETS_SHA and prepare["targets_sha256"] == targets["target_seal_sha256"]
            and report["prepare_sha256"] == PREPARE_SHA, "prepare chain mismatch")
    require(event_report["schema"] == "codex.e1.sealed_event_axis_audit.v1" and
            event_report["journal_sha256"] == audit.EVENT_SHA and
            event_report["report_sha256"] == audit.REPORT_SHA and
            event_report["receipt_sha256"] == audit.RECEIPT_SHA and
            event_report["status"] == "partial_resource_chain_only", "event audit binding mismatch")
    rows, groups = targets["rows"], event_report["groups"]
    require(type(rows) is list and len(rows) == 6 and type(groups) is list and len(groups) == 12,
            "six-slot denominator")
    by_key = {}
    for group in groups:
        key = (group["slot"], group["phase"])
        require(type(key[0]) is int and key[0] in range(6) and key[1] in ("controller", "replay")
                and key not in by_key, "duplicate or invalid event group")
        by_key[key] = group
    slots = []
    for i, row in enumerate(rows):
        root, arm = i // 3, audit.ARMS[i % 3]
        require(type(row["root_slot"]) is int and row["root_slot"] == root and row["arm"] == arm,
                "target slot mismatch")
        raw = base64.b64decode(row["target_f32_le_base64"], validate=True)
        require(len(raw) == 768 and audit.sha(raw) == row["target_data_sha256"], "target bytes mismatch")
        phases = {p: {"scope": "unknown", "receipt_sha256": None,
                      "measurements": {axis: None for axis in ledger.AXES}} for p in ledger.PHASES}
        for phase in ("controller", "replay"):
            group = by_key[(i, phase)]
            require(type(group["root"]) is int and group["root"] == root and group["arm"] == arm,
                    "event group identity mismatch")
            require(set(group["measurements"]) == set(AXES) | {"predictor_module_calls"},
                    "event measurement axes")
            phases[phase]["scope"] = "per_arm"
            phases[phase]["receipt_sha256"] = audit.RECEIPT_SHA
            for axis in AXES:
                phases[phase]["measurements"][axis] = group["measurements"][axis].copy()
        item = {"schema": ledger.SCHEMA, "scope_sha256": PREPARE_SHA, "arm": arm,
                "task_sha256": row["task_sha256"], "target_sha256": row["target_data_sha256"],
                "phases": phases}
        checked = ledger.inspect(item)
        require(checked["structural_status"] == "incomplete" and
                "proposal:receipt_missing" in checked["blockers"] and
                "inherited:receipt_missing" in checked["blockers"], "missing budget not blocked")
        slots.append({"slot": i, "root": root, "ledger_input": item, "ledger_preflight": checked})
    return {"schema": "codex.e1.event_ledger_consumer.v1", "status": "blocked_incomplete",
            "sources": {"targets_sha256": TARGETS_SHA, "prepare_sha256": PREPARE_SHA,
                        "report_sha256": audit.REPORT_SHA, "receipt_sha256": audit.RECEIPT_SHA,
                        "journal_sha256": audit.EVENT_SHA},
            "slots": slots,
            "unresolved": ["proposal_costs_and_attempts", "historical_inherited_and_shared_costs",
                           "all_other_unmeasured_axes", "event_coverage_outside_saved_journal",
                           "distinct_axes_overlap_and_flops", "end_to_end_fairness"],
            "equal_budget_certified": False, "ranking_authorized": False, "scientific_admission": False,
            "boundary": "pinned saved E1 records; no P1 files, fullstrict, model, GPU, or physics"}


def main():
    try:
        targets = audit.parse(audit.read_bound(TARGETS, TARGETS_SHA, 4 * 1024 * 1024))
        prepare = audit.parse(audit.read_bound(PREPARE, PREPARE_SHA, 4 * 1024 * 1024))
        report = audit.parse(audit.read_bound(audit.REPORT, audit.REPORT_SHA, 4 * 1024 * 1024))
        receipt = audit.parse(audit.read_bound(audit.RECEIPT, audit.RECEIPT_SHA, 4 * 1024 * 1024))
        records, terminal = audit.decode(audit.read_bound(audit.EVENTS, audit.EVENT_SHA, 16 * 1024 * 1024))
        output = consume(targets, prepare, report, audit.audit(records, terminal, report, receipt))
    except (ValueError, KeyError, TypeError, OSError, zlib.error, binascii.Error) as exc:
        print(json.dumps({"status": "blocked_invalid", "reason": str(exc),
                          "ranking_authorized": False, "equal_budget_certified": False}), file=sys.stderr)
        return 2
    print(json.dumps(output, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
