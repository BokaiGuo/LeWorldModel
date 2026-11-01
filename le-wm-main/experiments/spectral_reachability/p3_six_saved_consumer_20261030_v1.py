"""Strict callback-record checks, deliberately not a native HOST certifier."""
from collections import Counter

from experiments.spectral_reachability import p3_feedback_reverse_control_20261030_v1 as control
from experiments.spectral_reachability import p3_six_numeric_bridge_20261030_v1 as bridge

need = control.need


def audit_slot(result, events):
    """Verify durable callback records against the result; never infer native IO."""
    need(type(events) is list and type(result) is dict, "record types")
    finals = [e["result"] for e in events if e.get("kind") == "slot_final"]
    need(len(finals) == 1 and events[-1].get("kind") == "slot_final" and
         finals[0] == result, "unique terminal record")
    need(result["schema"] == control.NAME + ".slot" and result["policy"] in control.POLICIES and
         result["source_authenticated"] is False and result["native_execution_certified"] is False and
         result["scientific_pass"] is False, "callback identity and boundary")
    receipt = result["resource_receipt"]
    control.core.audit_ledger(receipt)
    need(receipt["caps"] == control.design.expected(), "fixed 27-axis cap")
    attempted, completed, pending = Counter(), Counter(), {}
    for event in events:
        kind = event.get("kind")
        if kind == "callback_attempt":
            name, ordinal = event["callback"], event["ordinal"]
            need(name in ("packet", "encode") and not pending and type(ordinal) is int and
                 ordinal == attempted[name] + 1 and ordinal <= 4, "callback attempt")
            attempted[name] += 1
            pending[name] = ordinal
        elif kind in ("callback_return", "callback_failure"):
            name, ordinal = event["callback"], event["ordinal"]
            need(pending.pop(name, None) == ordinal, "callback disposition")
            if kind == "callback_return":
                completed[name] += 1
    counts = result["callback_receipts"]
    need(not pending and counts["attempted"] == dict(attempted) and
         counts["completed"] == dict(completed) and
         all(counts[k] is None for k in ("physical_steps", "encoder_kernels", "FLOPs")),
         "callback accounting and unknowns")
    # Match every ledger transition to its durable event, including nested calls.
    for entry in receipt["events"]:
        matches = [e for e in events if e.get("kind") in
                   ("resource_attempt", "resource_return", "resource_failure", "resource_blocked")
                   and e["entry"]["index"] == entry["index"]]
        if entry["status"] == "blocked_before_dispatch":
            need(len(matches) == 1 and matches[0]["kind"] == "resource_blocked" and
                 matches[0]["entry"] == entry, "blocked accounting")
        else:
            need(len(matches) == 2 and matches[0]["kind"] == "resource_attempt" and
                 matches[1]["kind"] == ("resource_return" if entry["status"] == "returned" else "resource_failure")
                 and matches[1]["entry"] == entry, "resource accounting")
            initial = dict(entry, status="attempted", ended_ns=None, error=None)
            need(matches[0]["entry"] == initial, "resource attempt identity")
    need(sum(e.get("kind") in ("resource_attempt", "resource_return", "resource_failure", "resource_blocked")
             for e in events) == sum(1 if x["status"] == "blocked_before_dispatch" else 2
                                     for x in receipt["events"]), "no extra resource transitions")
    need(result["status"] in ("failed", "native_stopped", "callback_feedback_complete_not_certified"),
         "known callback status")
    if result["status"] == "callback_feedback_complete_not_certified":
        need(not result["errors"] and attempted == completed == Counter(packet=4, encode=4) and
             receipt["attempted"] == receipt["completed"] == control.design.expected(),
             "complete accounting")
        blocks = result["blocks"]
        need(len(blocks) == 2, "two blocks")
        remaining = [1, 3]
        for stage, block in enumerate(blocks):
            h = control.choose(result["policy"], block["decision"], remaining, stage)
            need(block["stage"] == stage and block["horizon"] == h, "policy and gate")
            remaining.remove(h)
        need([p["start"] for p in result["packets"]] == [0, 5, 10, 15] and
             [p["end"] for p in result["packets"]] == [5, 10, 15, 20] and
             all(p["stopped"] is False for p in result["packets"]), "complete packet boundaries")
    return dict(status=result["status"], attempted=dict(attempted), completed=dict(completed),
                resource_attempted=receipt["attempted"], resource_completed=receipt["completed"],
                physical_steps=None, full_saved_strict=False, scientific_pass=False)


def consume(rows, slot_events):
    """Fail closed: only check provided callback records; missing slots stay missing."""
    need(type(rows) is list and len(rows) == 6 and type(slot_events) is dict, "six-slot denominator")
    need(all((row["slot"], row["root"], row["policy"]) == expected
             for row, expected in zip(rows, control.ROSTER, strict=True)), "fixed roster")
    need(all(type(k) is int and 0 <= k < 6 for k in slot_events), "slot event keys")
    audits = []
    for row in rows:
        result = row.get("result")
        events = slot_events.get(row["slot"])
        if result is None:
            need(events is None and row["status"] in
                 ("not_started", "setup_failed", "engineering_failed", "native_initial_stop"),
                 "missing slot preserved")
            audits.append(None)
        else:
            need(events is not None, "result without saved records")
            need(result["policy"] == row["policy"] and
                 (row["status"] == result["status"] or row["status"] == "cleanup_failed"),
                 "result row binding")
            audits.append(audit_slot(result, events))
    return dict(status="callback_records_only_not_host_certified", slots=audits, rows=rows,
                fixed_denominator=6, source_authenticated=False, physical_certified=False,
                full_saved_strict=False, scientific_pass=False)


def consume_transport(events):
    """Check a fully returned six-slot transport; caller must verify its journal."""
    need(type(events) is list and events, "transport events")
    markers = [e for e in events if e["origin"] == "physical" and e["resource"] in
               ("six_denominator", "six_slot_attempt", "six_slot_return")]
    need(len(markers) == 13 and markers[0]["resource"] == "six_denominator" and
         markers[0]["detail"]["fixed_denominator"] == 6, "fixed lifecycle markers")
    initial = markers[0]["detail"]["rows"]
    need(len(initial) == 6 and all((r["slot"], r["root"], r["policy"]) == ident and
         r["status"] == "not_started" and r["result_sha256"] is None
         for r, ident in zip(initial, control.ROSTER, strict=True)), "initial denominator")
    rows, slot_events = [], {}
    positions = {id(e): i for i, e in enumerate(events)}
    for i, ident in enumerate(control.ROSTER):
        attempt, returned = markers[1 + 2*i:3 + 2*i]
        slot, root, policy = ident
        need(attempt["resource"] == "six_slot_attempt" and
             attempt["detail"] == dict(slot=slot, root=root, policy=policy) and
             returned["resource"] == "six_slot_return", "fixed slot lifecycle")
        row = dict(returned["detail"])
        need((row["slot"], row["root"], row["policy"]) == ident, "returned identity")
        selected = [e for e in events if e["origin"] == "numerical" and e["resource"] == bridge.RESOURCE
                    and e["detail"].get("slot") == slot]
        need(all(positions[id(attempt)] < positions[id(e)] < positions[id(returned)] for e in selected),
             "slot transport boundaries")
        wrapped = bridge.reconstruct(events, slot)
        need(all(x["kind"] == "arm_event" and x["slot"] == slot for x in wrapped), "arm wrapper")
        payloads = [x["payload"] for x in wrapped]
        finals = [x["result"] for x in payloads if x.get("kind") == "slot_final"]
        if row["result_sha256"] is None:
            need(not finals and row["status"] in ("engineering_failed", "native_initial_stop"),
                 "missing result terminal")
            # Partial records without a final receipt cannot be reconciled; do not discard them.
            need(not payloads, "unsettled callback records require partial journal handling")
            row["result"] = None
        else:
            need(len(finals) == 1 and control.core.digest(finals[0]) == row["result_sha256"],
                 "terminal result hash")
            row["result"] = finals[0]
            if row["cleanup"] == "failed":
                row["status"] = "cleanup_failed"
            slot_events[slot] = payloads
        rows.append(row)
    report = consume(rows, slot_events)
    report["transport_scope"] = "complete markers only; external journal integrity required"
    return report
