"""Saved-only validation of six-slot decision journal; no GPU or physical certification."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path

from experiments.spectral_reachability import p3_decision_journal_20261030_codex_v1 as journal

control = journal.control
need = control.old.need
NAME = "p3_decision_saved_consumer_20261030_codex_v1"
RESOURCE = {"resource_attempt", "resource_return", "resource_failure", "resource_blocked"}
AUXILIARY = {"AD_terminal", "boundary", "direction", "io_input", "io_return", "jacobian",
             "nominal", "origin", "packet", "program_boundary", "program_packet", "raw_backward"}


def decode(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            need(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def read_records(path, expected_sha, max_bytes):
    path = Path(path)
    need(type(max_bytes) is int and max_bytes > 0 and path.is_file() and not path.is_symlink()
         and path.stat().st_size <= max_bytes, "bounded regular journal")
    raw = path.read_bytes()
    need(journal.sha(raw) == expected_sha and raw.endswith(b"\n"), "journal SHA or truncated")
    prev, records = None, []
    for index, line in enumerate(raw.splitlines()):
        need(0 < len(line) + 1 <= journal.MAX_RECORD, "record size")
        entry = decode(line)
        need(type(entry) is dict and set(entry) == {"index", "previous", "payload", "sha256"}
             and type(entry["index"]) is int and entry["index"] == index and entry["previous"] == prev
             and entry["sha256"] == control.core.digest({k: v for k, v in entry.items() if k != "sha256"})
             and control.core.canonical(entry) == line and type(entry["payload"]) is dict, "journal chain")
        records.append(entry["payload"])
        prev = entry["sha256"]
    return records


def audit_receipt(receipt, stage, resource_events, complete):
    control.core.audit_ledger(receipt)
    need(receipt["caps"] == control.expected() and all(e["scope"].get("stage") == stage
         for e in receipt["events"]), "per-decision caps/scope")
    if complete:
        need(receipt["attempted"] == receipt["completed"] == control.expected(), "complete decision budget")
    actual = [e for e in resource_events if e["entry"]["scope"].get("stage") == stage]
    expected = []
    for entry in receipt["events"]:
        if entry["status"] == "blocked_before_dispatch":
            expected.append(("resource_blocked", entry))
        else:
            expected.append(("resource_attempt", dict(entry, status="attempted", ended_ns=None, error=None)))
            expected.append(("resource_return" if entry["status"] == "returned" else "resource_failure", entry))
    # Nested operations mean return order need not match attempt order.
    canon = lambda item: control.core.canonical([item[0], item[1]])
    need(Counter(canon((e["kind"], e["entry"])) for e in actual) ==
         Counter(canon(x) for x in expected), "resource event census")
    for entry in receipt["events"]:
        seq = [e["kind"] for e in actual if e["entry"]["index"] == entry["index"]]
        need(seq == (["resource_blocked"] if entry["status"] == "blocked_before_dispatch" else
                     ["resource_attempt", "resource_return" if entry["status"] == "returned"
                      else "resource_failure"]), "resource transition order")
    stack = []
    for event in actual:
        kind, index = event["kind"], event["entry"]["index"]
        if kind == "resource_attempt":
            stack.append(index)
        elif kind != "resource_blocked":
            need(stack and stack.pop() == index, "resource nesting")
    need(not stack, "unsettled resource transitions")
    return dict(attempted=receipt["attempted"], completed=receipt["completed"], complete=complete)


def audit_slot(row, events):
    result = row["result"]
    need(type(result) is dict and result.get("schema") == control.NAME + ".slot" and
         result.get("contract") == control.contract() and result.get("policy") == row["policy"]
         and result.get("physical_certified") is False and result.get("scientific_pass") is False,
         "slot result identity")
    need(events and events[-1].get("kind") == "slot_final" and events[-1].get("result") == result
         and sum(e.get("kind") == "slot_final" for e in events) == 1, "slot terminal")
    status = result["status"]
    need(status in ("failed", "native_stopped", "callback_complete_unverified") and
         (row["status"] == status or row["status"] == "cleanup_failed"), "slot status")
    decisions, packets = result["decisions"], result["packets"]
    need(type(decisions) is list and type(packets) is list and 0 <= len(packets) <= len(decisions) <= 4,
         "decision/packet cardinality")
    resource_events = [e for e in events if e.get("kind") in RESOURCE]
    receipts = [b["resource_receipt"] for b in decisions]
    if "failed_decision_receipt" in result:
        need(status == "failed" and len(receipts) < 4, "failed decision identity")
        receipts.append(result["failed_decision_receipt"])
    need(all(e["entry"]["scope"].get("stage") in range(len(receipts)) for e in resource_events),
         "unaccounted resource events")
    audited = [audit_receipt(receipt, i, resource_events, i < len(decisions))
               for i, receipt in enumerate(receipts)]
    paid = [e for e in events if e.get("kind") == "decision_paid"]
    need(len(paid) == len(decisions), "paid decision records")
    for stage, (block, event) in enumerate(zip(decisions, paid, strict=True)):
        need(block["stage"] == stage and event == dict(kind="decision_paid", stage=stage,
             context_sha256=block["context_sha256"], selected_horizon=block["selected_horizon"],
             resource_receipt=block["resource_receipt"]) and
             block["selected_horizon"] == control.choose(row["policy"], block["decision"]) and
             [p["horizon"] for p in block["panels"]] == [1, 3], "paid decision identity")
    kinds = {"decision_paid", "packet_attempt", "packet_return", "encode_attempt", "encode_return"}
    phases = [e for e in events if e.get("kind") in kinds]
    need(all(e.get("kind") in RESOURCE | AUXILIARY | kinds | {"slot_final"} for e in events), "unknown slot event")
    state, stage = "decision_paid", 0
    for phase_index, event in enumerate(phases):
        kind = event["kind"]
        need(event.get("stage") == stage and kind == state, "decision/packet/encode phase order")
        if kind == "decision_paid":
            state = "packet_attempt"
        elif kind == "packet_attempt":
            need(event["start"] == stage * 5 and event["raw"] ==
                 decisions[stage]["panels"][0 if decisions[stage]["selected_horizon"] == 1 else 1]["winner_actions"][:5],
                 "dispatched winner prefix")
            state = "packet_return"
        elif kind == "packet_return":
            need(stage < len(packets) and event["end"] == packets[stage]["end"] and
                 event["stopped"] == packets[stage]["stopped"] and packets[stage]["start"] == stage * 5
                 and packets[stage]["raw"] == decisions[stage]["panels"][
                     0 if decisions[stage]["selected_horizon"] == 1 else 1]["winner_actions"][:len(packets[stage]["raw"])]
                 and event["end"] == stage * 5 + len(packets[stage]["raw"])
                 and 1 <= len(packets[stage]["raw"]) <= 5 and
                 (len(packets[stage]["raw"]) == 5 or event["stopped"] is True), "packet return")
            state = "stop" if event["stopped"] else "encode_attempt"
        elif kind == "encode_attempt":
            need(event["end"] == stage * 5 + 5, "encode boundary")
            state = "encode_return"
        else:
            prior = phases[phase_index - 1]
            need(event["end"] == prior["end"] and event["rgb_sha256"] == prior["rgb_sha256"], "encoder binding")
            stage += 1
            state = "decision_paid"
    positions = {id(e): pos for pos, e in enumerate(events)}
    for event in resource_events:
        s = event["entry"]["scope"]["stage"]
        earlier = [p for p in phases if p.get("kind") == "encode_return" and p.get("stage") == s - 1]
        later = [p for p in paid if p["stage"] == s]
        need((s == 0 or len(earlier) == 1 and positions[id(earlier[0])] < positions[id(event)]) and
             (not later or positions[id(event)] < positions[id(later[0])]), "resource phase order")
    auxiliary = [e for e in events if e.get("kind") in AUXILIARY]
    for event in auxiliary:
        scope = event.get("scope")
        need(type(scope) is dict and type(scope.get("stage")) is int and
             scope["stage"] in range(len(receipts)), "auxiliary stage")
        s = scope["stage"]
        earlier = [p for p in phases if p.get("kind") == "encode_return" and p.get("stage") == s - 1]
        later = [p for p in paid if p["stage"] == s]
        need((s == 0 or len(earlier) == 1 and positions[id(earlier[0])] < positions[id(event)]) and
             (not later or positions[id(event)] < positions[id(later[0])]), "auxiliary phase order")
    attempted, completed = result["callback_attempted"], result["callback_completed"]
    need(set(attempted) == set(completed) == {"packet", "encode"} and
         all(type(attempted[k]) is int and type(completed[k]) is int and
             0 <= completed[k] <= attempted[k] <= 4 for k in attempted), "callback counts")
    for name in ("packet", "encode"):
        need(attempted[name] == sum(e.get("kind") == name + "_attempt" for e in events)
             and sum(e.get("kind") == name + "_return" for e in events) <= completed[name]
             <= attempted[name], "callback event census")
    if status == "callback_complete_unverified":
        need(len(decisions) == len(packets) == 4 and stage == 4 and state == "decision_paid" and
             attempted == completed == {"packet": 4, "encode": 4} and result["error"] is None,
             "complete slot")
    elif status == "native_stopped":
        need(state == "stop" and packets[-1]["stopped"] is True and result["error"] is None, "stop evidence")
    else:
        need(type(result["error"]) is str and bool(result["error"]) and state != "stop", "failure evidence")
    return dict(status=status, decision_accounts=audited, packet_count=len(packets),
                callback_attempted=attempted, callback_completed=completed,
                auxiliary_event_counts=dict(Counter(e["kind"] for e in auxiliary)),
                auxiliary_content_certified=False)


def consume(path, expected_sha, max_bytes):
    events = read_records(path, expected_sha, max_bytes)
    need(len(events) >= 2 and events[0].get("kind") == "journal_begin" and
         events[-1].get("kind") == "journal_end", "missing journal terminal")
    pins = journal.source_pins()
    need(events[0]["pins"] == events[-1]["pins"] == pins and events[0]["contract"] == control.contract(),
         "source binding")
    roster = events[-1]["roster"]
    need(roster.get("schema") == control.NAME + ".roster" and roster.get("contract") == control.contract()
         and roster.get("fixed_denominator") == 6 and roster.get("scientific_pass") is False
         and roster.get("source_authenticated") is False and roster.get("physical_certified") is False,
         "roster contract")
    rows = roster["rows"]
    need(type(rows) is list and len(rows) == 6, "fixed denominator")
    body = events[1:-1]
    need(body and body[0].get("kind") == "fixed_denominator", "initial denominator")
    initial = body[0]["rows"]
    need(len(initial) == 6 and all((r["slot"], r["root"], r["policy"]) == ident and
         r["status"] == "not_started" and r["result"] is None and r["cleanup"] == "not_required"
         for r, ident in zip(initial, control.ROSTER, strict=True)), "initial roster")
    i, accounts = 1, []
    for row, ident in zip(rows, control.ROSTER, strict=True):
        slot, root, policy = ident
        need((row["slot"], row["root"], row["policy"]) == ident and i < len(body) and
             body[i] == dict(kind="slot_attempt", slot=slot, root=root, policy=policy), "slot attempt")
        i += 1
        start = i
        while i < len(body) and body[i].get("kind") != "slot_return":
            need(body[i].get("kind") not in ("slot_attempt", "fixed_denominator", "journal_begin", "journal_end"),
                 "unexpected lifecycle event")
            i += 1
        need(i < len(body) and body[i] == dict(kind="slot_return", row=row), "slot return")
        detail = body[start:i]
        if row["result"] is None:
            need(not detail and row["status"] in ("setup_failed", "cleanup_failed"), "unsettled slot")
            accounts.append(None)
        else:
            accounts.append(audit_slot(row, detail))
        need(row["cleanup"] in ("returned", "failed", "not_required") and
             (row["cleanup"] != "failed" or row["status"] == "cleanup_failed"), "cleanup status")
        i += 1
    need(i == len(body), "unexpected trailing events")
    for root in range(2):
        identities = {r["result"]["root_sha256"] for r in rows if r["root"] == root and r["result"] is not None}
        need(len(identities) <= 1, "paired root digest")
    return dict(schema=NAME + ".receipt", journal_sha256=expected_sha, fixed_denominator=6,
                slot_status=[r["status"] for r in rows], slot_accounts=accounts,
                status="callback_journal_consistent_not_native_certified", physical_certified=False,
                source_authenticated=False, full_saved_strict=False, scientific_pass=False,
                limitations="untrusted callback and self-reported resource hooks; no GPU or physics proof")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--journal", required=True)
    p.add_argument("--journal-sha", required=True)
    p.add_argument("--max-bytes", required=True, type=int)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    result = consume(a.journal, a.journal_sha, a.max_bytes)
    out = Path(a.output)
    need(out.is_absolute() and out.resolve() == out and not out.exists() and not out.is_symlink(), "exclusive output")
    raw = control.core.canonical(result) + b"\n"
    with out.open("xb") as stream:
        need(stream.write(raw) == len(raw), "receipt write")
        stream.flush()
        os.fsync(stream.fileno())
    journal.syncdir(out.parent)
    print(journal.sha(raw))


if __name__ == "__main__":
    main()
