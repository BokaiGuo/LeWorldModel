"""Saved-only P3 dual-branch audit, for a stopped user service and terminal run."""
import argparse
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np

from experiments.spectral_reachability import p3_decision_dual_branch_20261030_codex_v1 as dual
from experiments.spectral_reachability import p3_decision_journal_20261030_codex_v1 as journal
from experiments.spectral_reachability import p3_decision_saved_consumer_20261030_codex_v1 as old_consumer
from experiments.spectral_reachability import p3_six_durable_transport_20261030_v1 as transport
from experiments.spectral_reachability import p3_six_native_saved_20261030_v1 as native_saved

NAME = "p3_dual_branch_saved_consumer_20261030_codex_v1"
DRIVER = "p3_dual_branch_host_driver_20261030_codex_v1"
ADAPTER = "p3_decision_native_cuda_adapter_20261030_codex_v1"
core, need = dual.core, dual.need
RESOURCE = old_consumer.RESOURCE


def service_stopped(unit, runner=subprocess.run, cgroup_root=Path("/sys/fs/cgroup")):
    need(type(unit) is str and re.fullmatch(r"[A-Za-z0-9_.:@-]+\.service", unit),
         "explicit user service name")
    p = runner(["systemctl", "--user", "show", unit, "--property=LoadState",
                "--property=ActiveState", "--property=SubState", "--property=MainPID",
                "--property=ControlGroup", "--property=Result"], capture_output=True,
               text=True, check=True, timeout=10)
    fields = {}
    for line in p.stdout.splitlines():
        key, sep, value = line.partition("=")
        need(sep and key not in fields, "systemd fields")
        fields[key] = value
    need(set(fields) == {"LoadState", "ActiveState", "SubState", "MainPID", "ControlGroup", "Result"}
         and fields["LoadState"] == "loaded" and fields["ActiveState"] in ("inactive", "failed")
         and fields["MainPID"] == "0" and fields["ControlGroup"].startswith("/"), "service not conclusively stopped")
    group = cgroup_root / fields["ControlGroup"].lstrip("/")
    need(group.is_dir() and not group.is_symlink(), "service cgroup unavailable")
    proc_files = list(group.rglob("cgroup.procs"))
    need(proc_files and all(not p.read_text().strip() for p in proc_files),
         "service cgroup still contains processes")
    return fields


def file_sha(path):
    p = Path(path)
    need(p.is_file() and not p.is_symlink(), "regular saved file")
    digest = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pinned_json(path, expected, cap):
    p = Path(path)
    need(p.is_file() and not p.is_symlink() and p.stat().st_size <= cap, "bounded saved JSON")
    data = p.read_bytes()
    need(hashlib.sha256(data).hexdigest() == expected, "external terminal SHA")
    obj = old_consumer.decode(data)
    need(type(obj) is dict, "saved JSON object")
    return obj


def read_journal(path, expected, max_bytes):
    """Stream bytes and chain validation; payloads are kept for semantic checks."""
    p = Path(path)
    need(type(max_bytes) is int and max_bytes > 0 and p.is_file() and not p.is_symlink()
         and p.stat().st_size <= max_bytes, "bounded decision journal")
    digest, previous, result = hashlib.sha256(), None, []
    with p.open("rb") as f:
        for index, line in enumerate(f):
            need(len(line) <= journal.MAX_RECORD and line.endswith(b"\n"), "bounded complete journal line")
            digest.update(line)
            entry = old_consumer.decode(line[:-1])
            need(type(entry) is dict and set(entry) == {"index", "previous", "payload", "sha256"} and
                 type(entry["index"]) is int and entry["index"] == index and entry["previous"] == previous and
                 entry["sha256"] == core.digest({k: v for k, v in entry.items() if k != "sha256"}) and
                 core.canonical(entry) == line[:-1] and type(entry["payload"]) is dict, "journal chain")
            previous = entry["sha256"]
            result.append(entry)
    need(digest.hexdigest() == expected and result, "journal SHA")
    return result


def audit_receipt(receipt, stage, events, branch, complete):
    core.audit_ledger(receipt)
    caps = dual.admission_cost() if branch is None else dual.decision_cost(branch)
    need(receipt["caps"] == caps and all(e["scope"].get("stage") == stage
         for e in receipt["events"]), "decision budget or scope")
    if complete:
        need(receipt["attempted"] == receipt["completed"] == caps, "complete branch budget")
    actual = [e for e in events if e.get("kind") in RESOURCE]
    expected = []
    for e in receipt["events"]:
        if e["status"] == "blocked_before_dispatch":
            expected.append(("resource_blocked", e))
        else:
            expected.extend([("resource_attempt", dict(e, status="attempted", ended_ns=None, error=None)),
                             ("resource_return" if e["status"] == "returned" else "resource_failure", e)])
    canon = lambda x: core.canonical(x)
    need(Counter(canon([e["kind"], e["entry"]]) for e in actual) ==
         Counter(canon(list(e)) for e in expected), "resource event census")
    stack = []
    for e in actual:
        kind, index = e["kind"], e["entry"]["index"]
        if kind == "resource_attempt":
            stack.append(index)
        elif kind != "resource_blocked":
            need(stack and stack.pop() == index, "resource nesting")
    need(not stack, "unsettled resource event")


def audit_decision(block, events, stage, policy, previous_context):
    need(block.get("schema") == dual.NAME + ".decision" and block.get("contract") == dual.contract()
         and block.get("stage") == stage and block.get("policy") == policy and
         block.get("physical_certified") is False and block.get("scientific_pass") is False,
         "decision identity")
    need(previous_context is None or block["context_sha256"] == previous_context, "feedback context chain")
    original = [e for e in events if e.get("kind") == "original_decision"]
    maps = [e["mapping"] for e in events if e.get("kind") == dual.r.parent.NAME + ".signal_map"]
    branch_events = [e for e in events if e.get("kind") == "branch_budget"]
    decision, branch = block["decision"], block["branch"]
    task = None
    if decision is not None:
        need(len(original) == 1 and original[0] == dict(kind="original_decision", stage=stage,
             decision=decision, diagnostic=block["diagnostic"], signal_receipt=block["signal_receipt"])
             and len(maps) == 2, "original decision evidence")
        for mapping in maps:
            need(mapping["mapping_sha256"] == core.digest({k: v for k, v in mapping.items()
                                                           if k != "mapping_sha256"}), "signal map hash")
        task = maps[0]["parent_task"]
        responses = [m["native_response"] for m in maps]
        need([r["horizon"] for r in responses] == [1, 3] and
             all(m["parent_task"] == task for m in maps) and
             core.digest(task["root_context"]) == block["context_sha256"] and
             responses == block["signal_receipt"]["responses"], "saved signal/task binding")
        node, signals, receipt = dual.prior.design.signal.runtime_signal(task, responses, block["diagnostic"])
        need(receipt == block["signal_receipt"] and core.choose(task, node, signals, "adaptive") == decision,
             "original scheduler recomputation")
    else:
        need(not original and branch is None, "decision absent")
    if branch is not None:
        dual.validate_decision(decision)
        need(branch == decision["extension_type"] and
             branch_events == [dict(kind="branch_budget", branch=branch, caps=dual.decision_cost(branch))],
             "actual branch binding")
    else:
        need(not branch_events, "branch without budget")
    complete = block["status"] == "dual_panels_complete_not_certified"
    need(complete or block["status"] == "failed", "decision status")
    audit_receipt(block["resource_receipt"], stage, events, branch, complete)
    if complete:
        need(branch is not None and block["error"] is None and len(block["panels"]) == 2 and
             block["selected_horizon"] == dual.choose(policy, decision), "completed decision")
        winners = {}
        for h, panel in zip((1, 3), block["panels"], strict=True):
            rows = panel["rows"]
            need(panel["status"] == "witness_found" and
                 len(rows) == (4 if branch == "progress" else 8) and
                 [row["index"] for row in rows] == list(range(len(rows))) and
                 all(row["status"] == "complete" for row in rows), "completed panel")
            if branch == "progress":
                need(panel["horizon"] == h and all(row["eligible"] is True for row in rows),
                     "progress panel type")
            else:
                need(panel["bank"]["horizon"] == h and panel["control"]["status"] == "complete",
                     "R panel type")
                z = panel["control"]
                need(all(row["eligible"] == (row["immediate"] > z["immediate"] and
                                              row["reach"] < z["reach"]) for row in rows), "R witness")
            eligible = [row for row in rows if row["eligible"] is True]
            need(eligible and [row["index"] for row in eligible] == panel["eligible_indices"], "eligible census")
            winner = min(eligible, key=lambda row: (row["immediate"], row["index"]))
            need(block["winner_indices"][str(h)] == winner["index"], "winner rank")
            winners[h] = winner
        need(block["selected_actions"] == winners[block["selected_horizon"]]["actions"][:5],
             "selected action prefix")
        return_kind = "progress_return" if branch == "progress" else "R_return"
        need([e["result"] for e in events if e.get("kind") == return_kind] == block["panels"],
             "actual branch group return binding")
        calls = []
        active = None
        for e in events:
            if e.get("kind") == "native_call_attempt":
                need(active is None, "nested native call")
                active = e
            elif e.get("kind") in ("native_call_return", "native_call_failure"):
                need(active is not None, "native return without attempt")
                calls.append((active, e))
                active = None
        need(active is None and len(calls) >= 4, "verification call closure")
        for (attempt, ret), (h, segmented) in zip(calls[-4:],
                                                   ((1, False), (1, True), (3, False), (3, True)), strict=True):
            request = attempt["request"]
            need(attempt["method"] == "replay" and request["segmented"] is segmented and
                 request["actions"] == winners[h]["actions"] and
                 ret["kind"] == "native_call_return" and ret["result"] == winners[h]["path"],
                 "winner replay transcript")
    else:
        need(type(block["error"]) is str and block["error"] and block["selected_actions"] is None and
             block["selected_horizon"] is None, "failed decision preserved")
    return task


def audit_slot(row, records):
    result = row["result"]
    need(type(result) is dict and result.get("schema") == dual.NAME + ".slot" and
         result.get("contract") == dual.contract() and result.get("policy") == row["policy"] and
         result.get("physical_certified") is False and result.get("scientific_pass") is False,
         "slot result identity")
    need(records and records[-1]["payload"] == dict(kind="dual_slot_final", result=result), "slot final")
    status = result["status"]
    need(status in ("failed", "native_stopped", "callback_complete_not_certified") and
         (row["status"] == status or row["status"] == "cleanup_failed"), "slot status")
    decisions, packets = result["decisions"], result["packets"]
    need(type(decisions) is list and type(packets) is list and
         0 <= len(packets) <= len(decisions) <= 4, "decision/packet count")
    chunks, current, final_blocks = [], [], []
    phases = []
    allowed = {"packet_attempt", "packet_return", "encode_attempt", "encode_return"}
    for record in records[:-1]:
        payload = record["payload"]
        if payload.get("kind") == "io":
            value = payload["payload"]
            current.append(value)
            if value.get("kind") == "dual_decision_final":
                chunks.append(current)
                final_blocks.append(value["result"])
                current = []
                phases.append(("decision", value["result"]))
        else:
            need(payload.get("kind") in allowed and not current, "unexpected slot event or unfinished decision")
            phases.append((payload["kind"], payload))
    need(not current and final_blocks == decisions, "decision terminal census")
    state, stage, context_sha, tasks = "decision", 0, None, []
    seen = {"packet_attempt": 0, "packet_return": 0, "encode_attempt": 0, "encode_return": 0}
    for phase_index, (kind, value) in enumerate(phases):
        need(kind == state, "callback order")
        if kind == "decision":
            task = audit_decision(value, chunks[stage], stage, row["policy"], context_sha)
            tasks.append(task)
            state = "packet_attempt" if value["status"] == "dual_panels_complete_not_certified" else "decision_failed"
        elif kind == "packet_attempt":
            seen[kind] += 1
            need(value["stage"] == stage and value["start"] == stage * 5 and
                 value["raw"] == decisions[stage]["selected_actions"], "packet selected action")
            state = "packet_return"
        elif kind == "packet_return":
            seen[kind] += 1
            need(stage < len(packets) and value["stage"] == stage and
                 value["end"] == packets[stage]["end"] and value["stopped"] == packets[stage]["stopped"] and
                 packets[stage]["start"] == stage * 5 and type(value["stopped"]) is bool and
                 1 <= len(packets[stage]["raw"]) <= 5 and
                 packets[stage]["raw"] == decisions[stage]["selected_actions"][:len(packets[stage]["raw"])] and
                 value["end"] == 5 * stage + len(packets[stage]["raw"]) and
                 (len(packets[stage]["raw"]) == 5 or value["stopped"]), "packet acknowledgment")
            state = "stopped" if value["stopped"] else "encode_attempt"
        elif kind == "encode_attempt":
            seen[kind] += 1
            need(value["stage"] == stage and value["end"] == 5 * (stage + 1) and
                 re.fullmatch(r"[0-9a-f]{64}", value["rgb_sha256"]), "encoding input")
            state = "encode_return"
        else:
            seen[kind] += 1
            prev = phases[phase_index - 1][1]
            need(value["stage"] == stage and value["end"] == prev["end"] and
                 value["rgb_sha256"] == prev["rgb_sha256"] and
                 re.fullmatch(r"[0-9a-f]{64}", value["context_sha256"]), "encoding return")
            context_sha = value["context_sha256"]
            stage += 1
            state = "decision"
    attempted, completed = result["callback_attempted"], result["callback_completed"]
    need(set(attempted) == set(completed) == {"packet", "encode"}, "callback count ABI")
    for name in ("packet", "encode"):
        need(type(attempted[name]) is int and type(completed[name]) is int and
             attempted[name] == seen[name + "_attempt"] and
             seen[name + "_return"] <= completed[name] <= attempted[name] <= 4, "callback census")
    if status == "callback_complete_not_certified":
        need(state == "decision" and stage == 4 and len(decisions) == len(packets) == 4 and
             attempted == completed == {"packet": 4, "encode": 4} and result["error"] is None, "completed slot")
    elif status == "native_stopped":
        need(state == "stopped" and packets[-1]["stopped"] is True and result["error"] is None, "stopped slot")
    else:
        need(state != "stopped" and type(result["error"]) is str and result["error"], "failed slot")
    return dict(status=status, decisions=len(decisions), packets=len(packets), tasks=tasks,
                phases=phases, callback_attempted=attempted, callback_completed=completed)


def audit_native_binding(slot, section, account, directory):
    native = native_saved._native_slot(section, slot)
    if account is None:
        return dict(native=native, binding="setup_or_outer_failure_unverified")
    phases, tasks = account["phases"], account["tasks"]
    attempts = [v for k, v in phases if k == "packet_attempt"]
    returns = [v for k, v in phases if k == "packet_return"]
    encoded = [v for k, v in phases if k == "encode_return"]
    dispatch = [e for e in section if e["origin"] == "physical" and e["resource"] == "native_authorized_dispatch"]
    returned = [e for e in section if e["origin"] == "physical" and e["resource"] == "native_control_return"]
    sent = [command for packet in attempts for command in packet["raw"]]
    need(len(dispatch) <= len(sent) and len(returned) <= len(dispatch), "native attempted action count")
    for i, event in enumerate(dispatch):
        d = event["detail"]
        bits = np.asarray(sent[i], dtype="<f4").tobytes().hex()
        need(d["step"] == i + 1 and d["packet_boundary"] == (i // 5) * 5 and d["raw_bits"] == bits,
             "native dispatch action bits")
    for i, event in enumerate(returned):
        need(event["detail"]["raw_bits"] == dispatch[i]["detail"]["raw_bits"], "native return action bits")
    for event in returns:
        end = event["end"]
        need(len(returned) >= end and event["stopped"] ==
             bool(returned[end - 1]["detail"]["terminated"] or returned[end - 1]["detail"]["truncated"]),
             "packet/native return binding")
    if account["status"] in ("native_stopped", "callback_complete_not_certified"):
        need(len(returned) == (returns[-1]["end"] if returns else 0) == len(dispatch),
             "no extra native actions after terminal")
    records = [e for e in section if e["origin"] == "encoding" and e["resource"] == "encoding_complete"]
    need(len(records) == len(encoded), "native encoding count")
    warm = [e for e in section if e["origin"] == "physical" and e["resource"] == "native_matched_root_accepted"]
    if tasks and encoded:
        need(len(warm) == 1 and tasks[0] is not None, "warmup and initial task")
        previous = np.asarray(warm[0]["detail"]["warm10"][-5:], dtype=np.float32)
        context = tasks[0]["root_context"]
        for stage, event in enumerate(encoded):
            item = records[stage]["detail"]
            boundary, frame = item["boundary"], item["frame"]
            need(boundary == event["end"] and frame["sha256"] == event["rgb_sha256"], "frame/encoding binding")
            observations = [e["detail"]["receipt"] for e in section if e["origin"] == "physical" and
                            e["resource"] == "native_fresh_observation" and
                            e["detail"]["receipt"]["phase"] == "plan" and
                            e["detail"]["receipt"]["step"] == boundary]
            need(len(observations) == 1 and Path(observations[0]["path"]).name == frame["file"] and
                 observations[0]["file_sha256"] == frame["sha256"] and
                 observations[0]["pixel_sha256"] == frame["pixel_sha256"], "physical observation/frame binding")
            desc = item["arrays"]["latent"]
            path = directory / desc["file"]
            need(Path(desc["file"]).name == desc["file"] and file_sha(path) == desc["sha256"],
                 "encoded latent SHA")
            with path.open("rb") as f:
                latent = np.load(f, allow_pickle=False)
                need(f.read(1) == b"", "latent EOF")
            raw = np.asarray(attempts[stage]["raw"], dtype=np.float32)
            context = dual.prior.old.join(context, previous, raw, raw, latent,
                       np.asarray(tasks[0]["action_mean"], np.float32),
                       np.asarray(tasks[0]["action_std"], np.float32))
            need(core.digest(context) == event["context_sha256"], "saved physical feedback context")
            previous = raw
    return dict(native=native, dispatches_bound=len(dispatch), returns_bound=len(returned),
                encoded_contexts_bound=len(encoded), binding="saved_records_only")


def consume(directory, end_sha, unit, max_decision_bytes, max_native_bytes, *, check_service=service_stopped):
    """Only read after stopped-service check and an externally pinned end.json."""
    before = check_service(unit)
    root = Path(directory)
    need(root.is_absolute() and root.resolve() == root and root.is_dir() and not root.is_symlink(),
         "canonical run directory")
    end = pinned_json(root / "end.json", end_sha, 8 * 1024**2)
    need(end.get("schema") == DRIVER + ".end" and end.get("full_saved_strict") is False and
         end.get("physical_certified") is False and end.get("scientific_pass") is False,
         "driver terminal schema")
    records = read_journal(root / "decision/journal.jsonl", end["decision_sha256"], max_decision_bytes)
    payloads = [r["payload"] for r in records]
    need(payloads[0].get("kind") == "journal_begin" and payloads[-1].get("kind") == "dual_journal_end" and
         payloads[0]["pins"] == payloads[-1]["pins"], "decision journal terminal/pins")
    rows = payloads[-1]["rows"]
    need(type(rows) is list and len(rows) == 6 and rows == end["rows"], "terminal six rows")
    need(payloads[1].get("kind") == "dual_fixed_denominator", "fixed denominator")
    initial = payloads[1]["rows"]
    need(type(initial) is list and len(initial) == 6 and
         all((x["slot"], x["root"], x["policy"]) == ident and x["status"] == "not_started" and
             x["result"] is None and x["cleanup"] == "not_required"
             for x, ident in zip(initial, dual.ROSTER, strict=True)), "initial roster")
    accounts, boundaries, cursor = [], [], 2
    for summary, ident in zip(rows, dual.ROSTER, strict=True):
        slot, root_index, policy = ident
        need((summary["slot"], summary["root"], summary["policy"]) == ident and cursor < len(records) and
             payloads[cursor] == dict(kind="dual_slot_attempt", slot=slot, root=root_index, policy=policy),
             "slot attempt order")
        start = cursor
        cursor += 1
        begin = cursor
        while cursor < len(records) - 1 and payloads[cursor].get("kind") != "dual_slot_return":
            need(payloads[cursor].get("kind") not in ("dual_slot_attempt", "dual_fixed_denominator",
                                                        "dual_journal_end"), "slot nesting")
            cursor += 1
        need(cursor < len(records) - 1, "missing slot return")
        row = payloads[cursor]["row"]
        need({k: row[k] for k in ("slot", "root", "policy", "status", "cleanup", "error")} == summary,
             "terminal row binding")
        if row["result"] is None:
            need(row["status"] in ("setup_failed", "cleanup_failed"), "missing slot result")
            account = None
        else:
            account = audit_slot(row, records[begin:cursor])
        need(row["cleanup"] in ("returned", "failed", "not_required") and
             (row["cleanup"] != "failed" or row["status"] == "cleanup_failed"), "cleanup state")
        accounts.append(account)
        boundaries.append((start, cursor))
        cursor += 1
    need(cursor == len(records) - 1, "unexpected terminal events")
    for r in range(2):
        roots = {payloads[boundaries[i][1]]["row"]["result"]["root_sha256"]
                 for i in range(6) if rows[i]["root"] == r and
                 payloads[boundaries[i][1]]["row"]["result"] is not None}
        need(len(roots) <= 1, "same root digest")
    directory = root / "native"
    need(directory.is_dir() and not directory.is_symlink(), "native directory")
    expected_inventory = end["native_files"]
    need(type(expected_inventory) is list and expected_inventory, "native inventory")
    inventory = []
    for p in sorted(directory.iterdir()):
        need(p.is_file() and not p.is_symlink(), "flat native inventory")
        inventory.append(dict(name=p.name, bytes=p.stat().st_size, sha256=file_sha(p)))
    need(inventory == expected_inventory, "native inventory hashes")
    journal_item = next((x for x in inventory if x["name"] == "events.jsonl"), None)
    need(journal_item is not None, "native journal missing")
    inspected = transport.inspect(directory / "events.jsonl", journal_item["sha256"], max_native_bytes)
    events = inspected["events"]
    need(not inspected["trailing_bytes"] and len(events) == end["native_index"] and
         inspected["tail"] == end["native_tail"] and len(inspected["pending"]) == end["native_pending"],
         "native terminal journal state")
    artifacts = native_saved._artifacts(directory, events, inventory)
    spans, active = [[] for _ in range(6)], None
    begins, ends = {}, {}
    for event in events:
        res, detail = event["resource"], event["detail"]
        if event["origin"] == "physical" and res == ADAPTER + "_slot_begin":
            slot = detail["slot"]
            need(active is None and type(slot) is int and 0 <= slot < 6 and slot not in begins, "native slot begin")
            start, _ = boundaries[slot]
            need(detail["root"] == rows[slot]["root"] and detail["policy"] == rows[slot]["policy"] and
                 detail["decision_index"] == start + 1 and detail["decision_tail"] == records[start]["sha256"],
                 "native/decision start anchor")
            active = slot
            begins[slot] = event
        elif event["origin"] == "physical" and res == ADAPTER + "_slot_end":
            slot = detail["slot"]
            need(active == slot and slot not in ends, "native slot end")
            _, stop = boundaries[slot]
            need(detail["root"] == rows[slot]["root"] and detail["policy"] == rows[slot]["policy"] and
                 detail["decision_index"] == stop and detail["decision_tail"] == records[stop - 1]["sha256"] and
                 detail["cleanup"] == rows[slot]["cleanup"], "native/decision end anchor")
            spans[slot].append(event)
            ends[slot], active = event, None
            continue
        if active is not None:
            spans[active].append(event)
    need(active is None or rows[active]["status"] in ("setup_failed", "cleanup_failed"),
         "unclosed active native slot")
    for slot, row in enumerate(rows):
        if accounts[slot] is not None:
            need(slot in begins and slot in ends, "missing native slot anchors")
    native_accounts = [audit_native_binding(i, spans[i], accounts[i], directory) if spans[i] else
                       dict(binding="no_native_slot_records") for i in range(6)]
    after = check_service(unit)
    need(after == before and file_sha(root / "end.json") == end_sha, "stopped service or terminal changed")
    return dict(schema=NAME + ".receipt", status="saved_records_checked_not_certified",
                decision_sha256=end["decision_sha256"], end_sha256=end_sha,
                fixed_denominator=6, rows=rows,
                slot_accounts=[None if a is None else {k: v for k, v in a.items() if k not in ("tasks", "phases")}
                               for a in accounts], native_accounts=native_accounts,
                native_transport_complete=inspected["transport_complete"], artifacts=artifacts,
                source_authenticated=False, physical_certified=False, full_saved_strict=False,
                unconditional_equal_budget=False, scientific_pass=False,
                limitations="saved records and self-reported hooks only; no full source, device or physics attestation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--end-sha256", required=True)
    parser.add_argument("--unit", required=True, help="stopped systemd --user .service")
    parser.add_argument("--max-decision-bytes", type=int, required=True)
    parser.add_argument("--max-native-bytes", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = consume(args.directory, args.end_sha256, args.unit, args.max_decision_bytes, args.max_native_bytes)
    out = Path(args.output)
    need(out.is_absolute() and out.resolve() == out and not out.exists() and not out.is_symlink()
         and out.parent.is_dir(), "exclusive receipt")
    raw = core.canonical(result) + b"\n"
    with out.open("xb") as f:
        need(f.write(raw) == len(raw), "receipt write")
        f.flush()
        os.fsync(f.fileno())
    journal.syncdir(out.parent)
    print(hashlib.sha256(raw).hexdigest())


if __name__ == "__main__":
    main()
