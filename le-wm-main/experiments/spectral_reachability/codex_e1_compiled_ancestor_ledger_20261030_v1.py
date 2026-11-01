"""Audit pinned E1 compiled-target ancestor hook events; no production or P1 files."""
import base64
import hashlib
import json
from pathlib import Path
import struct
import sys
import zlib

from . import codex_e1_sealed_event_axis_audit_20261030_v1 as util

BASE = Path(__file__).resolve().parents[2] / "results"
OLD = BASE / "agent_experiment1_common_controller_20261028_v1_prepared_MAIN01.json"
NEW = BASE / "agent_e1_six_slot_common_prepare_20261029_v1_prepare_MAIN01.json"
DIR = BASE / "agent_kinodynamic_matched_domain_20261028_v1_cuda_MAIN01"
REPORT, EVENTS = DIR / "report.json", DIR / "events.jsonl.gz"
OUTPUT = BASE / "codex_e1_compiled_ancestor_ledger_20261030_v1_LOCAL01.json"
OLD_SHA = "8c472485d44c40ce3d95f8043d2c58f9ec2694fef74786a840d1699d5a1a86bd"
NEW_SHA = "cd96f6e28b25d3b7e3b6649c760b2f111747aec5c0ce9c8a41741318209df8a0"
REPORT_SHA = "d4976aeca357b5324c74be8b2b7e3bfc254e82b834b3acfe1b0ab90a1ef4614c"
EVENT_SHA = "ef8c43ec73ba9cc66f9f1708bd7c348651817f211c0a43302f24637c7469b623"
ARM = "matched_domain_H2_beam20x6"
RESOURCES = ("predict_API", "predictor", "action_encoder", "pred_proj")


def need(ok, message):
    if not ok:
        raise ValueError(message)


def events_from_gzip(raw):
    d = zlib.decompressobj(31)
    expanded = d.decompress(raw, 256 * 1024 * 1024 + 1)
    need(len(expanded) <= 256 * 1024 * 1024 and d.eof and not d.unused_data
         and not d.unconsumed_tail and expanded.endswith(b"\n"), "incomplete gzip")
    return [(util.parse(line), hashlib.sha256(line).hexdigest()) for line in expanded.splitlines(keepends=True)]


def analyze(old, new, report, events):
    need(old["schema"] == "agent_experiment1_common_controller_20261028_v1.prepare" and
         new["schema"] == "agent_e1_six_slot_common_prepare_20261029_v1.prepare" and
         report["schema"] == "agent_kinodynamic_matched_domain_20261028_v1.report" and
         report["status"] == "matched_domain_beam_plans_sealed_not_admitted" and
         report["errors"] == [] and report["artifact_sha256"]["events.jsonl.gz"] == EVENT_SHA,
         "report identity")
    need(old["bindings_sha256"][str(REPORT)] == REPORT_SHA and old["bindings_sha256"][str(EVENTS)] == EVENT_SHA,
         "old prepare source binding")
    need(len(report["rows"]) == len(old["compiled"]) == len(old["slots"]) == 2 and
         len(new["rows"]) == 6, "fixed root denominator")
    groups = [[], []]
    prev = -1
    for index, (item, line_sha) in enumerate(events):
        need(set(item) == {"source_slot", "arm", "task_id", "event"} and
             type(item["source_slot"]) is int and item["source_slot"] in (0, 1)
             and item["source_slot"] >= prev and item["arm"] == ARM, "invalid event scope/order")
        prev = item["source_slot"]
        groups[prev].append((index, item, line_sha))
    output = []
    for root in (0, 1):
        row, source, compiled, target = report["rows"][root], old["slots"][root], old["compiled"][root], new["rows"][2 + 3 * root]
        need(source["source_slot"] == compiled["source_slot"] == row["source_slot"] == root and
             target["root_slot"] == root and target["arm"] == "dynamics_compiled_witness" and
             target["origin"]["kind"] == "inherited_original_target" and
             target["task_sha256"] == source["task_sha256"] and
             target["blind_sha256"] == source["blind_sha256"] == row["blind_sha256"] and
             target["task_id"] == source["task_id"] == row["task_id"] and
             compiled["selection_sha256"] == row["selection_sha256"] and
             compiled["result_sha256"] == report["artifact_sha256"][f"root_{root}.result.json.gz"],
             "ancestor target/task binding")
        vector = compiled["target"]
        need(type(vector) is list and len(vector) == 192, "target size")
        packed = struct.pack("<192f", *vector)
        need(packed == base64.b64decode(target["target_f32_le_base64"], validate=True) and
             hashlib.sha256(packed).hexdigest() == target["target_data_sha256"], "target bytes mismatch")
        group = groups[root]
        need(bool(group) and group[0][1]["event"]["kind"] == "root_started" and
             group[0][1]["event"]["task_sha256"] == source["task_sha256"] and
             group[-1][1]["event"]["kind"] == "plan_sealed" and
             group[-1][1]["event"]["selection_sha256"] == row["selection_sha256"],
             "root event boundaries")
        counts = {k: [0, 0] for k in RESOURCES}
        refs = {k: [] for k in RESOURCES}
        samples = [0, 0]
        pending = {}
        next_index = 0
        for index, item, line_sha in group:
            need(item["task_id"] == source["task_id"], "event task mismatch")
            ev = item["event"]
            kind = ev["kind"]
            if kind not in ("dispatch_attempted", "dispatch_returned"):
                need(not kind.startswith("dispatch_"), "unknown dispatch")
                continue
            ticket = ev["receipt"]
            resource, ticket_index, n = ticket["resource"], ticket["index"], ticket["samples"]
            need(resource in RESOURCES and type(ticket_index) is int and type(n) is int and n > 0,
                 "invalid ticket")
            key = (ticket_index, resource)
            if kind == "dispatch_attempted":
                need(ticket_index == next_index and ticket["status"] == "pending", "invalid attempt")
                next_index += 1
                pending[key] = (ev["evaluation_id"], ticket)
                column = 0
            else:
                need(key in pending and ticket["status"] == "returned", "orphan return")
                evaluation, old_ticket = pending.pop(key)
                need(evaluation == ev["evaluation_id"] and
                     {k: v for k, v in ticket.items() if k not in ("status", "output")} ==
                     {k: v for k, v in old_ticket.items() if k != "status"}, "ticket identity mismatch")
                column = 1
            counts[resource][column] += 1
            if resource == "predictor":
                samples[column] += n
            refs[resource].append({"line": index, "line_sha256": line_sha})
        need(not pending, "unreturned attempt")
        axes = {}
        for resource in RESOURCES:
            a, c = counts[resource]
            need(row["resources"][resource] == {"attempted": a, "completed": c}, "receipt count mismatch")
            body = {"schema": "codex.e1.compiled_ancestor_events.v1", "journal_sha256": EVENT_SHA,
                    "root": root, "axis": resource, "records": refs[resource]}
            axes[resource] = {"attempted": a, "completed": c,
                              "event_set_sha256": util.sha(util.canonical(body))}
        need(row["predictor_samples"] == {"attempted": samples[0], "completed": samples[1]},
             "sample count mismatch")
        body = {"schema": "codex.e1.compiled_ancestor_events.v1", "journal_sha256": EVENT_SHA,
                "root": root, "axis": "predictor_samples", "records": refs["predictor"]}
        axes["predictor_samples"] = {"attempted": samples[0], "completed": samples[1],
                                     "event_set_sha256": util.sha(util.canonical(body))}
        output.append({"root": root, "arm": target["arm"], "target_sha256": target["target_data_sha256"],
                       "task_sha256": target["task_sha256"], "selection_sha256": row["selection_sha256"],
                       "event_axes": axes})
    return {"schema": "codex.e1.compiled_ancestor_ledger.v1", "status": "incomplete",
            "report_sha256": REPORT_SHA, "event_sha256": EVENT_SHA, "old_prepare_sha256": OLD_SHA,
            "new_prepare_sha256": NEW_SHA, "rows": output,
            "scope": "ancestor beam run observed hooks including search and replay, not entire proposal cost",
            "unresolved": ["earlier_and_shared_ancestor_costs", "other_proposal_and_resource_axes",
                           "record_coverage_and_flops", "end_to_end_fairness"],
            "equal_budget_certified": False, "ranking_authorized": False, "scientific_admission": False}


def load():
    old = util.parse(util.read_bound(OLD, OLD_SHA, 4 * 1024 * 1024))
    new = util.parse(util.read_bound(NEW, NEW_SHA, 4 * 1024 * 1024))
    report = util.parse(util.read_bound(REPORT, REPORT_SHA, 4 * 1024 * 1024))
    return old, new, report, events_from_gzip(util.read_bound(EVENTS, EVENT_SHA, 32 * 1024 * 1024))


def main():
    try:
        need(len(sys.argv) == 1, "no arguments")
        data = json.dumps(analyze(*load()), sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
        with OUTPUT.open("xb") as stream:
            stream.write(data)
        print(json.dumps({"path": str(OUTPUT), "sha256": util.sha(data), "status": "incomplete"}))
    except (ValueError, KeyError, TypeError, OSError, zlib.error, struct.error) as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc)}), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
