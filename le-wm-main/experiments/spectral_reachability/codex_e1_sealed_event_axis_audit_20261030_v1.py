"""Audit fixed E1 sealed hook events without loading models, P1 assets or physics."""
import hashlib
import json
from pathlib import Path
import sys
import zlib

BASE = Path(__file__).resolve().parents[2] / "results"
DIR = BASE / "agent_e1_six_slot_common_cuda_20261029_v2_cuda_MAIN01"
REPORT = DIR / "report.json"
EVENTS = DIR / "events.jsonl.gz"
RECEIPT = BASE / "agent_e1_six_slot_common_cuda_20261029_v2_consume_MAIN01.json"
REPORT_SHA = "cf3cb2a506ec1739ca53f58faa00bb07972c6e546cd75e02e0c609dccd7b7578"
EVENT_SHA = "2a3570a54333ff728136b03b4cfebbf203f42a48cc07d3567dbb20441762f5df"
RECEIPT_SHA = "1ca8e949cc4b219b9b03adb3d34ee859e924e0078f9f81c519be2c03ed77be3d"
ARMS = ("arbitrary_latent_waypoint", "existing_learned_prior", "dynamics_compiled_witness")
MAP = {"predict_API": "predict_api_calls", "predictor": "predictor_module_calls",
       "action_encoder": "action_encoder_calls", "pred_proj": "pred_proj_calls"}
MAX_RAW = 64 * 1024 * 1024


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"


def parse(raw):
    def pairs(items):
        obj = {}
        for k, v in items:
            need(k not in obj, "duplicate JSON key")
            obj[k] = v
        return obj
    def bad(_):
        raise ValueError("nonfinite JSON")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)


def read_bound(path, expected, limit):
    need(path.is_file() and not path.is_symlink(), "source must be regular file")
    with path.open("rb") as file:
        raw = file.read(limit + 1)
    need(len(raw) <= limit and sha(raw) == expected, "source digest or size mismatch")
    return raw


def decode(raw):
    d = zlib.decompressobj(31)
    data = d.decompress(raw, MAX_RAW + 1)
    need(len(data) <= MAX_RAW and d.eof and not d.unused_data and not d.unconsumed_tail,
         "compressed stream invalid or too large")
    need(data.endswith(b"\n"), "journal truncated")
    previous = "0" * 64
    records = []
    for line in data.splitlines(keepends=True):
        need(len(line) <= 1024 * 1024 and len(records) < 10000, "journal bounds")
        item = parse(line)
        need(type(item) is dict and set(item) == {"index", "previous", "event", "sha256"}
             and canonical(item) == line, "noncanonical record")
        body = {"index": len(records), "previous": previous, "event": item["event"]}
        need(type(item["index"]) is int and item["index"] == len(records)
             and item["previous"] == previous and item["sha256"] == sha(canonical(body)),
             "journal chain mismatch")
        previous = item["sha256"]
        records.append(item)
    return records, previous


def audit(records, terminal, report, receipt):
    need(report["schema"] == "agent_e1_six_slot_common_cuda_20261029_v2.report" and
         report["status"] == "all_six_sealed" and report["errors"] == [] and
         report["artifacts"]["events.jsonl.gz"] == EVENT_SHA and
         report["journal_events"] == len(records) and report["journal_sha256"] == terminal,
         "report journal mismatch")
    need(receipt["schema"] == "agent_e1_six_slot_common_cuda_20261029_v2.consume" and
         receipt["status"] == "all_six_saved_controller_and_replays_verified" and
         receipt["report_sha256"] == REPORT_SHA and receipt["scientific_admission"] is False and
         len(receipt["rows"]) == len(report["rows"]) == 6, "receipt mismatch")
    groups = {(i, phase): {"next": 0, "open": {}, "axes": {x: [] for x in MAP},
                            "counts": {x: [0, 0] for x in MAP}, "samples": [0, 0]}
              for i in range(6) for phase in ("controller", "replay")}
    for rec in records:
        event = rec["event"]
        need(type(event) is dict, "invalid event")
        if "event" not in event:
            continue
        need(set(event) == {"slot", "phase", "event"} and type(event["slot"]) is int and
             (event["slot"], event["phase"]) in groups and type(event["event"]) is dict,
             "invalid event scope")
        inner = event["event"]
        kind = inner.get("kind")
        if kind not in ("dispatch_attempted", "dispatch_returned"):
            need(not (type(kind) is str and kind.startswith("dispatch_")), "unknown dispatch event")
            continue
        g = groups[(event["slot"], event["phase"])]
        ticket = inner["receipt"]
        resource, index, samples = ticket["resource"], ticket["index"], ticket["samples"]
        need(resource in MAP and type(index) is int and type(samples) is int and samples > 0,
             "invalid ticket")
        ident = (index, resource)
        if kind == "dispatch_attempted":
            need(index == g["next"] and ticket["status"] == "pending" and ident not in g["open"],
                 "invalid attempted ticket")
            g["next"] += 1
            g["open"][ident] = (inner["evaluation_id"], ticket)
            g["counts"][resource][0] += 1
            if resource == "predictor":
                g["samples"][0] += samples
        else:
            need(ticket["status"] == "returned" and ident in g["open"], "return without attempt")
            evaluation, attempted = g["open"].pop(ident)
            projection = {k: v for k, v in ticket.items() if k not in ("status", "output")}
            need(evaluation == inner["evaluation_id"] and
                 projection == {k: v for k, v in attempted.items() if k != "status"},
                 "attempt/return identity mismatch")
            g["counts"][resource][1] += 1
            if resource == "predictor":
                g["samples"][1] += samples
        g["axes"][resource].append({"index": rec["index"], "sha256": rec["sha256"]})
    output = []
    for i in range(6):
        rr, row = report["rows"][i], receipt["rows"][i]
        root, arm = i // 3, ARMS[i % 3]
        need(type(row["root"]) is int and type(rr["root"]) is int and
             row["root"] == rr["root"] == root and row["arm"] == rr["arm"] == arm and
             row["selection_sha256"] == rr["selection_sha256"] and row["status"] == "sealed",
             "slot binding")
        for phase in ("controller", "replay"):
            g = groups[(i, phase)]
            need(not g["open"], "unreturned attempts")
            measures = {}
            for resource, axis in MAP.items():
                a, c = g["counts"][resource]
                need(row[phase + "_resources"][resource] == {"attempted": a, "completed": c},
                     "hook count mismatch")
                measures[axis] = event_set(i, phase, axis, g["axes"][resource], a, c)
            a, c = g["samples"]
            need(row[phase + "_samples"] == {"attempted": a, "completed": c}, "sample count mismatch")
            measures["predictor_samples"] = event_set(i, phase, "predictor_samples",
                                                       g["axes"]["predictor"], a, c)
            output.append({"slot": i, "root": root, "arm": arm, "phase": phase, "measurements": measures})
    return {"schema": "codex.e1.sealed_event_axis_audit.v1", "status": "partial_resource_chain_only",
            "journal_sha256": EVENT_SHA, "journal_terminal": terminal, "journal_events": len(records),
            "report_sha256": REPORT_SHA, "receipt_sha256": RECEIPT_SHA, "groups": output,
            "end_to_end_complete": False, "equal_budget_certified": False, "ranking_authorized": False,
            "scientific_admission": False,
            "boundary": "derived from pinned sealed events; no arrays, upstream fullstrict, P1 or model execution"}


def event_set(slot, phase, axis, refs, attempted, completed):
    body = {"schema": "codex.e1.axis_event_set.v1", "journal_sha256": EVENT_SHA,
            "slot": slot, "phase": phase, "axis": axis, "records": refs}
    return {"attempted": attempted, "completed": completed, "event_set_sha256": sha(canonical(body))}


def main():
    try:
        report = parse(read_bound(REPORT, REPORT_SHA, 4 * 1024 * 1024))
        receipt = parse(read_bound(RECEIPT, RECEIPT_SHA, 4 * 1024 * 1024))
        records, terminal = decode(read_bound(EVENTS, EVENT_SHA, 16 * 1024 * 1024))
        result = audit(records, terminal, report, receipt)
    except (ValueError, KeyError, TypeError, OSError, zlib.error) as exc:
        print(json.dumps({"status": "blocked_invalid", "reason": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
