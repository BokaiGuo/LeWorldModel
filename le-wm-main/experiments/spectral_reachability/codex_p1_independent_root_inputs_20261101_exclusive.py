"""CPU-only source-record inventory; source-slot equivalence is not independence."""
import argparse
import base64
import gzip
import json
from pathlib import Path

from experiments.spectral_reachability import codex_p1_reset_sources_aggregate_20261101_exclusive as base
from experiments.spectral_reachability import codex_p1_original_reset_inventory_20261101 as original

OVERLAY = "p1_moving_target_overlay_source_20261027_v1_main01"
REPORT_SHA = "85ca1df3de910628bfd6c5bd1a0a482f61880a2ee76bfd404732a77dd8c7666a"
PREPARE_SHA = "88957b9d251595e5e35ed46ba4a9aa003eb937bbfff0f58ae12540418dcb39a1"
PARENT_SHA = base.SOURCES["p1_constructive_moving_history_source_20261027_v1_main01"]


def overlay(report_raw, journal_raw, report_path, journal_path, prepare_raw):
    if original.sha(report_raw) != REPORT_SHA:
        raise ValueError("report hash mismatch")
    if len(journal_raw) > 16 * 1024 * 1024 or not journal_raw.endswith(b"\n"):
        raise ValueError("journal limit or truncation")
    report = original.parse(report_raw)
    if original.sha(prepare_raw) != PREPARE_SHA:
        raise ValueError("prepare hash mismatch")
    envelope = original.parse(prepare_raw)
    decoded = gzip.decompress(base64.b64decode(envelope["gzip_base64"], validate=True))
    if len(decoded) > 2 * 1024 * 1024:
        raise ValueError("prepare size limit")
    plan = original.parse(decoded)
    plan_hash = original.sha(json.dumps(plan, sort_keys=True, allow_nan=False).encode())
    parents = [value for key, value in plan["old_inputs"].items()
               if key.endswith("/p1_constructive_moving_history_source_20261027_v1_main01/sourcepack.json")]
    if plan_hash != envelope.get("plan_sha256") or plan_hash != report.get("prepare_id") or parents != [PARENT_SHA]:
        raise ValueError("parent or prepare identity mismatch")
    events, previous = [], None
    for index, line in enumerate(journal_raw.splitlines()):
        event = original.parse(line)
        digest = original.sha(json.dumps({k: v for k, v in event.items() if k != "event_sha256"},
                                         sort_keys=True, allow_nan=False).encode())
        if event.get("index") != index or event.get("previous_sha256") != previous or event.get("event_sha256") != digest:
            raise ValueError("journal chain mismatch")
        previous = digest
        events.append(event)
    if not events or events[0].get("kind") != "run_started" or events[-1].get("kind") != "run_finished":
        raise ValueError("run boundary")
    start, end = events[0], events[-1]
    if (previous != report.get("journal_chain_tail") or start.get("prepare_id") != report.get("prepare_id")
            or start.get("contract") != report.get("contract") or end.get("counts") != report.get("counts")
            or end.get("status") != report.get("status") or report.get("source_verified") is not True):
        raise ValueError("report binding")
    rows, pending = [], {}
    for event in events:
        if event.get("resource") != "reset":
            continue
        key = (event.get("key"), event.get("source_slot"), event.get("role"), event.get("seed"))
        if event.get("kind") == "operation_attempted":
            if key in pending or key[1] not in (0, 1) or key[2] not in ("seed", "placement"):
                raise ValueError("ambiguous reset")
            row = {"source_slot": key[1], "phase_key": key[0], "role": key[2], "seed": key[3],
                   "start_line": event["index"] + 1, "start_event_sha256": event["event_sha256"],
                   "completion_line": None, "completion_event_sha256": None}
            rows.append(row)
            pending[key] = row
        elif event.get("kind") == "operation_completed":
            if key not in pending:
                raise ValueError("unpaired completion")
            row = pending.pop(key)
            row["completion_line"] = event["index"] + 1
            row["completion_event_sha256"] = event["event_sha256"]
        else:
            raise ValueError("unexpected reset kind")
    counts = report["counts"]["reset"]
    if len(rows) != counts["attempted"] or sum(r["completion_line"] is not None for r in rows) != counts["completed"]:
        raise ValueError("reset count mismatch")
    return {"report_path": str(report_path), "report_sha256": original.sha(report_raw),
            "journal_path": str(journal_path), "journal_sha256": original.sha(journal_raw),
            "prepare_sha256": original.sha(prepare_raw),
            "parent_sourcepack_sha256": PARENT_SHA, "events": rows}


def build(root=original.ROOT):
    root = Path(root)
    aggregate = base.aggregate(root / "results")
    path = root / "results" / OVERLAY
    failures = list(aggregate["unknown_sources"])
    extra = None
    try:
        extra = overlay((path / "report.json").read_bytes(), (path / "journal.jsonl").read_bytes(),
                        path / "report.json", path / "journal.jsonl",
                        (root / "results/p1_moving_target_overlay_source_20261027_v1_prepare_main01.json").read_bytes())
    except (OSError, ValueError, KeyError, TypeError, EOFError) as exc:
        failures.append({"source": str(path), "status": "unknown", "reason": str(exc)})
    # The report declares reuse of two source slots; these are task-source references,
    # not proof of equal complete physical state or disjointness from other sources.
    groups = []
    if extra is not None:
        for slot in (0, 1):
            groups.append({"sourcepack_sha256": PARENT_SHA, "source_slot": slot,
                           "overlay_event_indices": [i for i, row in enumerate(extra["events"]) if row["source_slot"] == slot],
                           "equivalence": "reported_source_slot_reuse",
                           "physical_root_identity": "unknown", "independence": "unknown"})
    return {"schema": "codex_p1_independent_root_inputs_20261101_exclusive.v1",
            "baseline": aggregate, "additional_overlay": extra, "dedup_input_source_slot_groups": groups,
            "unknown_sources": failures, "physical_root_deduplication": "unknown",
            "global_history_coverage": "unknown", "training_exclusion": "unknown",
            "freshness": "unknown", "independent_roots_certified": False, "P1_admitted": False,
            "new_environment_or_gpu_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build()
    data = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as stream:
        stream.write(data)
    print(json.dumps({"output": str(args.output), "sha256": original.sha(data),
                      "baseline_events": result["baseline"]["reset_records"],
                      "overlay_events": len(result["additional_overlay"]["events"]) if result["additional_overlay"] else 0,
                      "unknown_sources": len(result["unknown_sources"])}))


if __name__ == "__main__":
    main()
