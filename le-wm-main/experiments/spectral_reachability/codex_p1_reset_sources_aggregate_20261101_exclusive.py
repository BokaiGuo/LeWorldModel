"""CPU-only, bounded inventory of four historical sourcepack/journal pairs."""
import argparse
import json
from pathlib import Path

from experiments.spectral_reachability import codex_p1_original_reset_inventory_20261101 as original

ROOT = original.ROOT
SOURCES = {
    "p1_conditional_calibration2_main_20261027_v1": original.SOURCE_SHA,
    "p1_observed_history_source_20261027_v1_main01": "0b89d77494acc6cf03bb0110e87231c1c240a253e844326acc668b7e4f93d546",
    "p1_constructive_reachable_source_20261027_v1_main01": "9ad3a414fd2148258524d4a7e9d4f070f59532ec2e5943cfa61f9145b3a41bd9",
    "p1_constructive_moving_history_source_20261027_v1_main01": "c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6",
}


def extract_other(source_raw, journal_raw, source_path, journal_path):
    source = original.parse(source_raw)
    if len(source_raw) > 8 * 1024 * 1024 or len(journal_raw) > 16 * 1024 * 1024 or not journal_raw.endswith(b"\n"):
        raise ValueError("input limit or incomplete journal")
    events, previous = [], None
    for index, line in enumerate(journal_raw.splitlines()):
        event = original.parse(line)
        body = {k: v for k, v in event.items() if k != "event_sha256"}
        digest = original.sha(json.dumps(body, sort_keys=True, allow_nan=False).encode())
        if event.get("index") != index or event.get("previous_sha256") != previous or event.get("event_sha256") != digest:
            raise ValueError("journal chain")
        previous = digest
        events.append(event)
    if not events or events[0].get("kind") != "run_started" or events[-1].get("kind") != "run_finished":
        raise ValueError("run boundary")
    if source.get("journal_sha256") is not None:
        if source["journal_sha256"] != original.sha(journal_raw) or events[0].get("prepare") != source.get("prepare"):
            raise ValueError("journal or prepare binding")
    elif source.get("journal_chain_tail") != previous:
        raise ValueError("journal tail binding")
    if events[-1].get("counts") != source.get("counts") or events[-1].get("status") != source.get("status"):
        raise ValueError("run finish mismatch")
    rows, pending = [], {}
    for event in events:
        if event.get("resource") != "reset":
            continue
        key = (event.get("slot"), event.get("candidate_id"), event.get("phase"), event.get("kind_of_reset"))
        kind = event.get("kind")
        if kind in ("operation_started", "operation_attempted"):
            if key in pending:
                raise ValueError("ambiguous reset pairing")
            row = {"source_path": source_path, "source_sha256": original.sha(source_raw),
                   "journal_path": journal_path, "journal_sha256": original.sha(journal_raw),
                   "start_line": event["index"] + 1, "start_event": event,
                   "completion_line": None, "completion_event": None,
                   "identity_status": "unknown_not_independence_evidence"}
            rows.append(row)
            pending[key] = row
        elif kind == "operation_completed":
            if key not in pending:
                raise ValueError("unpaired reset completion")
            row = pending.pop(key)
            row["completion_line"] = event["index"] + 1
            row["completion_event"] = event
        else:
            raise ValueError("unknown reset event kind")
    counts = source["counts"]["reset"]
    if len(rows) != counts["attempted"] or sum(r["completion_event"] is not None for r in rows) != counts["completed"]:
        raise ValueError("reset counts mismatch")
    return rows


def aggregate(base=ROOT / "results", sources=SOURCES):
    all_rows, failures = [], []
    for name, expected in sources.items():
        source_path = Path(base) / name / "sourcepack.json"
        journal_path = source_path.with_name("journal.jsonl")
        try:
            source_raw, journal_raw = source_path.read_bytes(), journal_path.read_bytes()
            if original.sha(source_raw) != expected:
                raise ValueError("pinned source SHA drift")
            if name == "p1_conditional_calibration2_main_20261027_v1":
                rows = original.extract(source_raw, journal_raw, str(source_path), str(journal_path))["events"]
                for row in rows:
                    row["identity_status"] = "original_attempt_not_independence_evidence"
            else:
                rows = extract_other(source_raw, journal_raw, str(source_path), str(journal_path))
            all_rows.extend(rows)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures.append({"source_path": str(source_path), "journal_path": str(journal_path),
                             "expected_source_sha256": expected, "status": "unknown",
                             "reason": str(exc)})
    return {"schema": "codex_p1_reset_sources_aggregate_20261101_exclusive.v1",
            "scope": "four explicitly pinned sourcepack/journal pairs only",
            "events": all_rows, "unknown_sources": failures,
            "source_count": len(sources), "reset_records": len(all_rows),
            "history_complete": False, "complete_deduplication": False,
            "independent_roots_certified": False, "training_exclusion_certified": False,
            "freshness_certified": False, "P1_admitted": False,
            "unknown": ["uninspected journals and sourcepacks, failed and unsaved runs",
                        "physical root equivalence across events and runs",
                        "training exposure and global freshness"],
            "new_environment_or_gpu_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = aggregate()
    data = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as stream:
        stream.write(data)
    print(json.dumps({"output": str(args.output.resolve()), "sha256": original.sha(data),
                      "reset_records": len(result["events"]), "unknown_sources": len(result["unknown_sources"])}))


if __name__ == "__main__":
    main()
