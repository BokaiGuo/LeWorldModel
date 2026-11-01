"""Verify one directly bound physical reset journal without asserting root equivalence."""
import argparse
import json
from pathlib import Path

from experiments.spectral_reachability import codex_p1_original_reset_inventory_20261101 as original

NAME = "p1_constructive_moving_history_physical_20261027_v1_main01"
SOURCE = "p1_constructive_moving_history_source_20261027_v1_main01"
SOURCE_SHA = "c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6"
REPORT_SHA = "a5322dc4f483e82b5ed9d7b3996cf8a63845261d65774f90a4ae341773016aaf"


def verify(report_raw, journal_raw, source_raw):
    if original.sha(report_raw) != REPORT_SHA or original.sha(source_raw) != SOURCE_SHA:
        raise ValueError("pinned report or source drift")
    if len(journal_raw) > 8 * 1024 * 1024 or not journal_raw.endswith(b"\n"):
        raise ValueError("journal limit or truncation")
    report, source = original.parse(report_raw), original.parse(source_raw)
    if report.get("sourcepack_sha256") != SOURCE_SHA or report.get("journal_sha256") != original.sha(journal_raw):
        raise ValueError("source or journal binding")
    events, previous = [], None
    for i, line in enumerate(journal_raw.splitlines()):
        event = original.parse(line)
        digest = original.sha(json.dumps({k: v for k, v in event.items() if k != "event_sha256"},
                                         sort_keys=True, allow_nan=False).encode())
        if event.get("index") != i or event.get("previous_sha256") != previous or event.get("event_sha256") != digest:
            raise ValueError("journal chain mismatch")
        events.append(event)
        previous = digest
    if not events or events[0].get("kind") != "run_started" or events[-1].get("kind") != "run_finished":
        raise ValueError("run boundary")
    start, end = events[0], events[-1]
    prepare = start["prepare"]
    if (prepare.get("sourcepack_sha256") != SOURCE_SHA or prepare.get("contract") != report.get("contract")
            or end.get("counts") != report.get("counts") or end.get("status") != report.get("status")
            or previous != report.get("journal_chain_tail")):
        raise ValueError("prepare, finish or chain binding")
    slots = {}
    for row in source["rows"]:
        slot = row["slot"]
        if slot["index"] in slots:
            raise ValueError("duplicate source slot")
        slots[slot["index"]] = slot
    branches = {}
    for i, row in enumerate(report["rows"]):
        key = (i, row["root_slot"], row["arm"], row["selection_digest"])
        if key in branches or row["root_slot"] not in slots:
            raise ValueError("ambiguous report branch")
        branches[key] = row
    records, pending, seen = [], {}, set()
    for event in events:
        if event.get("resource") != "reset":
            continue
        branch_key = tuple(event.get(k) for k in ("branch", "root_slot", "arm", "selection_digest"))
        kind = event.get("kind_of_reset")
        seed = event.get("seed")
        if branch_key not in branches or kind not in ("original_seed", "placement"):
            raise ValueError("unmatched reset branch")
        if seed != (slots[branch_key[1]]["reset_seed"] if kind == "original_seed" else 0):
            raise ValueError("reset seed mismatch")
        key = branch_key + (kind, seed)
        if event.get("kind") == "operation_attempted":
            if key in seen:
                raise ValueError("duplicate reset")
            seen.add(key)
            row = {"branch": branch_key[0], "source_slot": branch_key[1], "arm": branch_key[2],
                   "selection_digest": branch_key[3], "kind_of_reset": kind, "seed": seed,
                   "start_line": event["index"] + 1, "start_event_sha256": event["event_sha256"],
                   "completion_line": None, "completion_event_sha256": None,
                   "source_recipe_reference": {"sourcepack_sha256": SOURCE_SHA, "slot": branch_key[1]},
                   "physical_root_equivalence": "unknown", "independent_root": "unknown"}
            records.append(row)
            pending[key] = row
        elif event.get("kind") == "operation_completed":
            if key not in pending:
                raise ValueError("unpaired reset")
            row = pending.pop(key)
            row["completion_line"] = event["index"] + 1
            row["completion_event_sha256"] = event["event_sha256"]
        else:
            raise ValueError("unexpected reset event")
    counts = report["counts"]["reset"]
    if len(records) != counts["attempted"] or sum(r["completion_line"] is not None for r in records) != counts["completed"]:
        raise ValueError("reset count mismatch")
    return {"schema": "codex_p1_bound_physical_reset_20261101_exclusive.v1",
            "report_sha256": original.sha(report_raw), "journal_sha256": original.sha(journal_raw),
            "sourcepack_sha256": SOURCE_SHA, "records": records,
            "scope": "source recipe reference only; report describes new moving roots/tasks",
            "physical_root_equivalence": "unknown", "training_exclusion": "unknown",
            "freshness": "unknown", "independent_roots_certified": False,
            "P1_admitted": False, "new_environment_or_gpu_calls": 0}


def build(root=original.ROOT):
    root = Path(root)
    directory = root / "results" / NAME
    source = root / "results" / SOURCE / "sourcepack.json"
    try:
        result = verify((directory / "report.json").read_bytes(), (directory / "journal.jsonl").read_bytes(), source.read_bytes())
        result.update({"report_path": str(directory / "report.json"), "journal_path": str(directory / "journal.jsonl"),
                       "sourcepack_path": str(source), "status": "verified_source_record"})
        return result
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"schema": "codex_p1_bound_physical_reset_20261101_exclusive.v1", "status": "unknown",
                "reason": str(exc), "records": [], "independent_roots_certified": False,
                "P1_admitted": False, "new_environment_or_gpu_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = build()
    data = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as stream:
        stream.write(data)
    print(json.dumps({"status": result["status"], "records": len(result["records"]),
                      "output": str(args.output), "sha256": original.sha(data)}))


if __name__ == "__main__":
    main()
