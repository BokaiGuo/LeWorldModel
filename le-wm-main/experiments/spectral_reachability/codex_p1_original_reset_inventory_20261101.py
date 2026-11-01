"""CPU-only extraction of one pinned historical reset journal; not certification."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/p1_conditional_calibration2_main_20261027_v1/sourcepack.json"
JOURNAL = SOURCE.with_name("journal.jsonl")
SOURCE_SHA = "728f14cb83c4fcd0ea4f5348bad2bff5b5dd372ab0185b6e6b3a26dbfda58057"
COORDS = ("candidate_id", "proposal_index", "attempt_index", "random_reset_seed", "placement_reset_seed")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def parse(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")))


def extract(source_raw, journal_raw, source_path=str(SOURCE), journal_path=str(JOURNAL)):
    if len(source_raw) > 128 * 1024 or len(journal_raw) > 8 * 1024 * 1024:
        raise ValueError("input size limit")
    source = parse(source_raw)
    if source.get("schema") != "p1_conditional_procedural_source_20261027_v1.sourcepack":
        raise ValueError("source schema")
    if sha(journal_raw) != source.get("journal_sha256") or not journal_raw.endswith(b"\n"):
        raise ValueError("journal binding or truncation")
    previous = None
    events = []
    for index, raw in enumerate(journal_raw.splitlines()):
        event = parse(raw)
        body = {k: v for k, v in event.items() if k != "event_sha256"}
        digest = sha(json.dumps(body, sort_keys=True, allow_nan=False).encode())
        if event.get("index") != index or event.get("previous_sha256") != previous or event.get("event_sha256") != digest:
            raise ValueError("journal chain")
        previous = digest
        events.append(event)
    if not events or events[0].get("kind") != "run_started" or events[0].get("plan") != source["plan"]:
        raise ValueError("run and plan mismatch")
    if events[-1].get("kind") != "run_finished" or events[-1].get("resources") != source["resources"] or events[-1].get("status") != source["status"]:
        raise ValueError("run finish mismatch")
    attempts = {}
    for i, slot in enumerate(source["slots"]):
        for j, proposal in enumerate(slot["proposals"]):
            for k, attempt in enumerate(proposal["attempts"]):
                key = tuple(attempt[c] for c in COORDS)
                if key in attempts:
                    raise ValueError("duplicate attempt")
                attempts[key] = (attempt, f"/slots/{i}/proposals/{j}/attempts/{k}")
    operations = {}
    for event in events:
        if event.get("kind") not in ("operation_started", "operation_completed") or event.get("operation") not in ("random_reset", "placement_reset"):
            continue
        coord = tuple(event[c] for c in COORDS)
        if coord not in attempts:
            raise ValueError("journal attempt missing from sourcepack")
        key = (coord, event["operation"])
        row = operations.setdefault(key, {})
        kind = event["kind"]
        if kind in row or (kind == "operation_completed" and "operation_started" not in row):
            raise ValueError("duplicate or unpaired reset")
        row[kind] = event
    rows = []
    for (coord, operation), pair in operations.items():
        attempt, pointer = attempts[coord]
        start = pair.get("operation_started")
        end = pair.get("operation_completed")
        if start is None:
            raise ValueError("missing start")
        if end and operation == "placement_reset" and end["receipt"]["state"] != attempt.get("state"):
            raise ValueError("placement state mismatch")
        rows.append({"coordinates": dict(zip(COORDS, coord)), "operation": operation,
                     "seed": coord[3] if operation == "random_reset" else coord[4],
                     "status": "recorded_completed" if end else "started_completion_unknown",
                     "state": end["receipt"]["state"] if end else None,
                     "source_attempt": {"path": source_path, "sha256": sha(source_raw), "pointer": pointer},
                     "start": {"path": journal_path, "sha256": sha(journal_raw), "line": start["index"] + 1, "event_sha256": start["event_sha256"]},
                     "completion": None if not end else {"path": journal_path, "sha256": sha(journal_raw), "line": end["index"] + 1, "event_sha256": end["event_sha256"]}})
    expected = source["resources"]["reset"]
    if len(rows) != expected["attempted"] or sum(r["completion"] is not None for r in rows) != expected["completed"]:
        raise ValueError("reset resource count mismatch")
    return {"schema": "codex_p1_original_reset_inventory_20261101.v1",
            "scope": "one pinned historical calibration sourcepack and its bound journal; source-recorded events",
            "events": rows, "recorded_reset_starts": len(rows),
            "recorded_reset_completions": sum(r["completion"] is not None for r in rows),
            "attempt_records": len(attempts), "journal_complete_claim": source.get("durable_journal_complete"),
            "unknown": ["other runs, failures, interrupted attempts and restarts outside this journal",
                        "A-H original reset attempts and execution closure",
                        "global historical coverage, physical independence, training exposure, and freshness"],
            "history_complete": False, "freshness_certified": False, "training_exclusion_certified": False,
            "P1_admitted": False, "new_environment_or_gpu_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raw = SOURCE.read_bytes()
    if sha(raw) != SOURCE_SHA:
        raise ValueError("sourcepack SHA drift")
    result = extract(raw, JOURNAL.read_bytes())
    data = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as stream:
        stream.write(data)
    print(json.dumps({"output": str(args.output.resolve()), "sha256": sha(data), "events": len(result["events"])}))


if __name__ == "__main__":
    main()
