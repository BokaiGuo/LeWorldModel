"""Join saved reset diagnostic replays to pinned original operations, CPU only."""
import argparse
import json
from pathlib import Path

from experiments.spectral_reachability import codex_p1_independent_root_inputs_20261101_exclusive as prior
from experiments.spectral_reachability import codex_p1_original_reset_inventory_20261101 as original

NAME = "p1_source_reset_goal_audit_20261027_v1_main01"
REPORT_SHA = "2f6b7c4ad8b54a3157f1ac2f514698b96a5cce1550a6c81a4f57d206df7ebe24"
FIELDS = ("candidate_id", "proposal_index", "attempt_index")


def join(report_raw, journal_raw, baseline, report_path, journal_path):
    if original.sha(report_raw) != REPORT_SHA:
        raise ValueError("report hash mismatch")
    report = original.parse(report_raw)
    if len(journal_raw) > 8 * 1024 * 1024 or not journal_raw.endswith(b"\n"):
        raise ValueError("journal limit or truncation")
    if original.sha(journal_raw) != report.get("journal_sha256"):
        raise ValueError("journal binding")
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
    if (events[0].get("prepare") != report.get("prepare")
            or report["prepare"].get("report_sha256") != original.SOURCE_SHA
            or events[-1].get("counts") != report.get("counts")
            or events[-1].get("status") != report.get("status")):
        raise ValueError("report or parent binding")
    originals = {}
    for row in baseline:
        if row.get("identity_status") != "original_attempt_not_independence_evidence":
            continue
        key = tuple(row["coordinates"][field] for field in FIELDS) + (row["operation"],)
        if key in originals:
            raise ValueError("ambiguous original operation")
        originals[key] = row
    reported = {}
    for row in report["rows"]:
        for phase in ("random_reset", "placement_reset"):
            key = tuple(row[field] for field in FIELDS) + (phase,)
            if key in reported:
                raise ValueError("ambiguous report coordinates")
            reported[key] = row[phase + "_seed"]
    pending, result = {}, []
    for event in events:
        if event.get("kind") not in ("reset_started", "reset_returned"):
            continue
        coord = event["coordinates"]
        key = tuple(coord[field] for field in FIELDS) + (coord["reset_phase"],)
        if event["kind"] == "reset_started":
            if key in pending or key not in originals or key not in reported:
                raise ValueError("unmatched or ambiguous replay")
            source = originals[key]
            if source["seed"] != reported[key]:
                raise ValueError("seed mismatch")
            row = {"coordinates": coord, "seed": reported[key], "start_line": event["index"] + 1,
                   "start_event_sha256": event["event_sha256"], "completion_line": None,
                   "completion_event_sha256": None,
                   "original_start": source["start"], "original_source_attempt": source["source_attempt"],
                   "relationship": "same_source_attempt_and_reset_phase_diagnostic_replay",
                   "physical_root_identity": "unknown", "independence": "unknown"}
            pending[key] = row
            result.append(row)
        else:
            if key not in pending:
                raise ValueError("unpaired replay completion")
            row = pending.pop(key)
            row["completion_line"] = event["index"] + 1
            row["completion_event_sha256"] = event["event_sha256"]
    counts = report["counts"]["reset"]
    if len(result) != counts["attempted"] or sum(r["completion_line"] is not None for r in result) != counts["completed"]:
        raise ValueError("reset count mismatch")
    return {"report_path": str(report_path), "report_sha256": original.sha(report_raw),
            "journal_path": str(journal_path), "journal_sha256": original.sha(journal_raw),
            "parent_sourcepack_sha256": original.SOURCE_SHA, "events": result}


def build(root=original.ROOT):
    root = Path(root)
    before = prior.build(root)
    path = root / "results" / NAME
    failures = list(before["unknown_sources"])
    diagnostic = None
    try:
        diagnostic = join((path / "report.json").read_bytes(), (path / "journal.jsonl").read_bytes(),
                          before["baseline"]["events"], path / "report.json", path / "journal.jsonl")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        failures.append({"source": str(path), "status": "unknown", "reason": str(exc)})
    return {"schema": "codex_p1_root_replay_join_20261101_exclusive.v1",
            "prior": before, "diagnostic_replay": diagnostic, "unknown_sources": failures,
            "global_history_coverage": "unknown", "physical_root_deduplication": "unknown",
            "training_exclusion": "unknown", "freshness": "unknown",
            "independent_roots_certified": False, "P1_admitted": False, "new_environment_or_gpu_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = build()
    data = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as stream:
        stream.write(data)
    print(json.dumps({"output": str(args.output), "sha256": original.sha(data),
                      "diagnostic_replays": len(result["diagnostic_replay"]["events"]) if result["diagnostic_replay"] else 0,
                      "unknown_sources": len(result["unknown_sources"])}))


if __name__ == "__main__":
    main()
