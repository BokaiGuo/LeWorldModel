"""Read-only verification of development random-reset receipts."""
import argparse
import json
from pathlib import Path

from . import p1_dev_reset_bound_20261101_codex as producer

q = producer.queue


def verify(prepared, expected_prepare, directory, expected_report):
    require = q.require
    prepared, directory = Path(prepared), Path(directory)
    require(producer.sha(prepared) == expected_prepare, "prepare SHA")
    plan = q.read_json(prepared)
    require(plan == producer.contract(), "source and prepare drift")
    report_path, journal = directory / "report.json", directory / "journal.jsonl"
    require(producer.sha(report_path) == expected_report, "report SHA")
    report = q.read_json(report_path)
    require(report["prepared_sha256"] == expected_prepare and producer.sha(journal) == report["journal_sha256"],
            "report bindings")
    raw = journal.read_bytes()
    require(raw.endswith(b"\n") and len(raw) <= q.MAX_BYTES, "journal size")
    events, previous = [json.loads(line) for line in raw.splitlines()], None
    for index, event in enumerate(events):
        body = {k: v for k, v in event.items() if k != "sha256"}
        require(event["index"] == index and event["previous"] == previous
                and event["sha256"] == q.digest(body), "journal chain")
        previous = event["sha256"]
    require(events[0]["kind"] == "run_started" and events[0]["prepare_sha256"] == expected_prepare
            and events[-1]["kind"] == "run_finished" and previous == report["journal_tail"], "run boundary")
    pos = 1
    require(len(report["rows"]) == len(plan["seeds"]), "full roster")
    for index, (row, seed) in enumerate(zip(report["rows"], plan["seeds"], strict=True)):
        require(row["index"] == index and row["seed"] == seed, "roster identity")
        if row["status"] == "not_started":
            continue
        attempt, returned = events[pos:pos+2]
        require(attempt["kind"] == "reset_attempt" and attempt["index_in_roster"] == index
                and attempt["seed"] == seed and returned["index_in_roster"] == index
                and returned["seed"] == seed, "reset dispatch")
        if row["status"] == "returned":
            receipt = row["receipt"]
            require(returned["kind"] == "reset_return" and returned["receipt"] == receipt
                    and receipt["state"] == receipt["snapshot"]["state"]
                    and all(receipt[key + "_sha256"] == q.digest(receipt[key])
                            for key in ("state", "goal", "snapshot"))
                    and receipt["history_sha256"] is None, "reset receipt")
        else:
            require(row["status"] == "failed" and returned["kind"] == "reset_failed"
                    and row["error_class"] == returned["error_class"], "failure receipt")
        pos += 2
    require(pos == len(events)-1, "unexpected events")
    return dict(schema=producer.NAME + ".saved_audit.v1", status="saved_integrity_not_independent_replay",
                source_signature_count=len(plan["source_sha256"]), rows=len(report["rows"]),
                returned=sum(r["status"] == "returned" for r in report["rows"]),
                errors=report["errors"], report_sha256=expected_report,
                training_membership="unknown", historical_coverage="unknown", freshness="unknown",
                independent_confirmation=False, P1_admission=False)


def main():
    parser = argparse.ArgumentParser()
    for name in ("prepared", "prepare-sha", "directory", "report-sha", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    value = verify(args.prepared, args.prepare_sha, args.directory, args.report_sha)
    print(q.write_new(args.output, value))


if __name__ == "__main__":
    main()
