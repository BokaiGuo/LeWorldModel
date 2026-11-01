"""Verify saved physical events and online CUDA planning before subsequent dispatch."""
import argparse
import json
from pathlib import Path

from . import p1_response_midpoint_online_20261031 as online

ROOT, need, sha = online.ROOT, online.need, online.sha
NAME = "p1_response_midpoint_online_consume_20261031"
PREP = ROOT / "results/p1_response_midpoint_online_20261031_prepare_RUN01.json"
PREP_SHA = "86a81ddb5252ca181fd1f53e7b43a2cdae639256cf9a38c0b83217141a7269a0"


def check_events(events, item, expected):
    plans = [event for event in events if event["kind"] == "online_midpoint_plan"]
    captures = [event for event in events if event["kind"] == "fresh_observation"
                and event["receipt"]["phase"] == "probe" and event["receipt"]["step"] == 5]
    transitions = [event for event in events if event["kind"] == "returned_transition"
                   and event["phase"] == "probe" and event["step"] == 6]
    if expected is None:
        need(not plans and not transitions, "unexpected online plan")
        return
    need(len(plans) == 1 and len(captures) == 3 and len(transitions) <= 1, "online event roster")
    event = plans[0]
    need({k: event[k] for k in expected} == expected and event["anchor"] == item["index"]
         and event["midpoint_pixel_sha256"] == item["midpoint"]["pixel_sha256"]
         and event["second_packet_hex"] == item["raw10_float32_hex"][80:]
         and event["calls_code_path"] == {"encoder": 2, "predict": 22, "action_encoder": 22, "backward": 20}
         and event["baseline_score"] >= 0 and event["optimized_score"] >= 0
         and event["optimizer_seconds"] >= 0, "online plan identity")
    need(max(event["index"] for event in captures) < event["index"]
         and (not transitions or event["index"] < transitions[0]["index"]), "online event ordering")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--report-sha", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    directory, output = Path(args.directory), Path(args.output)
    need(directory.is_absolute() and directory.is_dir() and not directory.is_symlink()
         and output.is_absolute() and output.parent == ROOT / "results" and output.name.startswith(NAME + "_")
         and not output.exists() and not output.is_symlink(), "paths")
    need(sha(PREP) == PREP_SHA and sha(directory / "run_report.json") == args.report_sha, "input hashes")
    read = online.physical.prior.bundle.physical.prior.native.read
    prepared, report = read(PREP, cap=64 * 1024**2), read(directory / "run_report.json", cap=64 * 1024**2)
    need(prepared == online.build() and report["status"] == "recorded_not_consumed"
         and report["prepare_sha256"] == PREP_SHA and len(report["shards"]) == 13
         and report["host"]["device"] == "cuda:0"
         and report["model_before"] == report["model_after"] == online.source.common.base.MODEL_HASH,
         "report/source/model")
    events_by_anchor = {event["anchor"]: event for event in report["online_events"]}
    need(len(events_by_anchor) == len(report["online_events"]), "unique online events")
    rows = []
    for anchor, item, shard in zip(prepared["physical"]["anchors"], prepared["physical"]["roster"],
                                    report["shards"], strict=True):
        need(anchor["index"] == item["index"] == shard["index"]
             and shard["status"] == "recorded_not_consumed", "shard identity")
        path = directory / f"row_{anchor['index']:02d}"
        row = online.physical.prior.saved.check_row(path, anchor, item, shard["sha256"])
        row_report = read(path / "report.json", cap=64 * 1024**2)
        events = online.physical.prior.journal.events(path / "journal.jsonl", row_report["journal_chain_tail"])
        expected = events_by_anchor.pop(anchor["index"], None)
        need((expected is not None) == (len(row_report["row"]["responses"]) >= 5),
             "midpoint event required")
        check_events(events, item, expected)
        rows.append(row)
    need(not events_by_anchor and sha(PREP) == PREP_SHA
         and sha(directory / "run_report.json") == args.report_sha, "source drift")
    value = {"schema": NAME, "status": "saved_checks_passed", "rows": rows, "fixed_rows": 16,
             "online_event_count": len(report["online_events"]), "host": report["host"],
             "prepare_sha256": PREP_SHA, "run_report_sha256": args.report_sha,
             "P1_admission": False, "independent_confirmation": False, "equal_budget_certified": False}
    digest = online.physical.prior.bundle.physical.write(output, value)
    print(json.dumps({"status": value["status"], "rows": len(rows), "sha256": digest}), flush=True)


if __name__ == "__main__":
    main()
