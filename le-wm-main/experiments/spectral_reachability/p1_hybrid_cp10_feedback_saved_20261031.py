"""Verify feedback physical rows and the pre-dispatch midpoint identity gate."""
import argparse
import json
from pathlib import Path

from . import p1_hybrid_cp10_feedback_physical_20261031 as run
from . import p1_hybrid_cp5_physical_saved_20261031 as saved


def verify_midpoint_gate(events, midpoint, trace):
    verified = [e for e in events if e["kind"] == "midpoint_refresh_verified"]
    steps = [e for e in events if e["kind"] == "operation_dispatch_intent"
             and e["receipt"].get("resource") == "public_step"
             and e["receipt"].get("phase") == "probe" and e["receipt"].get("step") == 6]
    frames = [e["receipt"] for e in events if e["kind"] == "fresh_observation"
              and e["receipt"]["phase"] == "probe" and 1 <= e["receipt"]["step"] <= 5
              and e["receipt"]["overlay_index"] == 0]
    run.need(len(verified) == len(steps) == 1 and verified[0]["index"] < steps[0]["index"]
             and verified[0]["checkpoint"] == midpoint and verified[0]["trace"] == trace
             and len(frames) == len(trace) == 5
             and all(all(frame[k] == reference[k] for k in reference)
                     for frame, reference in zip(frames, trace, strict=True)),
             "first packet/feedback gate")


def consume(directory, report_sha, prepared, prepare_sha):
    run.need(run.sha(prepared) == prepare_sha, "prepare SHA")
    fixed = run.old.h.read(prepared)
    run.need(fixed == run.build(), "reconstructed prepare")
    directory = Path(directory)
    path = directory / "run_report.json"
    run.need(directory.is_absolute() and directory.resolve() == directory
             and run.sha(path) == report_sha, "run report SHA")
    report = run.old.h.read(path)
    run.need(report["schema"] == run.NAME + ".run" and report["status"] == "recorded_not_consumed"
             and report["prepare_sha256"] == prepare_sha and len(report["shards"]) == 2
             and report["P1_admission"] is False, "complete run")
    output = []
    for anchor, item, midpoint, trace, shard in zip(
            fixed["anchors"], fixed["items"], fixed["midpoints"], fixed["probe_traces"],
            report["shards"], strict=True):
        index = anchor["index"]
        run.need(shard["index"] == index and shard["status"] == "recorded_not_consumed", "shard identity")
        folder = directory / f"row_{index:02d}"
        row = saved.check_row(folder, anchor, item, shard["sha256"])
        value = run.old.h.read(folder / "report.json")
        journal = folder / "journal.jsonl"
        run.need(run.sha(journal) == value["journal_sha256"], "journal hash")
        events = saved.saved.events(journal, value["journal_chain_tail"])
        verify_midpoint_gate(events, midpoint, trace)
        output.append(row)
    run.need(all(run.sha(p) == d for p, d in fixed["bindings"].items())
             and run.sha(path) == report_sha, "source drift")
    return {"schema": "p1_hybrid_cp10_feedback_saved_20261031", "status": "saved_checks_passed",
            "rows": output, "run_report_sha256": report_sha, "prepare_sha256": prepare_sha,
            "P1_admission": False, "independent_confirmation": False,
            "limitations": "Known roots, extra computation; saved verification not OS isolation."}


def main():
    parser = argparse.ArgumentParser()
    for option in ("directory", "report-sha", "prepared", "prepare-sha", "output"):
        parser.add_argument("--" + option, required=True)
    args = parser.parse_args()
    output = Path(args.output)
    run.need(output.is_absolute() and output.parent == run.ROOT / "results"
             and output.name.startswith("p1_hybrid_cp10_feedback_saved_20261031_")
             and not output.exists(), "exclusive output")
    result = consume(args.directory, args.report_sha, args.prepared, args.prepare_sha)
    print(json.dumps({"status": result["status"], "sha256": run.old.h.engine._write_new(output, result)}))


if __name__ == "__main__":
    main()
