"""Saved-only per-slot resource reconciliation; does not rerun an experiment."""
import argparse
import json
import os
from pathlib import Path

from experiments.spectral_reachability import codex_p3_six_physical_batch_saved_20261030_v3 as saved

PREP_SHA = "cb779e9a043d5fd8f94ca454e264872f8cc9c30fab0c92abb1153929a2819d8d"
REPORT_SHA = "c349d76b4374d8aa690a24d39c9fbdc97d6bbfacdb6f0334e78a6f38b632b6b0"
ROOT = saved.host.ROOT
PREP = ROOT / "results/codex_p3_six_physical_host_20261030_v2_PREP_01.json"
RUN = ROOT / "results/codex_p3_six_physical_host_20261030_v2_RUN_01"
BIND = saved.host.BIND


def reconcile(report, events, binding):
    require = saved.require
    rows = report["rows"]
    sources = binding["rows"]
    require(len(rows) == len(sources) == 6, "six-slot roster")
    names = report["counts"].keys()
    counts = [{k: {"attempted": 0, "completed": 0} for k in names} for _ in rows]
    for event in events:
        if event["kind"] not in ("operation_attempted", "operation_completed"):
            continue
        i = event.get("branch")
        require(type(i) is int and 0 <= i < 6, "resource branch")
        row = rows[i]
        require(event.get("root_slot") == row["root_slot"] and event.get("arm") == row["arm"],
                "resource identity")
        resource = event["resource"]
        require(resource in names, "resource name")
        counts[i][resource]["attempted" if event["kind"] == "operation_attempted" else "completed"] += 1
    result = []
    for i, (row, source, resources) in enumerate(zip(rows, sources, counts, strict=True)):
        require((row["root_slot"], row["arm"]) == (source["root"], source["policy"]) and
                row["selected_plan"]["raw_float32_sha256"] == source["raw_float32_sha256"],
                "source winner binding")
        steps = row["warmup_returned"] + row["own_returned"]
        expected = dict(environment_construct=1, environment_close=1, reset=2, set_state=4,
                        goal_render=2, goal_bind=1, public_step=steps, physics_step=4 + 10 * steps,
                        explicit_render=row["PNG"])
        require(set(expected) == set(resources) and all(
            resources[k] == {"attempted": v, "completed": v} for k, v in expected.items()),
            "per-slot physical census")
        result.append(dict(root=row["root_slot"], policy=row["arm"], physical=resources,
                           prior_neural_attempts=source["actual_resource_attempts"],
                           prior_neural_returns=source["actual_resource_returns"],
                           prior_metric_returns=source["actual_metric_returns"],
                           winner_raw_float32_sha256=source["raw_float32_sha256"],
                           status=row["status"], native_success=row["native_success"],
                           object_position_error=row["final_metrics"]["object_position_error"],
                           wrapped_angle_error=row["final_metrics"]["wrapped_object_angle_error"]))
    require(all(sum(x[k]["attempted"] for x in counts) == report["counts"][k]["attempted"] and
                sum(x[k]["completed"] for x in counts) == report["counts"][k]["completed"] for k in names),
            "aggregate reconciliation")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    saved.require(output.is_absolute() and output.parent == ROOT / "results" and
                  output.name.startswith("codex_p3_six_physical_resource_review_20261030_v1_"),
                  "exclusive result path")
    check = saved.consume(RUN, REPORT_SHA, PREP, PREP_SHA)
    report = saved.host.helper.read(RUN / "report.json")
    events = saved.kernel.journal(RUN / "journal.jsonl", report["journal_chain_tail"])
    binding = saved.host.helper.read(BIND)
    result = dict(status="saved_per_slot_resource_reconciled", prepare_sha256=PREP_SHA,
                  report_sha256=REPORT_SHA, original_consumer_status=check["status"],
                  rows=reconcile(report, events, binding), physical_totals=report["counts"],
                  prior_signal_accounting=binding["signal_accounting"],
                  prior_signal_allocation=binding["signal_allocation"],
                  physical_elapsed_seconds=report["elapsed_seconds"],
                  per_slot_wall_seconds="unknown", total_compute_fairness="not established",
                  scope="known roots, exploratory; resource axes nested, not additive FLOPs",
                  independent_physical_reexecution=False, scientific_pass=False)
    with output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    print(result["status"], len(result["rows"]))


if __name__ == "__main__":
    main()
