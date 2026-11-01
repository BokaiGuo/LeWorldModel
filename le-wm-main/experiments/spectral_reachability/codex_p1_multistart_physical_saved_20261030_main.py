"""Consume all fixed multi-start physical rows with the existing strict journal verifier."""
import argparse
import json
from pathlib import Path

from . import codex_p1_multistart_physical_20261030_main as run
from . import codex_p1_cp5_physical_saved_20261030_v1 as saved
from . import codex_p1_h10_paired_table_20261030_main as metrics

NAME = "codex_p1_multistart_physical_saved_20261030_main"
DIR = run.old.ROOT / "results/codex_p1_multistart_physical_20261030_main_RUN01"
REPORT_SHA = "7b56cea4858893ceae88189c4d1c46030d0a4b15c178014ffd126ff5490b7d34"


def consume():
    report, _ = run.gap.read(DIR/"run_report.json", REPORT_SHA)
    prepared, _ = run.gap.read(DIR/"prepare.json", report["prepare_sha256"])
    run.need(prepared == run.build() and report["status"] == "recorded_not_consumed"
             and report["fixed_rows"] == len(report["shards"]) == 24, "complete identity")
    anchors = run.old.h10.native.read(run.old.h10.INPUT, cap=64*1024**2)["anchors"]
    rows = []
    for i, (anchor, item, shard) in enumerate(zip(anchors, prepared["roster"], report["shards"], strict=True)):
        run.need(i == shard["index"] == item["index"] and shard["status"] == "recorded_not_consumed",
                 "shard identity")
        row = saved.check_row(DIR/f"row_{i:02d}", anchor, item, shard["report_sha256"])
        row["logical_pair"] = item["logical_pair"]
        row["raw10_sha256"] = item["raw10_sha256"]
        row["endpoint_metrics"] = metrics.physical_metrics(row["completed_endpoint"], anchor["goal"])
        rows.append(row)
    run.need(run.old.sha(DIR/"run_report.json") == REPORT_SHA, "report drift")
    return {"schema": NAME, "status": "saved_checks_passed", "fixed_rows": 24, "rows": rows,
            "report_sha256": REPORT_SHA, "prepare_sha256": report["prepare_sha256"],
            "P1_admission": False, "fairness_proven": False, "independent_confirmation": False,
            "limitations": "known roots and repeated trajectories; model call counts are not independently "
                           "hook-metered; no independent confirmation or equal total budget claim"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    run.need(output.is_absolute() and output.parent == run.old.ROOT/"results"
             and output.name.startswith(NAME+"_") and not output.exists() and not output.is_symlink(),
             "exclusive output")
    value = consume()
    print(json.dumps({"status": value["status"], "sha256": run.old.h10._write_new(output, value),
                      "rows": len(value["rows"])}), flush=True)
