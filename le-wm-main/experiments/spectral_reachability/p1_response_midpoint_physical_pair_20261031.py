"""Pair verified midpoint-replanned physical outcomes with their same-action-prefix controls."""
import argparse
import json
from pathlib import Path

from . import p1_response_midpoint_physical_20261031 as experiment
from . import codex_p1_h10_paired_table_20261030_main as metrics

ROOT, need, sha = experiment.ROOT, experiment.need, experiment.sha
NAME = "p1_response_midpoint_physical_pair_20261031"
PREP = ROOT / "results/p1_response_midpoint_physical_20261031_prepare_RUN01.json"
PREP_SHA = "3d70d2feb2a05ff8241c63a4c00cf3a5d4f5a73d6e5450f5a4f5f2bf37541cf4"


def pair(prepared, original, current):
    need(prepared["fixed_rows"] == current["fixed_rows"] == 16
         and current["status"] == original["status"] == "saved_checks_passed"
         and current["excluded"] == prepared["excluded"]
         and len(prepared["anchors"]) == len(prepared["roster"]) == len(current["rows"]) == 13,
         "fixed roster")
    prior = {row["index"]: row for row in original["rows"]}
    need(len(prior) == 14, "original roster")
    rows = []
    for anchor, item, result in zip(prepared["anchors"], prepared["roster"], current["rows"], strict=True):
        old = prior[item["index"]]
        need(all(anchor[key] == item[key] == old[key] == result[key] for key in
                 ("index", "role", "root_slot", "condition", "arm")), "identity")
        need(old["status"] == "probe_completed" and old["completed_endpoint"] is not None,
             "original endpoint")
        need(result["status"] in ("probe_completed", "native_stopped", "censored_out_of_domain")
             and ((result["completed_endpoint"] is not None) == (result["status"] == "probe_completed")),
             "new endpoint and censor")
        original_end = metrics.physical_metrics(old["completed_endpoint"], anchor["goal"])
        current_end = metrics.physical_metrics(result["completed_endpoint"], anchor["goal"])
        rows.append({
            **{key: item[key] for key in ("index", "logical_pair", "root_slot", "condition", "arm", "role")},
            "prefix_metrics": metrics.physical_metrics(anchor["checkpoint"]["state"], anchor["goal"]),
            "baseline_status": old["status"], "baseline_endpoint": original_end,
            "midpoint_replanned_status": result["status"],
            "midpoint_replanned_returned_suffix": result["returned_suffix"],
            "midpoint_replanned_endpoint": current_end,
            "midpoint_replanned_minus_baseline": metrics.paired_difference(current_end, original_end),
            "last_accepted_state_metrics": result["last_accepted_state_metrics"],
        })
    return {"schema": NAME, "rows": rows, "excluded": prepared["excluded"], "fixed_rows": 16,
            "P1_admission": False, "independent_confirmation": False, "equal_budget_certified": False,
            "limitations": "Known roots, post-outcome offline midpoint replanning, extra model compute; "
                           "not online control or an independent confirmation. Early stops/censors do "
                           "not supply fixed-horizon endpoints."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("directory", "report-sha", "consumed", "consume-sha", "output"):
        parser.add_argument("--" + option, required=True)
    args = parser.parse_args()
    directory, consumed, output = Path(args.directory), Path(args.consumed), Path(args.output)
    need(output.is_absolute() and output.parent == ROOT / "results" and output.name.startswith(NAME + "_")
         and not output.exists() and not output.is_symlink(), "exclusive output")
    sources = ((PREP, PREP_SHA), (experiment.source.RECEIPT, experiment.source.RECEIPT_SHA),
               (directory / "run_report.json", args.report_sha), (consumed, args.consume_sha))
    need(all(sha(path) == digest for path, digest in sources), "source hash")
    read = experiment.prior.bundle.physical.prior.native.read
    prepared, original, report, receipt = [read(path, cap=64 * 1024**2) for path, _ in sources]
    need(prepared == experiment.build() and report["status"] == "recorded_not_consumed"
         and report["prepare_sha256"] == PREP_SHA and len(report["shards"]) == 13,
         "report binding")
    for item, saved, shard in zip(prepared["roster"], receipt["rows"], report["shards"], strict=True):
        path = directory / f"row_{item['index']:02d}" / "report.json"
        need(item["index"] == saved["index"] == shard["index"] and sha(path) == shard["sha256"],
             "shard binding")
        record = read(path, cap=64 * 1024**2)["row"]
        need(record["status"] == saved["status"] and
             (record["responses"][-1]["state"] if record["status"] == "probe_completed" else None)
             == saved["completed_endpoint"], "consumer binding")
    result = pair(prepared, original, receipt)
    need(all(sha(path) == digest for path, digest in sources), "source drift")
    need(all(sha(directory / f"row_{s['index']:02d}" / "report.json") == s["sha256"]
             for s in report["shards"]), "shard drift")
    result["source_sha256"] = {str(path): digest for path, digest in sources}
    result["analysis_sha256"] = sha(__file__)
    digest = experiment.prior.bundle.physical.write(output, result)
    print(json.dumps({"rows": len(result["rows"]), "sha256": digest}), flush=True)


if __name__ == "__main__":
    main()
