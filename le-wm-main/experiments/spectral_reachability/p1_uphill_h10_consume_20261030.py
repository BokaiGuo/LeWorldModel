"""Verify the entire fixed uphill H10 panel before emitting descriptive rows."""
import argparse
import json
from pathlib import Path

from . import p1_uphill_h10_20261030 as experiment
from . import codex_p1_h10_paired_table_20261030_main as metric

run, saved, require, sha = experiment.run, experiment.saved, experiment.require, experiment.sha
PREP = experiment.ROOT / "results/p1_uphill_h10_20261030_prepare_RUN01.json"
PREP_SHA = "890e38ecf7758ae739c25e9d355533e4d3b31be91a8c063c9625d1886c77d56d"


def consume(directory, report_sha):
    directory = Path(directory)
    require(sha(PREP) == PREP_SHA, "prepare SHA")
    p = experiment.uphill.runner.native.read(PREP, cap=64 * 1024**2)
    require(p == experiment.build(), "prepare/source rebuild")
    path = directory / "run_report.json"
    require(sha(path) == report_sha, "run report SHA")
    report = run.native.read(path, cap=4 * 1024**2)
    require(report["schema"] == experiment.NAME + ".run"
            and report["status"] == "recorded_not_consumed"
            and report["fixed_branches"] == 1008 and len(report["shards"]) == 24
            and report["P1_admission"] is False, "complete roster")
    bindings = dict(p["bindings"], **{str(PREP): PREP_SHA})
    expected = {"prepare": str(PREP), "prepare_sha256": PREP_SHA,
                "bindings": bindings, "fixed_branches": 1008}
    require(run.native.read(directory / "run_binding.json", cap=64 * 1024**2) == expected,
            "run binding")
    checked = []
    for i, (anchor, shard) in enumerate(zip(p["anchors"], report["shards"], strict=True)):
        require(shard["anchor"] == i and shard["status"] == "recorded_not_consumed", "shard order")
        bank = p["probe_banks"][anchor["logical_pair"]]
        target = directory / f"anchor_{i:02d}"
        observed = saved.check_shard(target, anchor, bank, shard["sha256"])
        report_row = run.native.read(target / "report.json", cap=64 * 1024**2)
        require(sha(target / "report.json") == shard["sha256"]
                and len(report_row["rows"]) == 42, "shard count/hash")
        checked.append((observed, report_row["rows"]))
        print(json.dumps({"verified_anchor": i}), flush=True)
    rows = []
    for logical in range(8):
        for j, probe in enumerate(p["probe_banks"][logical]):
            for k, anchor in enumerate(p["anchors"][logical * 3:logical * 3 + 3]):
                observed, raw_rows = checked[logical * 3 + k]
                raw = raw_rows[j]
                require(raw["probe"] == probe["name"] and raw["index"] == j
                        and raw["status"] == observed["statuses"][j], "branch identity")
                returned = raw["own_returned"] - 5
                require(0 <= returned <= 10, "returned suffix count")
                endpoint = metric.physical_metrics(observed["endpoints"][j], anchor["goal"])
                last_response = raw["responses"][-1] if raw["responses"] else None
                last_accepted = metric.physical_metrics(
                    last_response["state"] if last_response is not None else None, anchor["goal"])
                zero = metric.physical_metrics(observed["endpoints"][0], anchor["goal"])
                root = metric.physical_metrics(anchor["root_state"], anchor["goal"])
                prefix = metric.physical_metrics(anchor["checkpoint"]["state"], anchor["goal"])
                rows.append({"logical_pair": logical, "root_slot": anchor["root_slot"],
                             "physical_root_sha256": anchor["physical_root_sha256"],
                             "condition": anchor["condition"], "arm": anchor["arm"], "role": anchor["role"],
                             "task_id": anchor["task_id"], "probe": probe["name"],
                             "probe_sha256": probe["raw10_sha256"], "status": observed["statuses"][j],
                             "returned_suffix_steps": returned, "root": root, "prefix": prefix,
                             "prefix_minus_root": metric.paired_difference(prefix, root),
                             "last_accepted_metrics": last_accepted,
                             "last_accepted_terminated": None if last_response is None else last_response["terminated"],
                             "last_accepted_truncated": None if last_response is None else last_response["truncated"],
                             "endpoint": endpoint, "endpoint_minus_own_zero": metric.paired_difference(endpoint, zero)})
    require(len(rows) == 1008 and all(sha(k) == v for k, v in bindings.items())
            and sha(path) == report_sha, "final source drift")
    return {"schema": "p1_uphill_h10_consume_20261030", "run_report_sha256": report_sha,
            "prepare_sha256": PREP_SHA, "rows": rows, "fixed_anchors": 24, "fixed_branches": 1008,
            "P1_admission": False, "independent_confirmation": False, "equal_budget_certified": False,
            "limitation": "Known roots; common suffix effects are not independent planning capacity."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    require(out.is_absolute() and out.parent == experiment.ROOT / "results"
            and out.name.startswith("p1_uphill_h10_consume_") and not out.exists()
            and not out.is_symlink(), "exclusive output")
    result = consume(args.directory, args.report_sha256)
    result["consumer_sha256"] = sha(Path(__file__))
    result["metric_sha256"] = sha(Path(metric.__file__))
    print(json.dumps({"sha256": run._write_new(out, result), "rows": len(result["rows"])}))


if __name__ == "__main__":
    main()
