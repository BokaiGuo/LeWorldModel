"""Verify every joint H10 shard and produce a paired descriptive table."""
import argparse
import json
from pathlib import Path

from . import p1_joint_h10_20261030 as experiment
from . import codex_p1_h10_paired_table_20261030_main as metric

run, saved, require, sha = experiment.run, experiment.saved, experiment.require, experiment.sha


def consume(directory, report_sha, prepared, prepare_sha):
    directory, prepared = Path(directory), Path(prepared)
    require(sha(prepared) == prepare_sha, "prepare SHA")
    p = experiment.joint.runner.native.read(prepared, cap=64*1024**2)
    require(p == experiment.build(), "prepare/source rebuild")
    path = directory / "run_report.json"
    require(sha(path) == report_sha, "run report SHA")
    report = run.native.read(path, cap=4*1024**2)
    require(report["schema"] == experiment.NAME + ".run" and report["status"] == "recorded_not_consumed"
            and report["fixed_branches"] == 1008 and len(report["shards"]) == 24
            and report["P1_admission"] is False, "complete roster")
    bindings = dict(p["bindings"], **{str(prepared): prepare_sha})
    expected = {"prepare": str(prepared), "prepare_sha256": prepare_sha,
                "bindings": bindings, "fixed_branches": 1008}
    require(run.native.read(directory / "run_binding.json", cap=64*1024**2) == expected, "run binding")
    checked = []
    for i, (anchor, shard) in enumerate(zip(p["anchors"], report["shards"], strict=True)):
        require(shard["anchor"] == i and shard["status"] == "recorded_not_consumed", "shard order")
        bank = p["probe_banks"][anchor["logical_pair"]]
        checked.append(saved.check_shard(directory / f"anchor_{i:02d}", anchor, bank, shard["sha256"]))
        print(json.dumps({"verified_anchor": i}), flush=True)
    rows = []
    for logical in range(8):
        group = p["anchors"][logical*3:logical*3+3]
        for j, probe in enumerate(p["probe_banks"][logical]):
            for k, anchor in enumerate(group):
                observed = checked[logical*3+k]
                endpoint = metric.physical_metrics(observed["endpoints"][j], anchor["goal"])
                zero = metric.physical_metrics(observed["endpoints"][0], anchor["goal"])
                root = metric.physical_metrics(anchor["root_state"], anchor["goal"])
                prefix = metric.physical_metrics(anchor["checkpoint"]["state"], anchor["goal"])
                rows.append({"logical_pair": logical, "root_slot": anchor["root_slot"],
                             "condition": anchor["condition"], "arm": anchor["arm"], "role": anchor["role"],
                             "probe": probe["name"], "probe_sha256": probe["raw10_sha256"],
                             "status": observed["statuses"][j], "root": root, "prefix": prefix,
                             "prefix_minus_root": metric.paired_difference(prefix, root),
                             "endpoint": endpoint, "endpoint_minus_own_zero": metric.paired_difference(endpoint, zero)})
    require(len(rows) == 1008 and all(sha(k) == v for k, v in bindings.items())
            and sha(path) == report_sha, "final source drift")
    return {"schema": "p1_joint_h10_consume_20261030", "run_report_sha256": report_sha,
            "prepare_sha256": prepare_sha, "rows": rows, "fixed_anchors": 24, "fixed_branches": 1008,
            "P1_admission": False, "independent_confirmation": False, "equal_budget_certified": False,
            "limitation": "known roots; common suffix is not independent suffix planning; no cache serialization"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--directory", required=True)
    p.add_argument("--report-sha256", required=True)
    p.add_argument("--prepared", required=True)
    p.add_argument("--prepare-sha256", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    require(out.is_absolute() and out.parent == experiment.ROOT / "results"
            and out.name.startswith("p1_joint_h10_consume_") and not out.exists(), "exclusive output")
    result = consume(a.directory, a.report_sha256, a.prepared, a.prepare_sha256)
    result["consumer_sha256"] = sha(Path(__file__))
    result["metric_sha256"] = sha(Path(metric.__file__))
    print(json.dumps({"sha256": run._write_new(out, result), "rows": len(result["rows"])}))


if __name__ == "__main__":
    main()
