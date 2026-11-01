"""Export verified H10 branches with prefix and independently planned outcomes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from . import p1_joint_complete_pairing_20261030 as base


def flatten(pairing, shard_rows):
    anchors = {a["index"]: a for a in pairing["anchors"]}
    base.require(len(anchors) == 24 and len(shard_rows) == 24, "anchor roster")
    zero_anchors = {a["logical_pair"]: a for a in anchors.values() if a["role"] == "zero"}
    base.require(len(zero_anchors) == 8, "zero anchor roster")
    result = []
    for pair in pairing["common_suffix_pairs"]:
        for role, item in pair["roles"].items():
            anchor = anchors[item["anchor_index"]]
            source = shard_rows[anchor["index"]][pair["probe"]]
            base.require(source["status"] == item["status"], "branch status")
            returned = source["own_returned"] - 5
            base.require(0 <= returned <= 10, "suffix returned count")
            if item["status"] == "probe_completed":
                base.require(returned == 10, "completed suffix length")
            independent = anchor["independent_suffix_20step"]
            row = {
                "logical_pair": pair["logical_pair"], "root_slot": anchor["root_slot"],
                "physical_root_sha256": anchor["physical_root_sha256"],
                "condition": anchor["condition"], "arm": anchor["arm"], "role": role,
                "probe": pair["probe"], "probe_sha256": pair["probe_sha256"],
                "common_status": item["status"], "common_returned_suffix_steps": returned,
                "independent_status": independent["status"],
                "independent_returned_suffix_steps": independent["returned_suffix"],
                "prefix_native_success_before": anchor["root_metrics"]["native_success"],
                "prefix_native_success_after": anchor["prefix_metrics"]["native_success"],
            }
            metrics = {
                "prefix_change": anchor["prefix_minus_root"],
                "common_endpoint": item["endpoint"],
                "common_minus_own_zero": item["endpoint_minus_own_zero_suffix"],
                "common_minus_zero_prefix": item["same_suffix_endpoint_minus_zero_prefix"],
                "independent_endpoint": independent["endpoint"],
                "independent_minus_prefix": base.difference(independent["endpoint"], anchor["prefix_metrics"]),
                "independent_minus_zero_prefix": base.difference(
                    independent["endpoint"],
                    zero_anchors[pair["logical_pair"]]["independent_suffix_20step"]["endpoint"]),
            }
            for group, values in metrics.items():
                for name in base.NUMERIC:
                    row[group + "_" + name] = None if values is None else values[name]
            row["common_endpoint_native_success"] = (
                None if item["endpoint"] is None else item["endpoint"]["native_success"])
            row["independent_endpoint_native_success"] = (
                None if independent["endpoint"] is None else independent["endpoint"]["native_success"])
            result.append(row)
    base.require(len(result) == 1008, "branch roster")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    out = args.output
    base.require(out.is_absolute() and out.parent == base.ROOT / "results"
                 and out.name.startswith("p1_joint_flat_pairing_") and out.suffix == ".csv"
                 and not out.exists() and not out.is_symlink(), "exclusive output")
    prepared = base.read_checked(base.PREP_PATH, base.PREP_SHA)
    anchors = base.read_checked(base.ANCHOR_PATH, base.ANCHOR_SHA)
    from . import p1_joint_h10_consume_20261030 as consumer
    panel = consumer.consume(args.directory, args.report_sha256, base.PREP_PATH, base.PREP_SHA)
    pairing = base.join(prepared, anchors, panel)
    report = base.read_checked(args.directory / "run_report.json", args.report_sha256)
    shard_rows = {}
    for shard in report["shards"]:
        i = shard["anchor"]
        data = base.read_checked(args.directory / f"anchor_{i:02d}" / "report.json", shard["sha256"])
        rows = {r["probe"]: r for r in data["rows"]}
        base.require(len(rows) == 42, "shard probe roster")
        shard_rows[i] = rows
    rows = flatten(pairing, shard_rows)
    with out.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output": str(out), "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                      "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      "run_report_sha256": args.report_sha256,
                      "rows": len(rows), "independent_plans": 24,
                      "warning": "Independent plan values repeat across probes; do not count as independent samples."}))


if __name__ == "__main__":
    main()
