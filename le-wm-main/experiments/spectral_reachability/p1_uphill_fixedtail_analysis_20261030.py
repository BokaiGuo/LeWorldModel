"""Reverify and summarize the fixed original-tail physical intervention."""
import argparse
import json
from pathlib import Path

from . import p1_uphill_fixedtail_physical_20261030 as fixed
from . import p1_joint_anchor_table_20261030 as table
from . import p1_joint_complete_pairing_20261030 as pairing

physical = fixed.physical


def analyze(prepared, plan, observed):
    value = table.build(prepared, plan, observed)
    groups = {(x["logical_pair"], x["role"]): x for x in value["rows"]}
    table.require(len(groups) == 24, "role roster")
    zero_endpoints = {logical: groups[(logical, "zero")]["independent_suffix_20step"]["endpoint"]
                      for logical in range(8)}
    for row, observation in zip(value["rows"], observed["rows"], strict=True):
        item = row.pop("independent_suffix_20step")
        item["last_accepted_state_metrics"] = observation["last_accepted_state_metrics"]
        item["same_suffix_endpoint_minus_zero_prefix"] = pairing.difference(
            item["endpoint"], zero_endpoints[row["logical_pair"]])
        row["fixed_original_suffix"] = item
        row.pop("common_suffix")
        row.pop("common_suffix_status")
    value.update(schema="p1_uphill_fixedtail_analysis_20261030",
                 limitations="Known roots and duplicate arms; same suffix is not separately optimized capability.")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--prepare-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output
    table.require(output.is_absolute() and output.parent == physical.ROOT / "results"
                  and output.name.startswith("p1_uphill_fixedtail_analysis_")
                  and not output.exists() and not output.is_symlink(), "exclusive output")
    physical.NAME = fixed.NAME
    physical.configure, physical.build = fixed.configure, fixed.build
    observed = physical.consume(args.directory, args.report_sha256, args.prepared, args.prepare_sha256)
    table.require(physical.sha(args.prepared) == args.prepare_sha256, "prepare drift")
    plan = physical.design.uphill.runner.native.read(args.prepared, cap=64 * 1024**2)
    prepared = physical.design.uphill.runner.native.read(physical.bundle.PREP, cap=64 * 1024**2)
    result = analyze(prepared, plan, observed)
    result.update(physical_report_sha256=args.report_sha256, physical_prepare_sha256=args.prepare_sha256,
                  analysis_sha256=physical.sha(Path(__file__)))
    print(json.dumps({"sha256": physical.design.run._write_new(output, result), "rows": len(result["rows"])}))


if __name__ == "__main__":
    main()
