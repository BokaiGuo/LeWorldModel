"""Verify and summarize all fixed uphill independently optimized physical suffixes."""
import argparse
import json
from pathlib import Path

from . import p1_uphill_cp5_physical20_20261030 as physical
from . import p1_joint_anchor_table_20261030 as table
from . import p1_joint_complete_pairing_20261030 as pairing


def analyze(prepared, plan, observed, events):
    result = table.build(prepared, plan, observed)
    by_group = {}
    for row in result["rows"]:
        key = (row["logical_pair"], row["role"])
        table.require(key not in by_group, "duplicate role")
        by_group[key] = row
    for row, observed_row, event in zip(result["rows"], observed["rows"], events, strict=True):
        table.require(row["index"] == observed_row["index"] == event["index"], "cost identity")
        own = row["independent_suffix_20step"]["endpoint"]
        zero = by_group[(row["logical_pair"], "zero")]["independent_suffix_20step"]["endpoint"]
        greedy = by_group[(row["logical_pair"], "greedy")]["independent_suffix_20step"]["endpoint"]
        row["independent_suffix_20step"]["endpoint_minus_zero_prefix"] = pairing.difference(own, zero)
        row["independent_suffix_20step"]["endpoint_minus_greedy_prefix"] = pairing.difference(own, greedy)
        row["independent_suffix_20step"]["last_accepted_state_metrics"] = observed_row["last_accepted_state_metrics"]
        ledger = event["result"]["ledger_receipt"]
        row["planning_resource_receipt"] = {
            "encoding_attempted": ledger["encoding"]["attempted"],
            "planning_charged_upper_bounds": ledger["planning"]["charged_upper_bounds"],
            "planning_completed_events": ledger["planning"]["completed_events"],
            "actual_usage_verified": ledger["planning"]["actual_usage_verified"],
            "coverage_verified": ledger["planning"]["coverage_verified"],
        }
        row.pop("common_suffix")
        row.pop("common_suffix_status")
    result.update(schema="p1_uphill_cp5_analysis_20261030", fixed_anchors=24,
                  limitations="Known roots and duplicated trajectories; planning charges are upper bounds, "
                              "not certified actual total cost; native early stop is not a completed endpoint.")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--prepare-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    table.require(output.is_absolute() and output.parent == physical.ROOT / "results"
                  and output.name.startswith("p1_uphill_cp5_analysis_")
                  and not output.exists() and not output.is_symlink(), "exclusive output")
    observed = physical.consume(args.directory, args.report_sha256, args.prepared, args.prepare_sha256)
    # consume() has already rebuilt and checked this exact prepared file.
    # Rebuilding a second time in-process mutates legacy module globals and
    # can make the upstream contract check fail despite identical inputs.
    table.require(physical.sha(args.prepared) == args.prepare_sha256, "prepare drift")
    plan = physical.design.uphill.runner.native.read(args.prepared, cap=64 * 1024**2)
    prepared = physical.design.uphill.runner.native.read(physical.bundle.PREP, cap=64 * 1024**2)
    events = []
    for row in plan["roster"]:
        path = physical.PLAN_DIR / f"row_{row['index']:02d}_result.json"
        table.require(physical.sha(path) == plan["bindings"][str(path)], "planning row hash")
        events.append(json.loads(path.read_bytes()))
    value = analyze(prepared, plan, observed, events)
    value.update(physical_report_sha256=args.report_sha256, physical_prepare_sha256=args.prepare_sha256,
                 analysis_sha256=physical.sha(Path(__file__)))
    print(json.dumps({"sha256": physical.design.run._write_new(output, value), "rows": len(value["rows"])}))


if __name__ == "__main__":
    main()
