"""Export a sealed complete P1 pairing as a long-form, censored CSV table."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

from . import p1_uphill_complete_pairing_20261030 as pairing

METRICS = pairing.FIELDS


def flatten(result):
    pairing.require(result["schema"] == "p1_uphill_complete_pairing_20261030"
                    and result["P1_admission"] is False
                    and result["independent_confirmation"] is False
                    and result["equal_budget_certified"] is False, "pairing contract")
    anchors = {a["index"]: a for a in result["anchors"]}
    pairing.require(len(result["anchors"]) == len(anchors) == 24
                    and len(result["common_suffix_pairs"]) == 336,
                    "complete roster")
    anchor_by_role = {(a["identity"]["logical_pair"], a["identity"]["role"]): a
                      for a in result["anchors"]}
    pairing.require(len(anchor_by_role) == 24, "unique logical pair and role")
    pairing.require(set(anchor_by_role) == {(i, role) for i in range(8) for role in pairing.ROLES},
                    "logical pair and role roster")
    probe_keys = [(p["logical_pair"], p["probe"]) for p in result["common_suffix_pairs"]]
    pairing.require(len(set(probe_keys)) == 336
                    and all(sum(logical == i for logical, _ in probe_keys) == 42 for i in range(8)),
                    "unique probe roster")
    rows = []
    for pair in result["common_suffix_pairs"]:
        pairing.require(set(pair["roles"]) == set(pairing.ROLES), "probe role roster")
        for role in pairing.ROLES:
            common = pair["roles"][role]
            anchor = anchors[common["anchor_index"]]
            identity = anchor["identity"]
            pairing.require(identity["logical_pair"] == pair["logical_pair"]
                            and identity["role"] == role, "anchor binding")
            row = {**identity, "anchor_index": anchor["index"],
                   "prefix_action_sha256": anchor["prefix_action_sha256"],
                   "prefix_cp5_pixel_sha256": anchor["prefix_cp5_pixel_sha256"],
                   "probe": pair["probe"], "probe_sha256": pair["probe_sha256"],
                   "common_status": common["status"],
                   "common_returned_suffix_steps": common["returned_suffix_steps"],
                   "common_last_accepted_terminated": common["last_accepted_terminated"],
                   "common_last_accepted_truncated": common["last_accepted_truncated"],
                   "root_native_success": anchor["root_metrics"]["native_success"],
                   "prefix_native_success": anchor["prefix_metrics"]["native_success"]}
            for label, values in (
                ("root", anchor["root_metrics"]),
                ("prefix", anchor["prefix_metrics"]),
                ("prefix_delta", anchor["prefix_minus_root"]),
                ("common_endpoint", common["endpoint"]),
                ("common_last_accepted", common["last_accepted_metrics"]),
                ("common_delta_own_zero", common["endpoint_minus_own_zero_suffix"]),
                ("common_endpoint_delta_zero_prefix", common["endpoint_minus_zero_prefix"]),
                ("common_response_delta_zero_prefix", common["response_minus_zero_prefix"]),
            ):
                for metric in METRICS:
                    row[f"{label}_{metric}"] = None if values is None else values[metric]
            row["common_native_success"] = (None if common["endpoint"] is None
                                            else common["endpoint"]["native_success"])
            last = common["last_accepted_metrics"]
            row["common_last_accepted_native_success"] = None if last is None else last["native_success"]
            multi = anchor["common_multistart_adam"]
            zero_anchor = anchor_by_role[(pair["logical_pair"], "zero")]
            row["multistart_status"] = multi["status"]
            row["multistart_returned_suffix"] = multi["returned_suffix"]
            row["multistart_last_native_success"] = multi["last_native_success"]
            row["multistart_model_best_score"] = multi["model_best_score"]
            row["multistart_planning_wall_seconds"] = multi["planning_wall_seconds"]
            for resource, count in multi["calls"].items():
                row[f"multistart_reported_{resource}"] = count
            for strategy, outcome in anchor["midpoint_feedback"].items():
                pairing.require(strategy in zero_anchor["midpoint_feedback"], "feedback strategy binding")
                zero_outcome = zero_anchor["midpoint_feedback"][strategy]
                row[f"midpoint_{strategy}_status"] = outcome["status"]
                row[f"midpoint_{strategy}_returned"] = outcome["returned"]
                row[f"midpoint_{strategy}_success"] = outcome["success"]
                for metric in METRICS:
                    row[f"midpoint_{strategy}_endpoint_{metric}"] = (
                        None if outcome["endpoint"] is None else outcome["endpoint"][metric])
                    row[f"midpoint_{strategy}_endpoint_delta_prefix_{metric}"] = (
                        None if outcome["endpoint"] is None
                        else outcome["endpoint"][metric] - anchor["prefix_metrics"][metric])
                    row[f"midpoint_{strategy}_endpoint_delta_zero_prefix_{metric}"] = (
                        None if outcome["endpoint"] is None or zero_outcome["endpoint"] is None
                        else outcome["endpoint"][metric] - zero_outcome["endpoint"][metric])
            for metric in METRICS:
                row[f"multistart_endpoint_{metric}"] = (None if multi["endpoint"] is None
                                                        else multi["endpoint"][metric])
                row[f"multistart_endpoint_delta_prefix_{metric}"] = (
                    None if multi["endpoint"] is None
                    else multi["endpoint"][metric] - anchor["prefix_metrics"][metric])
                row[f"multistart_endpoint_delta_zero_prefix_{metric}"] = (
                    None if multi["endpoint_minus_zero_prefix"] is None
                    else multi["endpoint_minus_zero_prefix"][metric])
            for name in ("zero_initialization", "warm_initialization"):
                plan = anchor[name]
                zero_endpoint = zero_anchor[name]["endpoint"]
                row[f"{name}_status"] = plan["status"]
                row[f"{name}_returned_suffix"] = plan["returned_suffix"]
                endpoint = plan["endpoint"]
                row[f"{name}_native_success"] = None if endpoint is None else endpoint["native_success"]
                last = plan["last_accepted_state_metrics"]
                row[f"{name}_last_accepted_native_success"] = None if last is None else last["native_success"]
                for metric in METRICS:
                    row[f"{name}_endpoint_{metric}"] = None if endpoint is None else endpoint[metric]
                    row[f"{name}_endpoint_delta_prefix_{metric}"] = (
                        None if endpoint is None else endpoint[metric] - anchor["prefix_metrics"][metric])
                    row[f"{name}_endpoint_delta_zero_prefix_{metric}"] = (
                        None if endpoint is None or zero_endpoint is None
                        else endpoint[metric] - zero_endpoint[metric])
            rows.append(row)
    pairing.require(len(rows) == 1008, "branch count")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairing", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.pairing)
    output = Path(args.output)
    pairing.require(output.is_absolute() and output.parent == pairing.ROOT / "results"
                    and output.name.startswith("p1_uphill_pairing_table_")
                    and output.suffix == ".csv" and not output.exists()
                    and not output.is_symlink(), "exclusive output")
    rows = flatten(pairing.read_checked(source, args.sha256))
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    pairing.require(hashlib.sha256(source.read_bytes()).hexdigest() == args.sha256, "source drift")
    raw = text.getvalue().encode("utf-8")
    with output.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(output), "sha256": hashlib.sha256(raw).hexdigest(),
                      "source_sha256": args.sha256, "rows": len(rows)}))


if __name__ == "__main__":
    main()
