"""Join post-hoc observed H10 endpoints to the sealed P1 paired table."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

from . import p1_uphill_complete_pairing_20261030 as pairing
from . import p1_uphill_pairing_csv_20261031 as legacy
from . import p1_uphill_common_h10_observation_20261031 as common

NAME = "p1_uphill_observed_pairing_table_20261031"
COMMON_PATH = pairing.ROOT / "results/p1_uphill_common_h10_observation_20261031_RUN01.json"
COMMON_SHA = "c5387df3a73ed7536b77edccbe8ee77825bfbbf11905605c7ef611a2486a4f72"
PLAN_PATH = pairing.ROOT / "results/p1_uphill_horizon_observation_20261031_RUN01.json"
PLAN_SHA = "6b5d986d9a660b4b18a1b6e8cceeecd0d39dc2894c47d87c5772781a3589332e"
need = pairing.require


def join(data, common_data, plan_data):
    need(common_data["schema"] == common.NAME
         and common_data["input_sha256"] == common.SHA
         and common_data["P1_admission"] is False
         and common_data["independent_confirmation"] is False, "common provenance")
    need(plan_data["schema"] == "p1_uphill_horizon_observation_20261031"
         and plan_data["P1_admission"] is False
         and plan_data["independent_confirmation"] is False, "plan provenance")
    expected = common.build(data)
    need(common_data["rows"] == expected, "common recomputation")
    indexed = {(r["logical_pair"], r["probe"], r["role"]): r for r in expected}
    need(len(indexed) == 1008, "unique common identities")
    plans = {r["index"]: r for r in plan_data["rows"]}
    need(len(plans) == len(plan_data["rows"]) == 24
         and set(plans) == set(range(24)), "planning roster")
    rows = legacy.flatten(data)
    anchors = {a["index"]: a for a in data["anchors"]}
    for row in rows:
        key = (row["logical_pair"], row["probe"], row["role"])
        item = indexed.pop(key)
        plan = plans[row["anchor_index"]]
        need(item["status"] == row["common_status"]
             and item["returned_suffix_steps"] == row["common_returned_suffix_steps"]
             and plan["logical_pair"] == row["logical_pair"] and plan["role"] == row["role"]
             and plan["status"] == row["multistart_status"]
             and plan["returned_suffix"] == row["multistart_returned_suffix"],
             "row identity")
        need(plan["h10_observed"] == (plan["h10_metrics"] is not None)
             and plan["h10_observed"] == (plan["returned_suffix"] == 10),
             "plan observation")
        need(plan["native_success"] == row["multistart_last_native_success"],
             "plan success binding")
        metrics = plan["h10_metrics"]
        if metrics is not None:
            need(metrics["native_success"] == plan["native_success"], "plan metrics success")
            previous = anchors[row["anchor_index"]]["common_multistart_adam"]["endpoint"]
            if previous is not None:
                need(pairing.same_metrics(previous, metrics), "plan endpoint")
        row["observed_common_h10"] = item["h10_metrics"] is not None
        row["observed_plan_h10"] = plan["h10_observed"]
        for label, values in (
            ("observed_common_h10", item["h10_metrics"]),
            ("observed_common_delta_own_zero", item["endpoint_minus_own_zero_suffix"]),
            ("observed_common_endpoint_delta_zero_prefix", item["endpoint_minus_zero_prefix"]),
            ("observed_common_response_delta_zero_prefix", item["response_minus_zero_prefix"]),
            ("observed_plan_h10", metrics),
        ):
            for metric in pairing.FIELDS:
                row[f"{label}_{metric}"] = None if values is None else values[metric]
        row["observed_common_h10_native_success"] = (
            None if item["h10_metrics"] is None else item["h10_metrics"]["native_success"])
        row["observed_plan_h10_native_success"] = (
            None if metrics is None else metrics["native_success"])
    need(not indexed, "all common observations used")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    need(output.is_absolute() and output.parent == pairing.ROOT / "results"
         and output.name.startswith(NAME + "_") and output.suffix == ".csv"
         and not output.exists() and not output.is_symlink(), "exclusive output")
    paths = ((common.PATH, common.SHA), (COMMON_PATH, COMMON_SHA), (PLAN_PATH, PLAN_SHA))
    data, common_data, plan_data = [pairing.read_checked(path, sha) for path, sha in paths]
    rows = join(data, common_data, plan_data)
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    need(all(hashlib.sha256(path.read_bytes()).hexdigest() == sha for path, sha in paths),
         "source drift")
    raw = text.getvalue().encode("utf-8")
    with output.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(output), "rows": len(rows),
                      "common_observed": sum(r["observed_common_h10"] for r in rows),
                      "sha256": hashlib.sha256(raw).hexdigest(),
                      "sources": {str(path): sha for path, sha in paths}}))


if __name__ == "__main__":
    main()
