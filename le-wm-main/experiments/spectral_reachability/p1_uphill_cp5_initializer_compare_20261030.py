"""Pair verified warm- and zero-initialized suffix experiments."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from . import p1_joint_complete_pairing_20261030 as base

ROOT = base.ROOT
ZERO = ROOT / "results/p1_uphill_cp5_analysis_20261030_RUN01.json"
ZERO_SHA = "04c336045d689dd4629c1c88b08653e6ba7f3402b38a72f1e42203bd0572495e"


def compare(zero, warm):
    base.require(len(zero["rows"]) == len(warm["rows"]) == 24, "fixed roster")
    rows = []
    for z, w in zip(zero["rows"], warm["rows"], strict=True):
        for key in ("index", "logical_pair", "root_slot", "physical_root_sha256", "condition",
                    "arm", "role", "task_id", "prefix_action_sha256", "prefix_cp5_pixel_sha256",
                    "root_metrics", "prefix_metrics"):
            base.require(z[key] == w[key], "paired identity " + key)
        item = {key: z[key] for key in ("index", "logical_pair", "root_slot", "physical_root_sha256",
                                        "condition", "arm", "role", "task_id")}
        for label, row in (("zero_init", z), ("warm_init", w)):
            suffix = row["independent_suffix_20step"]
            item[label + "_status"] = suffix["status"]
            item[label + "_returned_suffix_steps"] = suffix["returned_suffix"]
            item[label + "_last_accepted_native_success"] = (
                None if suffix["last_accepted_state_metrics"] is None
                else suffix["last_accepted_state_metrics"]["native_success"])
            endpoint = suffix["endpoint"]
            item[label + "_endpoint_native_success"] = None if endpoint is None else endpoint["native_success"]
            for key in base.NUMERIC:
                item[label + "_endpoint_" + key] = None if endpoint is None else endpoint[key]
        difference = base.difference(w["independent_suffix_20step"]["endpoint"],
                                     z["independent_suffix_20step"]["endpoint"])
        for key in base.NUMERIC:
            item["warm_minus_zero_" + key] = None if difference is None else difference[key]
        rows.append(item)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warm-analysis", type=Path, required=True)
    parser.add_argument("--warm-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    base.require(out.is_absolute() and out.parent == ROOT / "results"
                 and out.name.startswith("p1_uphill_cp5_initializer_compare_")
                 and out.suffix == ".csv" and not out.exists() and not out.is_symlink(), "exclusive output")
    zero = base.read_checked(ZERO, ZERO_SHA)
    warm = base.read_checked(args.warm_analysis, args.warm_sha256)
    rows = compare(zero, warm)
    with out.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output": str(out), "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                      "rows": len(rows), "warm_analysis_sha256": args.warm_sha256,
                      "zero_analysis_sha256": ZERO_SHA,
                      "warning": "Known roots and duplicate arms; planning cost coverage is not certified."}))


if __name__ == "__main__":
    main()
