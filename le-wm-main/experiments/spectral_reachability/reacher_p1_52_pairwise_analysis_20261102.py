"""Analyze fixed replay distances; pairs within a root are not independent."""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

from . import reacher_p1_52_replay_contract_20261102_codexexclusive as contract
from . import reacher_pc_p1_fd_cp5_host_20261102_codexexclusive as host

METRICS = ("predicted_goal_distance", "actual_rgb_goal_distance", "physical_distance")
COMPARISONS = ((0, 1), (1, 2), (0, 2))


def compare(differences):
    output = {}
    for left, right in COMPARISONS:
        eligible = [(d[left], d[right]) for d in differences if d[left] != 0 and d[right] != 0]
        output[f"{METRICS[left]}_vs_{METRICS[right]}"] = {
            "reversed": sum(a * b < 0 for a, b in eligible),
            "non_tied": len(eligible), "total": len(differences)}
    return output


def analyze(receipt):
    host.require(receipt["status"] == "complete" and receipt["completed"] == 52, "incomplete receipt")
    rows = receipt["rows"]
    keys = [(group, seed, i) for group in ("L2", "Q")
            for seed in (20263160, 20263161) for i in range(13)]
    host.require([(r["group"], r["seed"], r["prefix_index"]) for r in rows] == keys, "ordered roster")
    for row in rows:
        host.require(row["status"] == "complete" and all(
            isinstance(row[m], (int, float)) and math.isfinite(row[m]) for m in METRICS), "finite metrics")
    by_key = {(r["group"], r["seed"], r["prefix_index"]): r for r in rows}
    panels, paired = [], []
    for group in ("L2", "Q"):
        for seed in (20263160, 20263161):
            group_rows = [by_key[group, seed, i] for i in range(13)]
            diffs = [[a[m] - b[m] for m in METRICS] for a, b in itertools.combinations(group_rows, 2)]
            panels.append({"group": group, "seed": seed, "pairwise": compare(diffs),
                           "mean_prediction_error": sum(r["prediction_error"] for r in group_rows) / 13})
    for seed in (20263160, 20263161):
        diffs = [[by_key["Q", seed, i][m] - by_key["L2", seed, i][m] for m in METRICS]
                 for i in range(13)]
        paired.append({"seed": seed, "q_minus_l2": compare(diffs)})
    return {"schema": "p1_52_pairwise_analysis_v1", "panels": panels, "same_prefix_q_vs_l2": paired,
            "scope": "two exposed roots; shared trajectories; pair counts are not independent samples",
            "ties": "exact equality only; no numerical tolerance", "independent_outcome": False,
            "P1_admission": False}


def main():
    parser = argparse.ArgumentParser()
    for name in ("receipt", "sha256", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    host.require(contract.sha(args.receipt) == args.sha256, "receipt hash")
    result = analyze(json.loads(Path(args.receipt).read_text()))
    result["receipt_sha256"] = args.sha256
    host.write_new(args.output, result)
    print(json.dumps({"output": args.output, "sha256": contract.sha(args.output)}))


if __name__ == "__main__":
    main()
