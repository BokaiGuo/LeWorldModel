"""Audit paired first-solve candidate equality and closed-loop outcomes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def analyze(root: Path, stem: str) -> dict:
    modes = ("mean", "sample", "joint")
    traces = {}
    outcomes = {}
    for mode in modes:
        base = root / stem.format(mode=mode)
        with base.with_suffix(".jsonl").open(encoding="utf-8") as stream:
            traces[mode] = [json.loads(line) for line in stream]
        with (base / "eval_metrics.json").open(encoding="utf-8") as stream:
            metrics = json.load(stream)
        outcomes[mode] = np.asarray(metrics["episode_successes"], dtype=bool)
        if len(outcomes[mode]) != 50 or not traces[mode] or traces[mode][0]["solve"] != 0:
            raise ValueError(f"incomplete run: {mode}")
    reference = traces["mean"][0]
    keys = ("mean_actions", "sample_actions", "mean_costs", "sample_costs")
    first_solve = {}
    for mode in modes:
        record = traces[mode][0]
        first_solve[mode] = {}
        for key in keys:
            left, right = np.asarray(reference[key]), np.asarray(record[key])
            if left.shape != right.shape or not np.isfinite(right).all():
                raise ValueError(f"invalid {mode}/{key}")
            first_solve[mode][key] = {
                "shape": list(right.shape),
                "exact": bool(np.array_equal(left, right)),
                "max_abs_difference": float(np.max(np.abs(left - right))),
            }
    pairs = {}
    for left, right in (("mean", "sample"), ("mean", "joint"), ("sample", "joint")):
        a, b = outcomes[left], outcomes[right]
        pairs[f"{left}_vs_{right}"] = {
            "both_success": int(np.sum(a & b)),
            "left_only": int(np.sum(a & ~b)),
            "right_only": int(np.sum(~a & b)),
            "both_fail": int(np.sum(~a & ~b)),
            "different_indices": np.flatnonzero(a != b).tolist(),
        }
    return {
        "scope": "development only; first solve equality is not later-context equality",
        "first_solve_comparison": first_solve,
        "first_solve_joint_chooses_sample": int(sum(traces["joint"][0]["choose_sample"])),
        "solves_and_batch_sizes": {m: [len(r["choose_sample"]) for r in traces[m]] for m in modes},
        "joint_choices_across_solves": {
            "sample": int(sum(sum(r["choose_sample"]) for r in traces["joint"])),
            "total": int(sum(len(r["choose_sample"]) for r in traces["joint"])),
        },
        "successes": {m: int(outcomes[m].sum()) for m in modes},
        "paired_outcomes": pairs,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stem", default="pc_native_mean_pool_{mode}_seed42_20261105_codex")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root, args.stem)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2, allow_nan=False))
