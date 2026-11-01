"""Pair immutable legacy/full-step and backtracking model-only profiles."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def _indexed(rows, key):
    result = {row[key]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate {key}")
    return result


def compare(legacy: dict, repaired: dict) -> dict:
    if legacy.get("status") != "completed" or repaired.get("status") != "completed":
        raise ValueError("complete profiles required")
    for key in ("weights_sha256", "reference_sha256", "dampings", "future_horizon", "root_seeds"):
        if legacy["protocol"][key] != repaired["protocol"][key]:
            raise ValueError(f"different profile {key}")
    if repaired["protocol"].get("suffix_step_rule") != "backtracking":
        raise ValueError("explicit backtracking profile required")
    old_roots, new_roots = _indexed(legacy["roots"], "seed"), _indexed(repaired["roots"], "seed")
    if old_roots.keys() != new_roots.keys() or set(old_roots) != set(legacy["protocol"]["root_seeds"]):
        raise ValueError("root identity mismatch")
    ascents = Counter()
    scales = Counter()
    max_full_error = 0.0
    max_ascent = 0.0
    paired = []
    for seed in sorted(old_roots):
        old, new = old_roots[seed], new_roots[seed]
        for key in ("root_rgb_sha256", "goal_rgb_sha256"):
            if old[key] != new[key]:
                raise ValueError(f"different {key}")
        old_rows, new_rows = _indexed(old["proposal_rows"], "proposal_index"), _indexed(new["proposal_rows"], "proposal_index")
        if old_rows.keys() != new_rows.keys():
            raise ValueError("candidate identity mismatch")
        for index, before in old_rows.items():
            after = new_rows[index]
            for key in ("immediate_goal_cost", "predicted_terminal_goal_cost", "singular_values"):
                if before[key] != after[key]:
                    raise ValueError(f"base model computation changed: {key}")
            old_grid = _indexed(before["diagnostic_damping_scores"], "damping")
            new_grid = _indexed(after["diagnostic_damping_scores"], "damping")
            if old_grid.keys() != new_grid.keys():
                raise ValueError("damping grid changed")
            for damping, original in old_grid.items():
                trial = new_grid[damping]
                cost, origin = trial["bounded_dls_suffix_goal_cost"], after["predicted_terminal_goal_cost"]
                full = trial["full_step_bounded_dls_suffix_goal_cost"]
                if not np.isfinite([cost, origin, full]).all():
                    raise ValueError("nonfinite model objective")
                max_full_error = max(max_full_error, abs(full - original["bounded_dls_suffix_goal_cost"]))
                max_ascent = max(max_ascent, cost - origin)
                ascents[str(damping)] += original["bounded_dls_suffix_goal_cost"] > origin
                scales[str(trial["accepted_step_scale"])] += 1
        old_summary, new_summary = _indexed(old["summary"], "damping"), _indexed(new["summary"], "damping")
        paired.append({"seed": seed, "cells": [
            {"damping": damping, "legacy_witness_count": old_summary[damping]["strong_future_witness_count"],
             "repaired_witness_count": new_summary[damping]["strong_future_witness_count"],
             "legacy_selected_index": old_summary[damping]["selected_index"],
             "repaired_selected_index": new_summary[damping]["selected_index"]}
            for damping in old_summary]})
    gates = {"same_full_step_scores_exact": max_full_error == 0.0, "all_accepted_costs_nonascending": max_ascent <= 0.0}
    return {"schema": "reacher_d2p_suffix_optimizer_repair_comparison.v1",
        "status": "pass" if all(gates.values()) else "fail", "gates": gates,
        "root_count": len(paired), "old_full_step_ascent_count_by_damping": dict(ascents),
        "accepted_step_scale_counts": dict(scales), "max_full_step_replay_error": max_full_error,
        "max_accepted_cost_increase": max_ascent, "paired_roots": paired,
        "claim_boundary": "Code-repair causal comparison of fixed model-only roots, not physical efficacy or P1 admission."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--repaired", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = compare(json.loads(args.legacy.read_text()), json.loads(args.repaired.read_text()))
    report["inputs"] = {label: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                        for label, path in (("legacy", args.legacy), ("repaired", args.repaired))}
    report["auditor_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open("x") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("paired_roots", "inputs")}))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
