"""Outcome-free impossibility checks for the instantiated stage-cost selector.

This bounds one finite bank under beta=1 and nonnegative suffix costs, not
physical controllability, D2P theory, or the effect of changing beta/tasks.
No neural inference is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage_bound(stages, feasible_zero, *, margin=1e-6):
    stage = np.asarray(stages, dtype=float)
    if (stage.shape != (17,) or not np.isfinite(stage).all() or np.any(stage < 0)
            or not np.isfinite(feasible_zero) or feasible_zero < 0 or margin != 1e-6):
        raise ValueError("all17 nonnegative stages and a finite feasible zero objective required")
    zero_upper = float(stage[0]+feasible_zero/192.)
    non_greedy = np.flatnonzero(stage > stage[0]+margin).tolist()
    # Each candidate's best possible suffix cost is at least zero. The zero
    # prefix already has a known feasible suffix. Ties lose to prefix index0.
    ruled_out = [i for i in non_greedy if stage[i] >= zero_upper]
    return {"zero_total_score_feasible_upper_bound": zero_upper,
        "non_greedy_indices": non_greedy, "ruled_out_even_with_perfect_suffix": ruled_out,
        "all_non_greedy_ruled_out": bool(non_greedy) and ruled_out == non_greedy,
        "no_non_greedy_candidates": not non_greedy,
        "physical_labels_used": False, "P1_admission": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("new immutable stage-contract report required")
    bindings, rows, seen = [], [], set()
    for path in args.reference:
        report = json.loads(path.read_text())
        pp, sp = path.with_suffix(".predictions.json"), path.with_suffix(".protocol.json")
        if (report.get("schema") != "reacher_d2p_masked_suffix_search.v1"
                or report.get("status") != "completed" or not report.get("gates")
                or any(v is not True for v in report["gates"].values())
                or digest(pp) != report["predictions_sha256"] or digest(sp) != report["protocol_sha256"]
                or json.loads(sp.read_text()) != report["sealed_protocol"]):
            raise ValueError("complete sealed suffix-search reference required")
        predictions = json.loads(pp.read_text())
        if predictions["protocol_sha256"] != digest(sp):
            raise ValueError("predictions must bind the pre-execution protocol")
        bindings.append({"path": str(path.resolve()), "sha256": digest(path),
            "predictions_sha256": digest(pp), "protocol_sha256": digest(sp)})
        for row in predictions["rows"]:
            if row["seed"] in seen:
                raise ValueError("duplicate root across model-only panels")
            seen.add(row["seed"])
            target, tasks = np.asarray(row["goal_vector"], float), {}
            if target.shape != (192,) or not np.isfinite(target).all():
                raise ValueError("full192 finite target required")
            for task, plan in row["tasks"].items():
                bank = plan["bank"]
                if [b["index"] for b in bank] != list(range(17)):
                    raise ValueError("all17 ordered original prefixes required")
                v = bank[0]["variants"]["cem_then_gn4"]
                vector, raw = np.asarray(v["terminal_vector"], float), np.asarray(v["raw_actions"], float)
                if (vector.shape != (192,) or raw.shape != (15, 2)
                        or not np.isfinite(vector).all() or not np.isfinite(raw).all() or np.any(np.abs(raw) > 1)):
                    raise ValueError("finite full-dimensional feasible zero-prefix action required")
                objective = float(np.sum((vector-target)**2)+.01*np.sum(raw[5:]**2))
                tasks[task] = stage_bound([b["prefix_cost"] for b in bank], objective)
            rows.append({"seed": row["seed"], "source_episode": row["source_episode"], "tasks": tasks})
    result = {"schema": "reacher_d2p_stage_objective_contract.v1", "bindings": bindings,
        "implementation_sha256": digest(Path(__file__)), "rows": sorted(rows, key=lambda r: r["seed"]),
        "aggregate": {task: {"roots": len(rows),
            "all_non_greedy_ruled_out": sum(r["tasks"][task]["all_non_greedy_ruled_out"] for r in rows),
            "no_non_greedy_candidates": sum(r["tasks"][task]["no_non_greedy_candidates"] for r in rows)}
            for task in ("full_drive", "shoulder_only", "wrist_only")},
        "physical_labels_used": False, "P1_admission": False,
        "scope": "Finite-bank model-only bound for prefix-goal-MSE plus nonnegative suffix objective/192, beta=1. Not a general D2P impossibility result or physical efficacy evidence. No coefficient/task fitting."}
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
