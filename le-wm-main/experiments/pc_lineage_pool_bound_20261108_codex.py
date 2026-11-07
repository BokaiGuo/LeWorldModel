"""Read-only finite-pool success bound, recomputed from sealed trajectories."""
import json
from pathlib import Path

import numpy as np

from experiments.pc_lineage_cross import run

NAME = "finite_pool_bound.json"


def success_bound(costs, selected, threshold=.05):
    costs, selected = np.asarray(costs, float), np.asarray(selected, float)
    if costs.ndim != 2 or selected.shape != (len(costs),):
        raise ValueError("one selected cost per finite candidate set required")
    if not np.isfinite(costs).all() or not np.isfinite(selected).all():
        raise ValueError("nonfinite evidence")
    minima = costs.min(1)
    if np.any(selected < minima - 1e-12):
        raise ValueError("selected result exceeds finite-pool bound")
    if not np.all(np.any(np.isclose(costs, selected[:, None], atol=1e-12, rtol=0), axis=1)):
        raise ValueError("selected cost not witnessed in pool")
    possible = minima < threshold
    actual = selected < threshold
    return {
        "per_run_pool_success": possible.tolist(),
        "per_run_selected_success": actual.tolist(),
        "possible_success_runs": int(possible.sum()),
        "selected_success_runs": int(actual.sum()),
        "missed_success_runs": int(np.sum(possible & ~actual)),
        "success_bound_attained_pointwise": bool(np.array_equal(possible, actual)),
    }


def main():
    run.manifest()
    summary = run.load("summary.json")
    sources = {str(path): run.old.sha(path) for path in (
        Path(__file__), run.PROTOCOL, run.OUT / "summary.json", run.OUT / "audit.json",
    )}
    if (run.OUT / NAME).exists():
        record = run.load(NAME)
        if record["sources"] != sources:
            raise ValueError("finite-pool audit source changed")
        print(json.dumps(record["bound"], indent=2))
        return
    costs, selected, rows = [], [], []
    for root in range(24):
        for repeat in range(2):
            name = run.old.record_name("evaluate", root, repeat)
            if summary["evaluations"][name] != run.old.sha(run.OUT / name):
                raise ValueError("raw evaluation changed")
            evaluation = run.load(name)
            plan_name = run.old.record_name("plan", root, repeat)
            if evaluation["plan_sha256"] != run.old.sha(run.OUT / plan_name):
                raise ValueError("plan source changed")
            plan = run.load(plan_name)
            trace = np.asarray(evaluation["full_pool_trajectories"])
            if trace.shape != (272, 16, 3):
                raise ValueError("incomplete finite-pool physical trace")
            recost = np.linalg.norm(trace[:, -1, :2] - plan["task"]["goal"], axis=1)
            np.testing.assert_allclose(recost, np.asarray(evaluation["truth"]).ravel(), atol=1e-12, rtol=0)
            bank = run.core.PairedBank(np.asarray(plan["nominals"]))
            all_pairs = np.asarray([(i, j) for i in range(16) for j in range(17)])
            all_actions = bank.compose(all_pairs)
            chosen = np.asarray(plan["outcomes"]["cross_interaction"]["choice"]["chain"])
            matches = np.flatnonzero(np.all(all_actions == chosen, axis=(1, 2)))
            if not len(matches):
                raise ValueError("selected chain is not in the declared pool")
            distance = plan["outcomes"]["cross_interaction"]["choice"]["score"]
            np.testing.assert_allclose(recost[matches], distance, atol=1e-12, rtol=0)
            costs.append(recost)
            selected.append(distance)
            rows.append({"root": root, "repeat": repeat, "matching_pool_cells": matches.tolist(),
                         "pool_min": float(recost.min()), "selected_distance": distance})
    bound = success_bound(costs, selected)
    bound.update({
        "whole_waypoint_success_runs": summary["arms"]["waypoint_random"]["successes"],
        "remaining_possible_success_increment_at_this_pool": bound["missed_success_runs"],
        "restricted_to_fixed_pool_and_terminal_success": True,
        "not_an_unrestricted_reachability_or_generalization_bound": True,
        "new_model_queries": 0, "gate_promotion": False,
    })
    run.save(NAME, {"sources": sources, "bound": bound, "rows": rows})
    print(json.dumps(bound, indent=2), flush=True)


if __name__ == "__main__":
    main()
