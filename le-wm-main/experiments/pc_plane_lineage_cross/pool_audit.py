"""Post-selection support certificate with all-pool independent scalar replay."""
import json
from pathlib import Path

import numpy as np

from experiments.pc_cross_query.audit import scalar_trajectory
from experiments.pc_plane_lineage_cross import run

NAME = "pool_audit.json"


def main():
    run.manifest()
    summary = run.load("summary.json")
    sources = {str(p): run.base.sha(p) for p in (
        Path(__file__), run.OUT / "summary.json", run.OUT / "audit.json",
        Path(scalar_trajectory.__code__.co_filename),
    )}
    hashes = {}
    for name, expected in summary["evaluations"].items():
        hashes[name] = run.base.sha(run.OUT / name)
        if hashes[name] != expected:
            raise ValueError("raw evaluation changed")
        pname = name.replace("evaluate_", "plan_")
        hashes[pname] = run.base.sha(run.OUT / pname)
        if run.load(name)["plan_sha256"] != hashes[pname]:
            raise ValueError("evaluation/plan binding changed")
    if (run.OUT / NAME).exists():
        old = run.load(NAME)
        if old["sources"] != sources or old["records"] != hashes:
            raise ValueError("support certificate binding changed")
        print("Verified completed all-pool support certificate.", flush=True)
        return
    rows, maximum_error, maximum_split_error = [], 0., 0.
    pair_ids = np.asarray([(i, j) for i in range(16) for j in range(17)])
    lineage_mask = np.zeros((16, 17), bool)
    lineage_mask[np.arange(16), np.arange(1, 17)] = True
    for root in range(24):
        for repeat in range(2):
            pname, ename = (run.base.record_name(stage, root, repeat) for stage in ("plan", "evaluate"))
            plan, evaluation = run.load(pname), run.load(ename)
            bank = run.core.PairedBank(np.asarray(plan["nominals"]))
            full = bank.compose(pair_ids)
            paths = np.stack([scalar_trajectory(plan["task"]["start"], actions) for actions in full])
            maximum_error = max(maximum_error, float(np.max(np.abs(paths - evaluation["full_pool_trajectories"]))))
            np.testing.assert_allclose(paths, evaluation["full_pool_trajectories"], atol=1e-12, rtol=0)
            # Independently compose two physical integrations; do not reset heading at the split.
            split_paths = []
            for actions in full:
                prefix = scalar_trajectory(plan["task"]["start"], actions[:5])
                tail = scalar_trajectory(prefix[-1], actions[5:])
                split_paths.append(np.concatenate((prefix, tail[1:])))
            maximum_split_error = max(maximum_split_error, float(np.max(np.abs(paths - split_paths))))
            np.testing.assert_allclose(paths, split_paths, atol=1e-12, rtol=0)
            costs = np.linalg.norm(paths[:, -1, :2] - plan["task"]["goal"], axis=1).reshape(16, 17)
            np.testing.assert_allclose(costs, evaluation["truth"], atol=1e-12, rtol=0)
            chosen = np.asarray(plan["outcomes"]["cross_interaction"]["choice"]["chain"])
            matches = np.flatnonzero(np.all(full == chosen, axis=(1, 2)))
            if len(matches) == 0:
                raise ValueError("selected chain is outside this candidate pool")
            selected_cost = plan["outcomes"]["cross_interaction"]["choice"]["score"]
            np.testing.assert_allclose(costs.ravel()[matches], selected_cost, atol=1e-12, rtol=0)
            nominal_success = bool(np.any(costs[lineage_mask] < .05))
            possible_success = bool(np.any(costs < .05))
            selected_success = selected_cost < .05
            if (nominal_success, possible_success, selected_success) != (
                    evaluation["nominal_success"], evaluation["pool_success"],
                    evaluation["outcomes"]["cross_interaction"]["success"]):
                raise ValueError("success evidence mismatch")
            rows.append({
                "root": root, "repeat": repeat, "nominal_success": nominal_success,
                "pool_success": possible_success, "selected_success": selected_success,
                "success_outside_nominal_cells": int(np.sum(costs[~lineage_mask] < .05)),
                "pool_min": float(costs.min()), "selected_cost": selected_cost,
                "matching_selected_pair_ids": pair_ids[matches].tolist(),
            })
    result = {
        "sources": sources, "records": hashes, "rows": rows,
        "independent_pool_chain_replays": 48*272,
        "independent_split_chain_replays": 48*272,
        "max_scalar_vs_saved_error": maximum_error, "max_split_vs_whole_error": maximum_split_error,
        "nominal_successes": sum(r["nominal_success"] for r in rows),
        "pool_successes": sum(r["pool_success"] for r in rows),
        "selected_successes": sum(r["selected_success"] for r in rows),
        "new_successful_runs_created_by_entire_pool": sum(r["pool_success"] and not r["nominal_success"] for r in rows),
        "missed_success_runs": sum(r["pool_success"] and not r["selected_success"] for r in rows),
        "success_bound_attained_pointwise": all(r["pool_success"] == r["selected_success"] for r in rows),
        "new_planning_queries": 0, "restricted_to_this_finite_pool": True,
        "not_an_unrestricted_reachability_or_all_composition_bound": True,
        "not_independent_environment_or_rgb_evidence": True,
    }
    run.save(NAME, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("sources", "records", "rows")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
