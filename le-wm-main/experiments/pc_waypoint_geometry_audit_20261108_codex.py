"""Independent raw-action audit of the frozen waypoint experiment.

No new planning decisions are made. All scalar replays are evaluation costs.
"""
import json
from pathlib import Path

import numpy as np

from experiments import pc_waypoint_geometry_20261108_codex as run

NAME = "independent_audit.json"


def check_selection(costs, choice):
    values = np.asarray(costs, float)
    if values.shape != (128,) or not np.isfinite(values).all():
        raise ValueError("128 finite queried costs required")
    if isinstance(choice, bool) or not isinstance(choice, int) or choice != int(np.argmin(values)):
        raise ValueError("selection is not the first minimum queried chain")
    return bool(values[choice] < .05)


def main():
    manifest = run.manifest()
    summary = run.load("summary.json")
    sources = {str(p): run.base.sha(p) for p in (
        Path(__file__), run.OUT / "manifest.json", run.OUT / "summary.json",
        Path(run.scalar_trajectory.__code__.co_filename),
    )}
    plans = {run.base.record_name("plan", i, j): run.base.sha(
        run.OUT / run.base.record_name("plan", i, j)) for i in range(24) for j in range(2)}
    evaluations = {name: run.base.sha(run.OUT / name) for name in summary["evaluations"]}
    if evaluations != summary["evaluations"]:
        raise ValueError("evaluation content changed")
    if (run.OUT / NAME).exists():
        old = run.load(NAME)
        if old["sources"] != sources or old["plans"] != plans or old["evaluations"] != evaluations:
            raise ValueError("independent audit binding changed")
        print("Verified completed independent waypoint audit.", flush=True)
        return

    rows, counts, max_path_error, max_score_error = [], 0, 0., 0.
    for root in range(24):
        for repeat in range(2):
            pname = run.base.record_name("plan", root, repeat)
            ename = run.base.record_name("evaluate", root, repeat)
            plan, evaluation = run.load(pname), run.load(ename)
            if (plan["root"], plan["repeat"], plan["task"]) != (root, repeat, manifest["tasks"][root]):
                raise ValueError("task identity changed")
            if evaluation["plan_sha256"] != plans[pname]:
                raise ValueError("evaluation plan binding changed")
            seed = 2831000 + 100 * root + repeat
            if plan["seed"] != seed:
                raise ValueError("seed changed")
            lateral = np.random.default_rng(seed).uniform(-2, 2, 128)
            longitudinal = np.random.default_rng(seed + 70000).uniform(-1, 1, 128)
            np.testing.assert_array_equal(plan["lateral"], lateral)
            record = {"root": root, "repeat": repeat, "arms": {}}
            for arm in run.ARMS:
                saved = plan["arms"][arm]
                longitudinal_used = longitudinal.copy()
                if arm == "line_1d":
                    longitudinal_used[:] = 0
                elif arm == "lineage_mixture":
                    longitudinal_used[:16] = 0
                np.testing.assert_array_equal(saved["longitudinal"], longitudinal_used)
                generated = [run.generate(plan["task"], a, b)
                             for a, b in zip(lateral, longitudinal_used, strict=True)]
                actions = np.asarray(saved["actions"])
                np.testing.assert_array_equal(actions, np.stack([g[0] for g in generated]))
                np.testing.assert_array_equal(saved["waypoints"], np.stack([g[1] for g in generated]))
                np.testing.assert_array_equal(saved["required_steps"], [g[2] for g in generated])
                if saved["segment_ends"] != [g[3] for g in generated]:
                    raise ValueError("segment duration changed")
                if actions.shape != (128, 15, 2) or saved["queries"] != 128:
                    raise ValueError("query count or physical horizon changed")
                if not np.isfinite(actions).all() or np.max(np.abs(actions)) > 1:
                    raise ValueError("illegal action")
                scalar = np.stack([run.scalar_trajectory(plan["task"]["start"], a) for a in actions])
                reference = evaluation["outcomes"][arm]
                max_path_error = max(max_path_error,
                                     float(np.max(np.abs(scalar - reference["full_trajectories"]))))
                np.testing.assert_allclose(scalar, reference["full_trajectories"], atol=1e-12, rtol=0)
                costs = np.linalg.norm(scalar[:, -1, :2] - plan["task"]["goal"], axis=1)
                max_score_error = max(max_score_error, float(np.max(np.abs(costs - saved["costs"]))))
                np.testing.assert_allclose(costs, saved["costs"], atol=1e-12, rtol=0)
                success = check_selection(saved["costs"], saved["choice"])
                if success != reference["success"] or success != bool(costs[saved["choice"]] < .05):
                    raise ValueError("success mismatch")
                if success != bool(np.any(costs < .05)):
                    raise ValueError("selected success does not attain this candidate pool ceiling")
                free_cost = np.asarray(reference["free_costs"])
                complete = np.asarray(saved["required_steps"]) <= 15
                if np.any(free_cost[complete] > 1e-10):
                    raise ValueError("obstacle-free compiler inconsistency")
                selected = saved["choice"]
                record["arms"][arm] = {
                    "success": success, "successful_candidates": int(np.sum(costs < .05)),
                    "choice": selected, "required_steps": saved["required_steps"][selected],
                    "selected_free_distance": float(free_cost[selected]),
                    "selected_collision_steps": np.flatnonzero(reference["blocked"][selected]).tolist(),
                    "first_success_query": int(np.flatnonzero(costs < .05)[0] + 1) if success else None,
                }
                counts += len(actions)
            np.testing.assert_array_equal(plan["arms"]["lineage_mixture"]["actions"][:16],
                                          plan["arms"]["line_1d"]["actions"][:16])
            np.testing.assert_array_equal(plan["arms"]["lineage_mixture"]["actions"][16:],
                                          plan["arms"]["plane_2d"]["actions"][16:])
            rows.append(record)
            print(f"Independently audited all 384 candidates r{root}/s{repeat}", flush=True)
    measured = {arm: sum(row["arms"][arm]["success"] for row in rows) for arm in run.ARMS}
    if measured != {arm: summary["arms"][arm]["successes"] for arm in run.ARMS}:
        raise ValueError("summary success totals changed")
    run.save(NAME, {
        "sources": sources, "plans": plans, "evaluations": evaluations, "rows": rows,
        "all_candidate_scalar_replays": counts, "all_candidate_scalar_steps": counts * 15,
        "maximum_path_error": max_path_error, "maximum_score_error": max_score_error,
        "successes_recomputed": measured,
        "all_candidates_seed_regenerated": True, "selected_success_equals_pool_success_pointwise": True,
        "planning_queries_added": 0, "evaluation_queries_added": counts,
        "independent_code_same_environment_not_independent_research_confirmation": True,
        "gate_promotion": False,
    })
    print(json.dumps({"successes": measured, "scalar_replays": counts,
                      "maximum_path_error": max_path_error, "maximum_score_error": max_score_error}), flush=True)


if __name__ == "__main__":
    main()
