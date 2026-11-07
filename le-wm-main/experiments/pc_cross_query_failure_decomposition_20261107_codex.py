"""Post-decision evaluation only. Never changes sealed searches or their exits."""
import json
from pathlib import Path

import numpy as np

from experiments.pc_cross_query import core, exact_run as run

PROTOCOL = run.ROOT.parent / "docs/PC_CROSS_QUERY_FAILURE_AUDIT_ADDENDUM_20261107_CODEX.md"
OUT = run.OUT / "failure_decomposition.json"


def alternative_prediction(observed, anchors):
    costs = np.asarray(observed, float)
    r = anchors[0]
    baseline = costs[:, :1] + (costs[r] - costs[r, 0])[None]
    columns = costs[:, :4] - baseline[:, :4]
    rows = costs[anchors] - baseline[anchors]
    coefficients = np.linalg.lstsq(columns[anchors].T, columns.T, rcond=1e-8)[0].T
    answer = baseline + coefficients @ rows
    mask = core.cross_mask(anchors)
    answer[mask] = costs[mask]
    return answer


def main():
    run.manifest()
    binding = {str(p): run.sha(p) for p in (Path(__file__), PROTOCOL, run.OUT / "summary.json",
                                           run.OUT / "audit.json")}
    if OUT.exists():
        previous = run.load_record(OUT.name)
        if previous["dependencies"] != binding:
            raise ValueError("diagnosis dependency drift")
        print(json.dumps(previous["summary"], indent=2))
        return
    rows = []
    for root, task in enumerate(run.previous.tasks()):
        for repeat in range(run.REPEATS):
            plan_name, eval_name = (run.record_name(stage, root, repeat) for stage in ("plan", "evaluate"))
            plan, evaluation = run.load_record(plan_name), run.load_record(eval_name)
            if evaluation["plan_sha256"] != run.sha(run.OUT / plan_name):
                raise ValueError("plan evaluation binding mismatch")
            outcome = evaluation["outcomes"]["cross_interaction"]
            success = outcome["success"]
            pool_success = evaluation["pool_success"]
            queried_success = min(
                [*np.concatenate([b["costs"] for b in plan["shared_trace"]]),
                 *plan["outcomes"]["cross_interaction"]["trace"][0]["costs"]]) < .05
            category = ("success" if success else "pool_coverage_failure" if not pool_success
                        else "selection_loss" if queried_success else "proposal_miss")
            truth = np.asarray(evaluation["truth"])
            predicted = np.asarray(plan["outcomes"]["cross_interaction"]["prediction"])
            mask = core.cross_mask(plan["anchors"])
            missing_ids = np.flatnonzero(~mask.ravel())
            ranking = missing_ids[np.argsort(predicted.ravel()[missing_ids], kind="stable")]
            success_ranks = np.flatnonzero(truth.ravel()[ranking] < .05) + 1
            observed = np.asarray([[np.nan if x is None else x for x in r] for r in plan["observed"]])
            alternative = alternative_prediction(observed, plan["anchors"])
            alternative_error = float(np.max(np.abs(alternative - predicted)))
            np.testing.assert_allclose(alternative, predicted, atol=1e-10, rtol=1e-10)

            # These generated nominal chains were NOT scored by the planner.
            # Their new scores exist solely in this post-decision diagnostic.
            nominals = np.stack([run.previous.geom.make_chain(task["start"], task["goal"], x)[0]
                                 for x in plan["offsets"]])
            path, events = run.trajectory(task["start"], nominals)
            distances = np.linalg.norm(path[:, -1, :2] - task["goal"], axis=1)
            results = {
                "prefix_origin_nominal_successes": int(np.sum(distances[:8] < .05)),
                "tail_origin_nominal_successes": int(np.sum(distances[8:] < .05)),
                "prefix_origin_success_but_pool_failure": bool(np.any(distances[:8] < .05) and not pool_success),
                "tail_origin_success_but_pool_failure": bool(np.any(distances[8:] < .05) and not pool_success),
            }
            cem = {}
            for arm in ("step_cem", "block_cem"):
                candidate_actions = np.concatenate([x["actions"] for x in plan["outcomes"][arm]["trace"]])
                candidate_scores = np.concatenate([x["costs"] for x in plan["outcomes"][arm]["trace"]])
                cpath, collisions = run.trajectory(task["start"], candidate_actions)
                updates = plan["outcomes"][arm]["updates"]
                for iteration, update in enumerate(updates):
                    batch = candidate_actions[iteration * 16:(iteration + 1) * 16]
                    samples = batch[:, ::5] if arm == "block_cem" else batch
                    expected_elite = np.argsort(candidate_scores[iteration * 16:(iteration + 1) * 16],
                                               kind="stable")[:4]
                    np.testing.assert_array_equal(expected_elite, update["elite_indices"])
                    np.testing.assert_array_equal(samples[0], update["mean"])
                    if iteration < 7:
                        np.testing.assert_allclose(samples[expected_elite].mean(0), updates[iteration + 1]["mean"],
                                                   atol=1e-12, rtol=0)
                        np.testing.assert_allclose(np.maximum(samples[expected_elite].std(0), 1e-5),
                                                   updates[iteration + 1]["std"], atol=1e-12, rtol=0)
                cem[arm] = {
                    "successful_candidates": int(np.sum(candidate_scores < .05)),
                    "candidates_crossing_wall_plane": int(np.any(cpath[:, :, 0] > 0, axis=1).sum()),
                    "candidates_blocked_at_least_once": int(collisions.any(1).sum()),
                    "minimum_cost": float(candidate_scores.min()),
                    "final_distribution_mean_std": float(np.mean(updates[-1]["std"])),
                    "elite_and_update_recomputed": True,
                }
            rows.append({
                "root": root, "repeat": repeat, "plan_sha256": run.sha(run.OUT / plan_name),
                "evaluation_sha256": run.sha(run.OUT / eval_name),
                "failure_category": category, "successful_missing_pair_ranks": success_ranks.tolist(),
                "missed_successful_pairs": [
                    {"pair": [int(p // 32), int(p % 32)], "predicted": float(predicted.ravel()[p]),
                     "actual": float(truth.ravel()[p])}
                    for p in ranking if truth.ravel()[p] < .05
                ] if category == "proposal_miss" else [],
                "pinv_vs_lstsq_max_error": alternative_error, **results,
                "nominal_actions_evaluation_only": nominals.tolist(),
                "nominal_distances_evaluation_only": distances.tolist(),
                "nominal_trajectories_evaluation_only": path.tolist(),
                "nominal_blocked_evaluation_only": events.tolist(), "cem_audit": cem,
            })
    summary = {
        "failure_categories": {name: sum(r["failure_category"] == name for r in rows)
                               for name in ("success", "pool_coverage_failure", "proposal_miss", "selection_loss")},
        "prefix_origin_success_but_pool_failure_runs": sum(r["prefix_origin_success_but_pool_failure"] for r in rows),
        "tail_origin_success_but_pool_failure_runs": sum(r["tail_origin_success_but_pool_failure"] for r in rows),
        "any_nominal_success_runs": sum(r["prefix_origin_nominal_successes"] + r["tail_origin_nominal_successes"] > 0
                                        for r in rows),
        "pinv_vs_lstsq_max_error": max(r["pinv_vs_lstsq_max_error"] for r in rows),
        "additional_evaluation_only_queries": 48 * 39,
        "additional_cem_audit_replay_queries": 48 * 2 * 128,
        "deployment_results_unchanged": True,
    }
    for arm in ("step_cem", "block_cem"):
        summary[arm] = {
            key: sum(r["cem_audit"][arm][key] for r in rows)
            for key in ("successful_candidates", "candidates_crossing_wall_plane", "candidates_blocked_at_least_once")
        }
    if binding != {p: run.sha(p) for p in binding}:
        raise ValueError("diagnosis source drift")
    run.save_record(OUT.name, {"dependencies": binding, "summary": summary, "rows": rows,
                               "posthoc_diagnostic_only": True, "gate_promotion": False})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
