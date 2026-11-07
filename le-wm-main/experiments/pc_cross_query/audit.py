"""Trace-level replay and independent scalar dynamics checks before evaluation."""
import math

import numpy as np

from experiments.pc_cross_query import core, exact_run as run


def scalar_trajectory(start, actions):
    """Independent scalar expression of the specified wall/unicycle dynamics."""
    x, y, theta = map(float, start)
    result = [[x, y, theta]]
    for velocity, turn in np.asarray(actions, float):
        dx = velocity * .3 * math.cos(theta)
        dy = velocity * .3 * math.sin(theta)
        end_x, end_y = x + dx, y + dy
        crosses = ((x <= 0 <= end_x) or (end_x <= 0 <= x)) and abs(dx) > 1e-15
        blocked = crosses and abs(y + (-x / dx) * dy) < 1.
        if not blocked:
            x, y = end_x, end_y
        theta += turn * math.pi / 2
        result.append([x, y, theta])
    return np.asarray(result)


def audit():
    run.manifest()
    paths = {run.record_name("plan", root, repeat): run.sha(run.OUT / run.record_name("plan", root, repeat))
             for root in range(24) for repeat in range(run.REPEATS)}
    if (run.OUT / "audit.json").exists():
        old = run.load_record("audit.json")
        if old["plans"] != paths:
            raise ValueError("audited plan changed")
        print("verified completed audit", flush=True)
        return
    counts, maximum_score_error, maximum_trajectory_error = 0, 0., 0.
    rank_counts, unique_prefix_counts, unique_tail_counts = [], [], []
    for name in paths:
        row = run.load_record(name)
        prefixes, tails, offsets = run.bank(row["task"], row["seed"])
        np.testing.assert_array_equal(prefixes, row["prefixes"])
        np.testing.assert_array_equal(tails, row["tails"])
        np.testing.assert_array_equal(offsets, row["offsets"])
        unique_prefix_counts.append(len(np.unique(prefixes.reshape(8, -1), axis=0)))
        unique_tail_counts.append(len(np.unique(tails.reshape(32, -1), axis=0)))
        observed = np.array([[np.nan if x is None else x for x in r] for r in row["observed"]])
        core.validate_observed(observed, row["anchors"])
        np.testing.assert_array_equal(core.anchor_rows(observed[:, :4]), row["anchors"])
        predictions, diagnostic = core.predict_all(observed, row["anchors"], row["seed"] + 20000)
        rank_counts.append(diagnostic["retained_rank"])
        mask = core.cross_mask(row["anchors"])
        for arm, outcome in row["outcomes"].items():
            batches = (row["shared_trace"] if outcome["uses_shared_cross"] else []) + outcome["trace"]
            costs, actions = [], []
            for batch in batches:
                chain = np.asarray(batch["actions"])
                # Historical vector backend pads to 16; compare the actual t=15 endpoint.
                endpoint = run.previous.base.exact.endpoints(row["task"], chain)
                score = np.linalg.norm(endpoint - row["task"]["goal"], axis=1)
                maximum_score_error = max(maximum_score_error, float(np.max(np.abs(score - batch["costs"]))))
                np.testing.assert_allclose(endpoint, batch["endpoints"], rtol=0, atol=1e-12)
                np.testing.assert_allclose(score, batch["costs"], rtol=0, atol=1e-12)
                if batch["pairs"] is not None:
                    np.testing.assert_array_equal(chain, core.compose(prefixes, tails, batch["pairs"]))
                costs.extend(batch["costs"])
                actions.extend(batch["actions"])
            if len(costs) != 128 or outcome["query_count"] != 128 or outcome["dynamics_steps"] != 1920:
                raise ValueError("trace budget mismatch")
            index = int(np.argmin(costs))
            if index != outcome["choice"]["query_index"] or costs[index] != outcome["choice"]["score"]:
                raise ValueError("choice does not preserve best measured chain")
            np.testing.assert_array_equal(actions[index], outcome["choice"]["chain"])
            scalar = scalar_trajectory(row["task"]["start"], actions[index])
            batch_path, _ = run.trajectory(row["task"]["start"], np.asarray(actions[index])[None])
            error = float(np.max(np.abs(scalar - batch_path[0])))
            maximum_trajectory_error = max(maximum_trajectory_error, error)
            np.testing.assert_allclose(scalar, batch_path[0], atol=1e-12, rtol=0)
            if outcome["uses_shared_cross"]:
                pairs = np.asarray(outcome["proposal_pairs"])
                if mask[pairs[:, 0], pairs[:, 1]].any() or len(np.unique(pairs, axis=0)) != 12:
                    raise ValueError("proposal repeats observed or duplicate cell")
                if arm in predictions:
                    np.testing.assert_allclose(predictions[arm], outcome["prediction"], atol=1e-12, rtol=0)
                    np.testing.assert_array_equal(core.proposals(predictions[arm], mask), pairs)
                else:
                    missing = np.argwhere(~mask)
                    expected = missing[np.random.default_rng(row["seed"] + 30000).choice(len(missing), 12, replace=False)]
                    np.testing.assert_array_equal(expected, pairs)
            counts += 1
        print(f"audited {name}", flush=True)
    run.save_record("audit.json", {
        "plans": paths, "plans_audited": counts,
        "maximum_score_error_vs_padded_backend": maximum_score_error,
        "maximum_trajectory_error_vs_scalar": maximum_trajectory_error,
        "retained_ranks": rank_counts, "unique_prefix_counts": unique_prefix_counts,
        "unique_tail_counts": unique_tail_counts,
        "all_budgets_actions_incumbents_and_proposals_verified": True,
        "source_and_same_physics_replication_not_independent_environment": True,
    })
