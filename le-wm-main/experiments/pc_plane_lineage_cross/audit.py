"""Audit immutable decisions before opening the exhaustive evaluator matrix."""
import hashlib

import numpy as np

from experiments.pc_cross_query.audit import scalar_trajectory
from experiments.pc_plane_lineage_cross import run


def verify_record(row):
    """Reconstruct decisions using only the seed, actions and queried costs."""
    seed = 2831000 + row["root"] * 100 + row["repeat"]
    if row["seed"] != seed:
        raise ValueError("random stream identity changed")
    nominals, parameters = run.source_chains(row["task"], seed)
    np.testing.assert_array_equal(nominals, row["nominals"])
    np.testing.assert_array_equal(parameters, row["parameters"])
    if hashlib.sha256(nominals.tobytes()).hexdigest() != row["nominal_sha256"]:
        raise ValueError("nominal action fingerprint changed")
    bank = run.core.PairedBank(nominals)
    observed = np.asarray(row["observed"], float)
    _, mask = run.core.check_observed(observed, row["anchors"])
    np.testing.assert_array_equal(run.core.anchors(observed[:, :4]), row["anchors"])
    if row["shared_queries"] != int(mask.sum()):
        raise ValueError("shared query count mismatch")
    reconstructed = np.full(bank.shape, np.nan)
    for batch in row["shared_trace"]:
        pairs = np.asarray(batch["pairs"])
        if np.isfinite(reconstructed[pairs[:, 0], pairs[:, 1]]).any():
            raise ValueError("shared pair queried twice")
        reconstructed[pairs[:, 0], pairs[:, 1]] = batch["costs"]
    np.testing.assert_allclose(reconstructed, observed, atol=0, rtol=0, equal_nan=True)
    baseline = observed[:, :1] + (observed[row["anchors"][0]] - observed[row["anchors"][0], 0])[None]
    columns = observed[:, :4] - baseline[:, :4]
    rows = observed[row["anchors"]] - baseline[row["anchors"]]
    independent = baseline + np.linalg.lstsq(columns[row["anchors"]].T, columns.T, rcond=1e-8)[0].T @ rows
    independent[mask] = observed[mask]
    prediction = row["outcomes"]["cross_interaction"]["prediction"]
    np.testing.assert_allclose(independent, prediction, atol=1e-10, rtol=1e-10)
    maximum_formula_error = float(np.max(np.abs(independent - prediction)))
    maximum_path_error = maximum_score_error = 0.
    scalar_replays = 0
    for arm, result in row["outcomes"].items():
        batches = run.batches(row, arm)
        np.testing.assert_array_equal(batches[0]["actions"], nominals)
        costs, actions, all_pairs = [], [], []
        for batch in batches:
            queried = np.asarray(batch["actions"])
            if queried.shape[1:] != (15, 2) or np.max(np.abs(queried)) > 1:
                raise ValueError("illegal queried action")
            vector, _ = run.base.trajectory(row["task"]["start"], queried)
            scalar = np.stack([scalar_trajectory(row["task"]["start"], chain) for chain in queried])
            maximum_path_error = max(maximum_path_error, float(np.max(np.abs(vector - scalar))))
            np.testing.assert_allclose(vector, scalar, atol=1e-12, rtol=0)
            actual = np.linalg.norm(scalar[:, -1, :2] - row["task"]["goal"], axis=1)
            maximum_score_error = max(maximum_score_error, float(np.max(np.abs(actual - batch["costs"]))))
            np.testing.assert_allclose(actual, batch["costs"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(scalar[:, -1, :2], batch["endpoints"], atol=1e-12, rtol=0)
            if batch["pairs"] is not None:
                np.testing.assert_array_equal(queried, bank.compose(batch["pairs"]))
                all_pairs.extend(batch["pairs"])
            costs.extend(batch["costs"])
            actions.extend(batch["actions"])
            scalar_replays += len(queried)
        if len(costs) != 128 or result["queries"] != 128:
            raise ValueError("query budget mismatch")
        best = int(np.argmin(costs))
        if result["choice"]["query_index"] != best or result["choice"]["score"] != costs[best]:
            raise ValueError("actual queried incumbent lost")
        np.testing.assert_array_equal(result["choice"]["chain"], actions[best])
        if costs[best] > min(costs[:16]):
            raise ValueError("original nominal witness lost")
        if result["uses_shared_cross"]:
            if len(np.unique(all_pairs, axis=0)) != 128:
                raise ValueError("paired queries not unique")
            pairs = np.asarray(result["proposal_pairs"])
            if mask[pairs[:, 0], pairs[:, 1]].any():
                raise ValueError("proposal queried an already observed cell")
            remaining = 128 - row["shared_queries"]
            if arm == "random_missing":
                missing = np.argwhere(~mask)
                expected_pairs = missing[np.random.default_rng(seed + 30000).choice(
                    len(missing), remaining, replace=False)]
            else:
                expected, _ = run.core.prediction(arm, observed, row["anchors"], seed + 20000)
                np.testing.assert_allclose(expected, result["prediction"], atol=1e-12, rtol=0)
                expected_pairs = run.core.choose(expected, mask, remaining)
            np.testing.assert_array_equal(pairs, expected_pairs)
        elif arm == "whole_plane":
            full, _ = run.source_chains(row["task"], seed, 128)
            np.testing.assert_array_equal(actions, full)
        elif arm in ("uniform_lookahead", "random_pairs"):
            known = np.zeros(bank.shape, bool)
            known[np.arange(16), np.arange(1, 17)] = True
            if arm == "uniform_lookahead":
                expected = np.asarray([(i, j) for j in range(17) for i in range(16) if not known[i, j]])[:112]
            else:
                missing = np.argwhere(~known)
                expected = missing[np.random.default_rng(seed + 40000).choice(len(missing), 112, replace=False)]
            np.testing.assert_array_equal(batches[1]["pairs"], expected)
            if len(np.unique(all_pairs, axis=0)) != 128:
                raise ValueError("baseline repeats pair identities")
        elif arm == "nominal_cem":
            rng = np.random.default_rng(seed + 60000)
            loc, scale = nominals[np.argmin(costs[:16])], np.full((15, 2), 1/3)
            if len(result["updates"]) != 7 or len(batches) != 8:
                raise ValueError("CEM iteration count mismatch")
            for i, update in enumerate(result["updates"]):
                np.testing.assert_array_equal(update["mean"], loc)
                np.testing.assert_array_equal(update["std"], scale)
                expected = np.clip(loc + rng.normal(size=(16, 15, 2)) * scale, -1, 1)
                expected[0] = loc
                np.testing.assert_array_equal(expected, batches[i + 1]["actions"])
                elite = np.argsort(batches[i + 1]["costs"], kind="stable")[:4]
                np.testing.assert_array_equal(elite, update["elite"])
                loc, scale = expected[elite].mean(0), np.maximum(expected[elite].std(0), 1e-5)
        else:
            raise ValueError("unknown arm")
    return {"scalar_replays": scalar_replays, "max_path_error": maximum_path_error,
            "max_score_error": maximum_score_error, "max_formula_error": maximum_formula_error}


def audit():
    manifest = run.manifest()
    plans = {run.base.record_name("plan", i, j): run.base.sha(run.OUT / run.base.record_name("plan", i, j))
             for i in range(24) for j in range(2)}
    references = {name: run.base.sha(run.geometry.OUT / name) for name in plans}
    if (run.OUT / "audit.json").exists():
        previous = run.load("audit.json")
        if previous["plans"] != plans or previous["prior_geometry_plans"] != references:
            raise ValueError("audit binding changed")
        print("Verified completed plane-lineage audit.", flush=True)
        return
    totals = {"scalar_replays": 0, "max_path_error": 0., "max_score_error": 0., "max_formula_error": 0.}
    for name in plans:
        row = run.load(name)
        if row["task"] != manifest["tasks"][row["root"]]:
            raise ValueError("task mismatch")
        measured = verify_record(row)
        for key in totals:
            totals[key] = totals[key] + measured[key] if key == "scalar_replays" else max(totals[key], measured[key])
        old = run.geometry.load(name)
        full_batches = run.batches(row, "whole_plane")
        np.testing.assert_array_equal(np.concatenate([b["actions"] for b in full_batches]), old["arms"]["plane_2d"]["actions"])
        np.testing.assert_allclose(np.concatenate([b["costs"] for b in full_batches]), old["arms"]["plane_2d"]["costs"],
                                   atol=1e-12, rtol=0)
        print(f"Audited all arms in {name}", flush=True)
    run.save("audit.json", {
        "plans": plans, "prior_geometry_plans": references, **totals,
        "selected_plans": 48*len(run.ARMS), "all_budgets_nominals_actions_and_updates_verified": True,
        "same_dynamics_second_implementation_not_independent_environment": True,
    })
    print(totals, flush=True)
