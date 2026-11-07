"""Verify nominal retention and complete query trace before any pool evaluation."""
import numpy as np

from experiments.pc_cross_query.audit import scalar_trajectory
from experiments.pc_lineage_cross import core, run


def audit():
    run.manifest()
    plans = {run.old.record_name("plan", root, repeat): run.old.sha(run.OUT / run.old.record_name("plan", root, repeat))
             for root in range(24) for repeat in range(2)}
    if (run.OUT / "audit.json").exists():
        if run.load("audit.json")["plans"] != plans:
            raise ValueError("audited plan drift")
        print("verified completed audit", flush=True)
        return
    max_score_error = max_scalar_error = max_formula_error = 0.
    for name in plans:
        row = run.load(name)
        nominals, offsets = run.source_chains(row["task"], row["seed"])
        np.testing.assert_array_equal(nominals, row["nominals"])
        np.testing.assert_array_equal(offsets, row["offsets"])
        bank = core.PairedBank(nominals)
        observed = np.asarray(row["observed"], dtype=float)
        mask = core.support(bank.shape, row["anchors"])
        core.check_observed(observed, row["anchors"])
        np.testing.assert_array_equal(core.anchors(observed[:, :4]), row["anchors"])
        # Independently solve the cross interpolation coefficients, without dense truth.
        baseline = observed[:, :1] + (observed[row["anchors"][0]] - observed[row["anchors"][0], 0])[None]
        da = observed[:, :4] - baseline[:, :4]
        di = observed[row["anchors"]] - baseline[row["anchors"]]
        fitted = baseline + np.linalg.lstsq(da[row["anchors"]].T, da.T, rcond=1e-8)[0].T @ di
        fitted[mask] = observed[mask]
        max_formula_error = max(max_formula_error, float(np.max(np.abs(
            fitted - row["outcomes"]["cross_interaction"]["prediction"]))))
        np.testing.assert_allclose(fitted, row["outcomes"]["cross_interaction"]["prediction"], atol=1e-10, rtol=1e-10)
        for arm, result in row["outcomes"].items():
            batches = run.batches(row, arm)
            np.testing.assert_array_equal(batches[0]["actions"], nominals)
            costs, actions, pair_list = [], [], []
            for batch in batches:
                act = np.asarray(batch["actions"])
                end = run.old.previous.base.exact.endpoints(row["task"], act)
                score = np.linalg.norm(end - row["task"]["goal"], axis=1)
                max_score_error = max(max_score_error, float(np.max(np.abs(score - batch["costs"]))))
                np.testing.assert_allclose(score, batch["costs"], atol=1e-12, rtol=0)
                np.testing.assert_allclose(end, batch["endpoints"], atol=1e-12, rtol=0)
                if batch["pairs"] is not None:
                    np.testing.assert_array_equal(act, bank.compose(batch["pairs"]))
                    pair_list.extend(batch["pairs"])
                costs.extend(batch["costs"])
                actions.extend(batch["actions"])
            if len(costs) != 128 or result["queries"] != 128:
                raise ValueError("query budget drift")
            best = int(np.argmin(costs))
            if result["choice"]["query_index"] != best or costs[best] != result["choice"]["score"]:
                raise ValueError("measured incumbent lost")
            if costs[best] > min(costs[:16]):
                raise ValueError("nominal witness lost")
            np.testing.assert_array_equal(actions[best], result["choice"]["chain"])
            selected = np.asarray(actions[best])
            path, _ = run.old.trajectory(row["task"]["start"], selected[None])
            scalar = scalar_trajectory(row["task"]["start"], selected)
            max_scalar_error = max(max_scalar_error, float(np.max(np.abs(path[0] - scalar))))
            np.testing.assert_allclose(path[0], scalar, atol=1e-12, rtol=0)
            if result["uses_shared_cross"]:
                if len(np.unique(np.asarray(pair_list), axis=0)) != 128:
                    raise ValueError("duplicated pair queries")
                proposal_pairs = np.asarray(result["proposal_pairs"])
                if mask[proposal_pairs[:, 0], proposal_pairs[:, 1]].any():
                    raise ValueError("proposals reused observed cells")
                if arm != "random_missing":
                    expected, _ = core.prediction(arm, observed, row["anchors"], row["seed"] + 20000)
                    np.testing.assert_allclose(expected, result["prediction"], atol=1e-12, rtol=0)
                    np.testing.assert_array_equal(core.choose(expected, mask, 128 - row["shared_queries"]), proposal_pairs)
            if arm == "nominal_cem":
                updates = result["updates"]
                np.testing.assert_array_equal(updates[0]["mean"], nominals[np.argmin(costs[:16])])
                for i, update in enumerate(updates):
                    batch = batches[i + 1]
                    samples, values = np.asarray(batch["actions"]), np.asarray(batch["costs"])
                    order = np.argsort(values, kind="stable")[:4]
                    np.testing.assert_array_equal(order, update["elite"])
                    np.testing.assert_array_equal(samples[0], update["mean"])
                    if i < 6:
                        np.testing.assert_allclose(samples[order].mean(0), updates[i + 1]["mean"], atol=1e-12, rtol=0)
                        np.testing.assert_allclose(np.maximum(samples[order].std(0), 1e-5),
                                                   updates[i + 1]["std"], atol=1e-12, rtol=0)
        print(f"audited {name}", flush=True)
    run.save("audit.json", {
        "plans": plans, "selected_plans": 48 * len(run.ARMS),
        "max_score_error": max_score_error, "max_scalar_error": max_scalar_error,
        "max_pinv_lstsq_error": max_formula_error,
        "all_nominals_counted_preserved_and_actions_verified": True,
        "independent_environment": False,
    })
