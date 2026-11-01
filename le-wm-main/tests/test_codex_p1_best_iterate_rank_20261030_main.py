import numpy as np

from experiments.spectral_reachability import codex_p1_best_iterate_rank_20261030_main as rank


def test_saved_rows_and_regularized_scores_reconstruct():
    read = rank.cmp.compare.plan.gap.prior.gap.read
    path = rank.base.ROOT / "results/codex_p1_best_iterate_rank_20261030_main_RUN01.json"
    data, _ = read(path, "fff29f94aa51c5cd19cf8f8019e26c514b584faff6735eb0e374533bf594f156")
    entries = {item[0]["index"]: item for item in rank.inputs()}
    original = rank.cmp.compare.plan.inputs()
    trust = read(rank.cmp.TRUST_PLAN, rank.cmp.TRUST_PLAN_SHA)[0]["rows"]
    best = read(rank.cmp.physical.PLAN, rank.cmp.physical.SHA)[0]["rows"]
    assert len(data["rows"]) == 22
    assert sum(not row["same_suffix"] for row in data["rows"]) == 6
    for row in data["rows"]:
        i = row["index"]
        entry = entries[i]
        reference = original[i][2][1]
        for label, packet, plan in zip(("trust", "best"), entry[5], (trust[i], best[i]), strict=True):
            penalty = float(np.mean((packet-reference)**2))
            objective = row[label]["predicted_goal_mse"] + 0.1*penalty
            expected = (min(plan["baseline_score"], plan["optimized_score"])
                        if label == "trust" else plan["selected_score"])
            assert abs(objective-expected) < 1e-5
            assert len(row[label]["predicted_latent"]) == len(row[label]["actual_latent"]) == 192
            pred = np.asarray(row[label]["predicted_latent"])
            actual = np.asarray(row[label]["actual_latent"])
            assert abs(np.mean((pred-actual)**2)-row[label]["prediction_error_mse"]) < 1e-6
