"""Budgeted multi-start continuation of the multidirection successor compiler."""
import numpy as np

from experiments.pc_budget_allocation_20261103_codex import allocate_samples
from experiments.spectral_reachability import pc_multidirection_successor_compiler_v1 as core
from experiments.spectral_reachability.pc_cem_scale_floor_v1 import optimize
from experiments.spectral_reachability.pc_spectral_bank_planner_20261103_codex import Meter


def decide(arm, root, target, predict, token_meter, cap):
    if arm not in ("multistart", "full_cem") or cap not in (480, 1920) or token_meter.token_samples:
        raise ValueError("invalid arm, budget or initial meter")
    best_chain, best_score = None, float("inf")

    def measured(actions):
        nonlocal best_chain, best_score
        actions = np.asarray(actions, dtype=float)
        pos, values = predict(actions)
        values = np.asarray(values)
        if actions.shape[1] == 15:
            index = int(np.argmin(values))
            if values[index] < best_score:
                best_chain, best_score = actions[index].copy(), float(values[index])
        return pos, values

    seed = 2028110300 + 100 * root
    programs, evidence, bank = [], None, []
    if arm == "full_cem":
        counts = allocate_samples(cap, 0, 4, 3)
        for index, count in enumerate(counts):
            chain, score, scores = optimize([], 15, count, np.random.default_rng(seed + index),
                                            lambda x: measured(x)[1])
            programs.append({"name": f"full_cem_{index}", "chain": chain.tolist(),
                             "terminal_score": score, "samples": count, "scores": scores})
        return {"selected": {"name": "global_incumbent", "chain": best_chain.tolist(),
                             "terminal_score": best_score}, "programs": programs, "structure_counts": {},
                "structure_seconds": {}, "abstained": None}

    meter = Meter({"prediction_samples": 63, "svd": 3, "linear_solve": 0})
    nominal = core.prior.NOMINALS[0].copy()
    first, evidence = core.measured_bank(np.asarray(target), measured, meter, nominal)
    first_positions, first_scores = [], []
    for item in first:
        pos, score = measured(np.asarray(item["prefix"])[None])
        first_positions.append(pos[0].tolist())
        first_scores.append(float(score[0]))
    parents = core.retain(first_scores, first_positions)
    successors = []
    for index in parents:
        prefix = np.asarray(first[index]["prefix"])

        def successor_predict(actions):
            actions = np.asarray(actions)
            return measured(np.concatenate((np.broadcast_to(prefix, (len(actions), 5, 2)), actions), axis=1))

        second, response = core.measured_bank(np.asarray(target), successor_predict, meter, nominal)
        positions, scores = [], []
        for item in second:
            pos, score = successor_predict(np.asarray(item["prefix"])[None])
            positions.append(pos[0].tolist())
            scores.append(float(score[0]))
        retained = core.retain(scores, positions)
        successors.append({"parent_index": index, "response": response, "bank": second,
                           "positions": positions, "scores": scores, "retained": retained})
        for j in retained:
            bank.append({"name": first[index]["name"] + "/" + second[j]["name"],
                         "prefix": np.concatenate((prefix, second[j]["prefix"])).tolist(),
                         "parent_index": index, "second_index": j})
    affordable = (cap - 3 - token_meter.token_samples) // 3
    counts = allocate_samples(3 * affordable // 4, 0, len(bank), 1)
    if not counts or min(counts) < 13:
        raise ValueError("insufficient iterative suffix budget")
    for index, (item, count) in enumerate(zip(bank, counts, strict=True)):
        starts = min(4, count // 13)
        for restart, size in enumerate(allocate_samples(count, 0, starts, 1)):
            chain, score, scores = optimize(item["prefix"], 15, size,
                                            np.random.default_rng(seed + 1000 + 10 * index + restart),
                                            lambda x: measured(x)[1])
            programs.append({"name": item["name"], "restart": restart, "chain": chain.tolist(),
                             "terminal_score": score, "samples": size, "scores": scores})
    count = (cap - 3 - token_meter.token_samples) // 3
    if count:
        core.prior.policy.refine(best_chain, best_score, count, np.random.default_rng(seed + 9000),
                                 lambda x: measured(x)[1])
    before = {"chain": best_chain.tolist(), "terminal_score": best_score}
    comparison = best_chain.copy()
    comparison[:5] = 0
    comparison_score = float(measured(comparison[None])[1][0])
    abstained = comparison_score <= before["terminal_score"]
    selected = {"name": "zero_prefix_counterfactual" if abstained else "global_incumbent",
                "chain": comparison.tolist() if abstained else before["chain"],
                "terminal_score": comparison_score if abstained else before["terminal_score"]}
    return {"first_bank": first, "first_positions": first_positions, "prefix_scores": first_scores,
            "retained": parents, "bank": bank, "response": evidence, "successor_response": successors,
            "programs": programs, "precomparison": before, "comparison_chain": comparison.tolist(),
            "comparison_score": comparison_score, "abstained": bool(abstained), "selected": selected,
            "structure_counts": meter.counts, "structure_seconds": meter.seconds}
