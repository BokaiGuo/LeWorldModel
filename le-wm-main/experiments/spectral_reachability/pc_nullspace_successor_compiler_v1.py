"""Response-nullspace exploration with matched extra-direction controls."""
import numpy as np

from experiments.pc_budget_allocation_20261103_codex import allocate_samples
from experiments.spectral_reachability import pc_goal_aligned_successor_compiler_v1 as old
from experiments.spectral_reachability.pc_cem_scale_floor_v1 import optimize
from experiments.spectral_reachability.pc_spectral_bank_planner_20261103_codex import Meter

ARMS = ("null", "random_extra", "full_cem")
CAP = 480


def extra_directions(rng, basis, mode, count=2):
    basis = np.asarray(basis, dtype=float).reshape(-1, 10)
    extras, attempts = [], []
    for _ in range(count):
        for trial in range(8):
            vector = rng.normal(size=10)
            if mode == "null":
                for row in basis:
                    vector -= (vector @ row) * row
            elif mode != "random_extra":
                raise ValueError("invalid direction mode")
            for row in extras:
                vector -= (vector @ row) * row
            magnitude = float(np.linalg.norm(vector))
            if magnitude > 1e-10:
                extras.append(vector / magnitude)
                attempts.append(trial + 1)
                break
        else:
            attempts.append(8)
    return np.asarray(extras).reshape(-1, 10), attempts


def measured_bank(mode, target, predict, meter, nominal, rng):
    _, evidence = old.directions("goal", target, predict, meter, nominal, rng)
    basis = np.asarray(evidence["proposal_directions"], dtype=float).reshape(-1, 10)
    extra, attempts = extra_directions(rng, basis, mode)
    directions = np.concatenate((basis, extra), axis=0)
    evidence.update({"extra_mode": mode, "extra_directions": extra.tolist(), "extra_attempts": attempts})
    return old.base.previous.first_bank(nominal, directions, mode), evidence


def decide(arm, root, target, predict, token_meter):
    if arm not in ARMS or token_meter.token_samples != 0:
        raise ValueError("invalid arm or nonempty meter")
    target = np.asarray(target, dtype=float)
    seed = 2032110300 + root
    best_chain, best_score = None, float("inf")

    def measured(actions):
        nonlocal best_chain, best_score
        actions = np.asarray(actions, dtype=float)
        positions, scores = predict(actions)
        scores = np.asarray(scores, dtype=float)
        if actions.ndim != 3 or scores.shape != (len(actions),) or not np.isfinite(scores).all():
            raise ValueError("invalid predictions")
        if actions.shape[1] == 15:
            index = int(np.argmin(scores))
            if scores[index] < best_score:
                best_chain, best_score = actions[index].copy(), float(scores[index])
        return positions, scores

    if arm == "full_cem":
        chain, score, values = optimize([], 15, CAP // 3, np.random.default_rng(seed), lambda x: measured(x)[1])
        return {"selected": {"name": "full_cem", "chain": chain.tolist(), "terminal_score": score},
                "programs": [{"samples": CAP // 3, "scores": values}], "structure_counts": {},
                "structure_seconds": {}, "abstained": None}
    meter = Meter({"prediction_samples": 63, "svd": 3, "linear_solve": 0})
    nominal = old.base.prior.NOMINALS[0].copy()
    rng = np.random.default_rng(seed + 1000)
    first, first_response = measured_bank(arm, target, measured, meter, nominal, rng)
    positions, scores = [], []
    for item in first:
        pos, value = measured(np.asarray(item["prefix"])[None])
        positions.append(pos[0].tolist())
        scores.append(float(value[0]))
    parents = old.base.retain(scores, positions)
    bank, successors = [], []
    for index in parents:
        prefix = np.asarray(first[index]["prefix"])

        def successor_predict(actions):
            actions = np.asarray(actions)
            return measured(np.concatenate((np.broadcast_to(prefix, (len(actions), 5, 2)), actions), axis=1))

        second, evidence = measured_bank(arm, target, successor_predict, meter, nominal, rng)
        second_positions, second_scores = [], []
        for item in second:
            pos, value = successor_predict(np.asarray(item["prefix"])[None])
            second_positions.append(pos[0].tolist())
            second_scores.append(float(value[0]))
        retained = old.base.retain(second_scores, second_positions)
        successors.append({"parent_index": index, "response": evidence, "bank": second,
                           "positions": second_positions, "scores": second_scores, "retained": retained})
        for j in retained:
            bank.append({"name": first[index]["name"] + "/" + second[j]["name"],
                         "prefix": np.concatenate((prefix, second[j]["prefix"])).tolist(),
                         "parent_index": index, "second_index": j})
    affordable = (CAP - 3 - token_meter.token_samples) // 3
    counts = allocate_samples(3 * affordable // 4, 0, len(bank), 1)
    if not counts or min(counts) < 13:
        raise ValueError("insufficient iterative suffix budget")
    programs = []
    for item, count in zip(bank, counts, strict=True):
        chain, score, values = optimize(item["prefix"], 15, count, np.random.default_rng(seed),
                                         lambda x: measured(x)[1])
        programs.append({"name": item["name"], "chain": chain.tolist(), "terminal_score": score,
                         "samples": count, "scores": values})
    count = (CAP - 3 - token_meter.token_samples) // 3
    if count:
        old.base.prior.policy.refine(best_chain, best_score, count, np.random.default_rng(seed + 1),
                                     lambda x: measured(x)[1])
    before = {"chain": best_chain.tolist(), "terminal_score": best_score}
    comparison = best_chain.copy()
    comparison[:5] = 0
    comparison_score = float(measured(comparison[None])[1][0])
    abstained = comparison_score <= before["terminal_score"]
    selected = {"name": "zero_prefix_counterfactual" if abstained else "global_incumbent",
                "chain": comparison.tolist() if abstained else before["chain"],
                "terminal_score": comparison_score if abstained else before["terminal_score"]}
    return {"first_bank": first, "first_positions": positions, "prefix_scores": scores, "retained": parents,
            "bank": bank, "response": first_response, "successor_response": successors, "programs": programs,
            "precomparison": before, "comparison_chain": comparison.tolist(), "comparison_score": comparison_score,
            "abstained": bool(abstained), "selected": selected,
            "structure_counts": meter.counts, "structure_seconds": meter.seconds}
