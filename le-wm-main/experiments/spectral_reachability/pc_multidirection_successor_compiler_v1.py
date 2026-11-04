"""Budgeted multi-direction successor compiler for exact-dynamics development."""
import numpy as np

from experiments import pc_successor_rgb_execution_dev_20261103_codex as prior
from experiments.pc_budget_allocation_20261103_codex import allocate_samples
from experiments.spectral_reachability import pc_response_diverse_compiler_v1 as previous
from experiments.spectral_reachability.pc_cem_scale_floor_v1 import optimize
from experiments.spectral_reachability.pc_spectral_bank_planner_20261103_codex import Meter


def retain(scores, positions):
    scores, positions = np.asarray(scores), np.asarray(positions)
    if scores.ndim != 1 or len(scores) == 0 or len(positions) != len(scores):
        raise ValueError("invalid candidates")
    best = int(np.argmin(scores))
    if len(scores) == 1:
        return [best]
    distance = np.linalg.norm(positions - positions[best], axis=1)
    distance[best] = -np.inf
    return [best, int(np.argmax(distance))]


def measured_bank(target, predict, meter, nominal):
    directions, evidence = previous.directions_from_response(target, predict, meter, nominal)
    singular = np.asarray(evidence["singular"])
    valid = [d for d, s in zip(directions, singular[:len(directions)], strict=True) if s > 1e-10]
    bank = previous.first_bank(nominal, valid, "response")
    evidence["retained_rank"] = len(valid)
    return bank, evidence


def decide(root, target, predict, token_meter):
    if token_meter.token_samples != 0:
        raise ValueError("nonempty meter")
    target = np.asarray(target, dtype=float)
    cap, reserve, seed = prior.policy.CAP, 3, 2027050100 + root
    meter = Meter({"prediction_samples": 63, "svd": 3, "linear_solve": 0})
    nominal = prior.NOMINALS[0].copy()
    first, first_response = measured_bank(target, predict, meter, nominal)
    positions, scores = [], []
    for item in first:
        pos, value = predict(np.asarray(item["prefix"])[None])
        positions.append(np.asarray(pos[0]))
        scores.append(float(value[0]))
    parents = retain(scores, positions)
    bank, successors = [], []
    for index in parents:
        prefix = np.asarray(first[index]["prefix"])

        def successor_predict(actions):
            actions = np.asarray(actions)
            joined = np.concatenate((np.broadcast_to(prefix, (len(actions), 5, 2)), actions), axis=1)
            return predict(joined)

        second, evidence = measured_bank(target, successor_predict, meter, nominal)
        second_positions, second_scores = [], []
        for candidate in second:
            pos, value = successor_predict(np.asarray(candidate["prefix"])[None])
            second_positions.append(np.asarray(pos[0]))
            second_scores.append(float(value[0]))
        retained = retain(second_scores, second_positions)
        successors.append({"parent_index": index, "response": evidence, "bank": second,
                           "positions": [x.tolist() for x in second_positions],
                           "scores": second_scores, "retained": retained})
        for j in retained:
            bank.append({"name": first[index]["name"] + "/" + second[j]["name"],
                         "parent_index": index, "second_index": j,
                         "prefix": np.concatenate((prefix, second[j]["prefix"])).tolist()})
    affordable = (cap - reserve - token_meter.token_samples) // 3
    counts = allocate_samples(3 * affordable // 4, 0, len(bank), 1)
    if not counts or min(counts) < 13:
        raise ValueError("insufficient iterative search budget")
    programs = []
    for item, count in zip(bank, counts, strict=True):
        chain, score, scores_out = optimize(item["prefix"], 15, count, np.random.default_rng(seed),
                                            lambda x: predict(x)[1])
        programs.append({"name": item["name"], "chain": chain.tolist(), "terminal_score": score,
                         "samples": count, "scores": scores_out})
    selected = min(programs, key=lambda x: x["terminal_score"])
    count = (cap - reserve - token_meter.token_samples) // 3
    if count:
        chain, score = prior.policy.refine(selected["chain"], selected["terminal_score"], count,
                                           np.random.default_rng(seed + 1), lambda x: predict(x)[1])
        if score < selected["terminal_score"]:
            selected = {"name": "joint_refined", "chain": chain.tolist(), "terminal_score": score,
                        "refined_from": selected["name"], "samples": count}
    before = selected.copy()
    comparison = np.asarray(selected["chain"]).copy()
    comparison[:5] = 0
    comparison_score = float(predict(comparison[None])[1][0])
    abstained = comparison_score <= before["terminal_score"]
    if abstained:
        selected = {"name": "zero_prefix_counterfactual", "chain": comparison.tolist(),
                    "terminal_score": comparison_score}
    return {"arm": "multidirection_successor", "first_bank": first,
            "first_positions": [x.tolist() for x in positions], "prefix_scores": scores,
            "retained": parents, "bank": bank, "response": first_response,
            "successor_response": successors, "programs": programs, "precomparison": before,
            "comparison_chain": comparison.tolist(), "comparison_score": comparison_score,
            "abstained": bool(abstained), "selected": selected, "structure_counts": meter.counts,
            "structure_seconds": meter.seconds}
