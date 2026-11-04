"""V1 feedback with a selection-only final-horizon incumbent repair."""
import numpy as np

from experiments.spectral_reachability import pc_response_diverse_feedback_v1 as v1
from experiments.spectral_reachability.pc_cem_scale_floor_v1 import optimize
from experiments.spectral_reachability.pc_spectral_bank_planner_20261103_codex import Meter

ARMS = v1.ARMS


def decide(arm, root, step, target, predict, token_meter, carry=None):
    if step != 2 or arm != "response_diverse":
        return v1.decide(arm, root, step, target, predict, token_meter, carry)
    if token_meter.token_samples != 0:
        raise ValueError("nonempty meter")
    carry = np.asarray(carry, dtype=float)
    if carry.shape != (5, 2) or not np.isfinite(carry).all() or np.any(np.abs(carry) > 1):
        raise ValueError("invalid previous plan tail")
    best_chain, best_score = None, float("inf")

    def measured(actions):
        nonlocal best_chain, best_score
        actions = np.asarray(actions, dtype=float)
        positions, values = predict(actions)
        values = np.asarray(values, dtype=float)
        if actions.ndim != 3 or actions.shape[1:] != (5, 2) or values.shape != (len(actions),):
            raise ValueError("invalid full-horizon prediction")
        if not np.isfinite(actions).all() or np.any(np.abs(actions) > 1) or not np.isfinite(values).all():
            raise ValueError("invalid candidate")
        for chain, score in zip(actions, values, strict=True):
            if score < best_score:
                best_chain, best_score = chain.copy(), float(score)
        return positions, values

    carry_score = float(measured(carry[None])[1][0])
    seed = 2026121900 + 10 * root + step
    meter = Meter({"prediction_samples": 21, "svd": 1, "linear_solve": 0})
    nominal = v1.core.prior.NOMINALS[0].copy()
    directions, evidence = v1.core.directions_from_response(np.asarray(target), measured, meter, nominal)
    bank = v1.core.first_bank(nominal, directions, "response")
    prefix_scores = [float(measured(np.asarray(item["prefix"])[None])[1][0]) for item in bank]
    programs = [{"name": item["name"], "chain": item["prefix"], "terminal_score": score,
                 "samples": 0, "scores": []} for item, score in zip(bank, prefix_scores, strict=True)]
    affordable = v1.CAP - 1 - token_meter.token_samples
    count = 3 * affordable // 4
    chain, score, scores = optimize([], 5, count, np.random.default_rng(seed), lambda x: measured(x)[1])
    programs.append({"name": "full_cem_candidate", "chain": chain.tolist(), "terminal_score": score,
                     "samples": count, "scores": scores})
    selected = min(programs, key=lambda x: x["terminal_score"])
    refine_count = v1.CAP - 1 - token_meter.token_samples
    chain, score = v1.core.prior.policy.refine(selected["chain"], selected["terminal_score"], refine_count,
                                               np.random.default_rng(seed + 1), lambda x: measured(x)[1])
    if score < selected["terminal_score"]:
        selected = {"name": "joint_refined", "chain": chain.tolist(), "terminal_score": score,
                    "refined_from": selected["name"], "samples": refine_count}
    if carry_score < selected["terminal_score"]:
        selected = {"name": "carry", "chain": carry.tolist(), "terminal_score": carry_score}
    if best_score < selected["terminal_score"]:
        selected = {"name": "scored_incumbent", "chain": best_chain.tolist(), "terminal_score": best_score}
    before = selected.copy()
    comparison = np.zeros((5, 2))
    comparison_score = float(measured(comparison[None])[1][0])
    abstained = comparison_score <= before["terminal_score"]
    if abstained:
        selected = {"name": "zero_prefix_counterfactual", "chain": comparison.tolist(),
                    "terminal_score": comparison_score}
    return {"arm": arm, "horizon": 5, "first_bank": bank, "prefix_scores": prefix_scores,
            "response": evidence, "programs": programs, "carry": carry.tolist(), "carry_score": carry_score,
            "precomparison": before, "comparison_chain": comparison.tolist(), "comparison_score": comparison_score,
            "abstained": bool(abstained), "selected": selected, "structure_counts": meter.counts,
            "structure_seconds": meter.seconds}
