"""Fixed-terminal continuation of the multidirection successor compiler."""
import numpy as np

from experiments.pc_budget_allocation_20261103_codex import allocate_samples
from experiments.spectral_reachability import pc_multidirection_successor_compiler_v1 as core
from experiments.spectral_reachability.pc_cem_scale_floor_v1 import optimize
from experiments.spectral_reachability.pc_spectral_bank_planner_20261103_codex import Meter

CAP = 480


def decide(root, step, target, predict, token_meter, carry):
    if step not in (1, 2) or token_meter.token_samples != 0:
        raise ValueError("invalid step or nonempty meter")
    horizon = 15 - 5 * step
    per = horizon // 5
    carry = np.asarray(carry, dtype=float)
    if carry.shape != (horizon, 2) or not np.isfinite(carry).all() or np.any(np.abs(carry) > 1):
        raise ValueError("invalid carry")
    best_chain, best_score = None, float("inf")

    def measured(actions):
        nonlocal best_chain, best_score
        actions = np.asarray(actions, dtype=float)
        positions, scores = predict(actions)
        scores = np.asarray(scores, dtype=float)
        if actions.ndim != 3 or actions.shape[2:] != (2,) or scores.shape != (len(actions),):
            raise ValueError("invalid predictions")
        if not np.isfinite(actions).all() or np.any(np.abs(actions) > 1) or not np.isfinite(scores).all():
            raise ValueError("invalid actions or scores")
        if actions.shape[1] == horizon:
            for chain, score in zip(actions, scores, strict=True):
                if score < best_score:
                    best_chain, best_score = chain.copy(), float(score)
        return positions, scores

    carry_score = float(measured(carry[None])[1][0])
    meter = Meter({"prediction_samples": 21, "svd": 1, "linear_solve": 0})
    nominal = core.prior.NOMINALS[0].copy()
    bank, response = core.measured_bank(np.asarray(target), measured, meter, nominal)
    scores = [float(measured(np.asarray(item["prefix"])[None])[1][0]) for item in bank]
    seed = 2027060100 + 10 * root + step
    remaining = (CAP - per - token_meter.token_samples) // per
    allocation = 3 * remaining // 4
    programs = []
    if horizon == 10:
        counts = allocate_samples(allocation, 0, len(bank), 1)
        if not counts or min(counts) < 13:
            raise ValueError("insufficient iterative suffix budget")
        for item, count in zip(bank, counts, strict=True):
            chain, score, history = optimize(item["prefix"], horizon, count,
                                             np.random.default_rng(seed), lambda x: measured(x)[1])
            programs.append({"name": item["name"], "chain": chain.tolist(), "terminal_score": score,
                             "samples": count, "scores": history})
    else:
        programs = [{"name": item["name"], "chain": item["prefix"], "terminal_score": score,
                     "samples": 0} for item, score in zip(bank, scores, strict=True)]
        chain, score, history = optimize([], horizon, allocation, np.random.default_rng(seed),
                                         lambda x: measured(x)[1])
        programs.append({"name": "full_cem_candidate", "chain": chain.tolist(), "terminal_score": score,
                         "samples": allocation, "scores": history})
    selected = min(programs, key=lambda item: item["terminal_score"])
    refine_count = (CAP - per - token_meter.token_samples) // per
    if refine_count:
        chain, score = core.prior.policy.refine(selected["chain"], selected["terminal_score"], refine_count,
                                                np.random.default_rng(seed + 1), lambda x: measured(x)[1])
        if score < selected["terminal_score"]:
            selected = {"name": "joint_refined", "chain": chain.tolist(), "terminal_score": score,
                        "refined_from": selected["name"], "samples": refine_count}
    if best_score < selected["terminal_score"]:
        selected = {"name": "scored_incumbent", "chain": best_chain.tolist(), "terminal_score": best_score}
    before = selected.copy()
    comparison = np.asarray(selected["chain"]).copy()
    comparison[:5] = 0
    comparison_score = float(measured(comparison[None])[1][0])
    abstained = comparison_score <= before["terminal_score"]
    if abstained:
        selected = {"name": "zero_prefix_counterfactual", "chain": comparison.tolist(),
                    "terminal_score": comparison_score}
    return {"horizon": horizon, "first_bank": bank, "prefix_scores": scores, "response": response,
            "programs": programs, "carry": carry.tolist(), "carry_score": carry_score,
            "precomparison": before, "comparison_chain": comparison.tolist(), "comparison_score": comparison_score,
            "abstained": bool(abstained), "selected": selected, "structure_counts": meter.counts,
            "structure_seconds": meter.seconds}
