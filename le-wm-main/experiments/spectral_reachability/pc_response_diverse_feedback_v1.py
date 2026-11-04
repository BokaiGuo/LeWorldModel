"""Fixed-terminal feedback adapter for the response-diverse compiler."""
import numpy as np

from experiments.pc_budget_allocation_20261103_codex import allocate_samples
from experiments.spectral_reachability import pc_response_diverse_compiler_v1 as core
from experiments.spectral_reachability.pc_cem_scale_floor_v1 import optimize
from experiments.spectral_reachability.pc_spectral_bank_planner_20261103_codex import Meter

ARMS = ("response_diverse", "full_cem")
CAP = 480


def decide(arm, root, step, target, predict, token_meter, carry=None):
    if arm not in ARMS or step not in (0, 1, 2) or token_meter.token_samples != 0:
        raise ValueError("invalid arm, step, or meter")
    if step == 0:
        if carry is not None:
            raise ValueError("unexpected carry")
        result = core.decide(arm, root, target, predict, token_meter)
        result.update({"horizon": 15, "carry": None, "carry_score": None})
        return result

    horizon = 15 - 5 * step
    per = horizon // 5
    carry = np.asarray(carry, dtype=float)
    if carry.shape != (horizon, 2) or not np.isfinite(carry).all() or np.any(np.abs(carry) > 1):
        raise ValueError("invalid previous plan tail")
    carry_score = float(predict(carry[None])[1][0])
    seed = 2026121900 + 10 * root + step
    meter = Meter({"prediction_samples": 21, "svd": 1, "linear_solve": 0})
    bank, programs, evidence, prefix_scores = [], [], None, []
    if arm == "full_cem":
        count = (CAP - per - token_meter.token_samples) // per
        chain, score, scores = optimize([], horizon, count, np.random.default_rng(seed), lambda x: predict(x)[1])
        selected = {"name": "full_cem", "chain": chain.tolist(), "terminal_score": score,
                    "samples": count, "scores": scores}
        programs.append(selected)
    else:
        nominal = core.prior.NOMINALS[0].copy()
        directions, evidence = core.directions_from_response(np.asarray(target), predict, meter, nominal)
        bank = core.first_bank(nominal, directions, "response")
        for item in bank:
            prefix_scores.append(float(predict(np.asarray(item["prefix"])[None])[1][0]))
        affordable = (CAP - per - token_meter.token_samples) // per
        suffix_total = 3 * affordable // 4
        if horizon == 10:
            counts = allocate_samples(suffix_total, 0, len(bank), 1)
            if min(counts) < 13:
                raise ValueError("insufficient iterative suffix budget")
            for item, count in zip(bank, counts, strict=True):
                chain, score, scores = optimize(item["prefix"], horizon, count,
                                                np.random.default_rng(seed), lambda x: predict(x)[1])
                programs.append({"name": item["name"], "chain": chain.tolist(), "terminal_score": score,
                                 "samples": count, "scores": scores})
        else:
            programs = [{"name": item["name"], "chain": item["prefix"], "terminal_score": score,
                         "samples": 0, "scores": []} for item, score in zip(bank, prefix_scores, strict=True)]
            chain, score, scores = optimize([], horizon, suffix_total, np.random.default_rng(seed),
                                            lambda x: predict(x)[1])
            programs.append({"name": "full_cem_candidate", "chain": chain.tolist(), "terminal_score": score,
                             "samples": suffix_total, "scores": scores})
        selected = min(programs, key=lambda x: x["terminal_score"])
        count = (CAP - per - token_meter.token_samples) // per
        chain, score = core.prior.policy.refine(selected["chain"], selected["terminal_score"], count,
                                               np.random.default_rng(seed + 1), lambda x: predict(x)[1])
        if score < selected["terminal_score"]:
            selected = {"name": "joint_refined", "chain": chain.tolist(), "terminal_score": score,
                        "refined_from": selected["name"], "samples": count}
    if carry_score < selected["terminal_score"]:
        selected = {"name": "carry", "chain": carry.tolist(), "terminal_score": carry_score}
    before = selected.copy()
    comparison = np.asarray(selected["chain"]).copy()
    if arm != "full_cem":
        comparison[:5] = 0
    comparison_score = float(predict(comparison[None])[1][0])
    abstained = arm != "full_cem" and comparison_score <= before["terminal_score"]
    if abstained:
        selected = {"name": "zero_prefix_counterfactual", "chain": comparison.tolist(),
                    "terminal_score": comparison_score}
    return {"arm": arm, "horizon": horizon, "first_bank": bank, "prefix_scores": prefix_scores,
            "response": evidence, "programs": programs, "carry": carry.tolist(), "carry_score": carry_score,
            "precomparison": before, "comparison_chain": comparison.tolist(), "comparison_score": comparison_score,
            "abstained": bool(abstained) if arm != "full_cem" else None, "selected": selected,
            "structure_counts": meter.counts, "structure_seconds": meter.seconds}
