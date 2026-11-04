"""First-decision development compiler and matched search controls.

The predictor and its token meter are supplied by the caller. This version
does not provide a multi-decision RGB controller or a PC gate verdict.
"""
import numpy as np

from experiments import pc_successor_rgb_execution_dev_20261103_codex as prior
from experiments.pc_budget_allocation_20261103_codex import allocate_samples
from experiments.spectral_reachability.pc_cem_scale_floor_v1 import optimize
from experiments.spectral_reachability.pc_spectral_bank_planner_20261103_codex import Meter

ARMS = ("successor", "same_bank_greedy", "random_bridge", "lookahead", "full_cem")


def decide(arm, root, target, predict, token_meter):
    if arm not in ARMS or token_meter.token_samples != 0:
        raise ValueError("invalid arm or nonempty meter")
    cap = prior.policy.CAP
    target = np.asarray(target)
    structure = Meter({"prediction_samples": 63, "svd": 3, "linear_solve": 0})
    first, bank, evidence, successor_evidence, programs, prefix_scores = [], [], None, [], [], []
    seed = (2026120800 if arm == "successor" else 2026121300) + root
    reserve = 3
    if arm == "full_cem":
        count = (cap - reserve - token_meter.token_samples) // 3
        chain, score, scores = optimize([], 15, count, np.random.default_rng(seed), lambda x: predict(x)[1])
        selected = {"name": "full_cem", "chain": chain.tolist(), "terminal_score": score,
                    "samples": count, "scores": scores}
    else:
        first, evidence = prior.policy.generator.generate(
            lambda a: predict(np.asarray(a)[None])[0][0], target, prior.NOMINALS, structure, mode="svd")
        if arm == "successor":
            for item in first:
                prefix = np.asarray(item["prefix"])
                second, response = prior.policy.generator.generate(
                    lambda a: predict(np.concatenate((prefix, np.asarray(a)))[None])[0][0],
                    target, prior.NOMINALS, structure, mode="svd")
                successor_evidence.append(response)
                for item2 in second:
                    bank.append({"name": item["name"] + "/" + item2["name"],
                                 "prefix": np.concatenate((prefix, item2["prefix"])).tolist()})
        elif arm == "same_bank_greedy":
            prefix_scores = [float(predict(np.asarray(item["prefix"])[None])[1][0]) for item in first]
            item = first[int(np.argmin(prefix_scores))]
            bank = [{"name": item["name"], "prefix": item["prefix"]}]
        else:
            rng = np.random.default_rng(2026121200 + root)
            for item in first:
                for name, suffix in (("zero", np.zeros((5, 2))), ("random", rng.uniform(-1, 1, (5, 2)))):
                    bank.append({"name": item["name"] + "/" + name,
                                 "prefix": np.concatenate((item["prefix"], suffix)).tolist()})
            if arm == "lookahead":
                prefix_scores = [float(predict(np.asarray(item["prefix"])[None])[1][0]) for item in bank]
                bank = [bank[int(np.argmin(prefix_scores))]]
        affordable = (cap - reserve - token_meter.token_samples) // 3
        counts = allocate_samples(3 * affordable // 4, 0, len(bank), 1)
        if min(counts) < 13:
            raise ValueError("insufficient iterative suffix budget")
        for item, count in zip(bank, counts, strict=True):
            chain, score, scores = optimize(item["prefix"], 15, count, np.random.default_rng(seed),
                                            lambda x: predict(x)[1])
            programs.append({"name": item["name"], "chain": chain.tolist(), "terminal_score": score,
                             "samples": count, "scores": scores})
        selected = min(programs, key=lambda item: item["terminal_score"])
        count = (cap - reserve - token_meter.token_samples) // 3
        chain, score = prior.policy.refine(selected["chain"], selected["terminal_score"], count,
                                           np.random.default_rng(seed + 1), lambda x: predict(x)[1])
        if score < selected["terminal_score"]:
            selected = {"name": "joint_refined", "chain": chain.tolist(), "terminal_score": score,
                        "refined_from": selected["name"], "samples": count}
    before = selected.copy()
    comparison = np.asarray(selected["chain"]).copy()
    if arm != "full_cem":
        comparison[:5] = 0
    comparison_score = float(predict(comparison[None])[1][0])
    abstained = arm != "full_cem" and comparison_score <= selected["terminal_score"]
    if abstained:
        selected = {"name": "zero_prefix_counterfactual", "chain": comparison.tolist(),
                    "terminal_score": comparison_score}
    return {"arm": arm, "first_bank": first, "bank": bank, "generator": evidence,
            "successor_generator": successor_evidence, "prefix_scores": prefix_scores,
            "programs": programs, "precomparison": before, "comparison_chain": comparison.tolist(),
            "comparison_score": comparison_score, "abstained": bool(abstained) if arm != "full_cem" else None,
            "selected": selected, "structure_counts": structure.counts, "structure_seconds": structure.seconds,
            "scale_floor": .05}
