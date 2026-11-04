"""Fixed-terminal feedback composition around the frozen temporal-search compiler."""
import numpy as np

from experiments.pc_budget_allocation_20261103_codex import allocate_samples
from experiments.spectral_reachability import pc_response_temporal_search_v1 as initial

METHOD = "response_temporal_fixed_terminal_dev_v1"


def decide(arm, root, step, target, predict, meter, carry=None):
    if arm not in initial.ARMS or step not in (0, 1, 2) or meter.token_samples:
        raise ValueError("invalid arm, step, or nonempty meter")
    target = np.asarray(target, dtype=float)
    if target.shape != (2,) or not np.isfinite(target).all():
        raise ValueError("invalid target")
    if step == 0:
        if carry is not None:
            raise ValueError("initial decision cannot accept carry")
        return initial.decide(arm, root, target, predict, meter)
    horizon = 15 - 5 * step
    carry = np.asarray(carry, dtype=float)
    if (carry.shape != (horizon, 2) or not np.isfinite(carry).all()
            or np.any(np.abs(carry) > 1)):
        raise ValueError("invalid carry")
    per = horizon // 5
    seed = 2027110400 + root * 100 + step * 10000
    carry_score = float(predict(carry[None])[1][0])
    structure = {"svd": 0, "fd_action_samples": 0, "svd_seconds": 0.0}
    if arm in ("smooth_cem", "action_cem"):
        zero = np.zeros((horizon, 2))
        zero_score = float(predict(zero[None])[1][0])
        count = (initial.CAP - meter.token_samples) // per
        chain, score, generations = initial.cem(zero, [], count, np.random.default_rng(seed + 1),
                                                 predict, zero_score, 4 if arm == "smooth_cem" else horizon)
        if carry_score < score:
            chain, score, source = carry.copy(), carry_score, "carry"
        else:
            source = arm
        return {"selected": {"name": source, "chain": chain.tolist(), "terminal_score": score},
                "carry": carry.tolist(), "carry_score": carry_score, "structure": structure,
                "generations": generations, "abstained": None}
    random_rng = np.random.default_rng(seed + 2) if arm == "matched_random" else None
    first = [{"name": "zero", "prefix": np.zeros((5, 2)).tolist()}]
    evidence = []
    for index, nominal in enumerate(initial.NOMINALS):
        bank, info = initial.response(nominal, [], predict, structure, random_rng)
        evidence.append(info)
        first.extend({"name": f"nominal{index}/{item['name']}", "prefix": item["prefix"]} for item in bank)
    positions, scores = predict(np.asarray([item["prefix"] for item in first]))
    scores = np.asarray(scores)
    programs, retained = [], []
    if step == 1:
        retained = initial.retain(scores, positions)
        seeds = []
        for index in retained:
            prefix = np.asarray(first[index]["prefix"])
            candidates, info = initial.response(initial.NOMINALS[0], prefix, predict, structure, random_rng)
            suffixes = [np.zeros((5, 2))] + [np.asarray(item["prefix"]) for item in candidates]
            joined = np.concatenate((np.broadcast_to(prefix, (len(suffixes), 5, 2)),
                                     np.asarray(suffixes)), axis=1)
            _, values = predict(joined)
            best = int(np.argmin(values))
            seeds.append({"index": index, "suffix": suffixes[best], "score": float(values[best]),
                          "seed_scores": values.tolist(), "successor_response": info})
        available = (initial.CAP - per - meter.token_samples) // per
        counts = allocate_samples(3 * available // 4, 0, len(seeds), 1)
        if not counts or min(counts) < 13:
            raise ValueError("insufficient iterative suffix budget")
        for index, (item, count) in enumerate(zip(seeds, counts, strict=True)):
            chain, score, generations = initial.cem(item["suffix"], first[item["index"]]["prefix"], count,
                np.random.default_rng(seed + 10 + index), predict, item["score"], 4)
            programs.append({"parent_index": item["index"], "chain": chain.tolist(), "terminal_score": score,
                             "samples": count, "generations": generations, "seed_scores": item["seed_scores"],
                             "successor_response": item["successor_response"]})
        best = min(programs, key=lambda entry: entry["terminal_score"])
        best_chain, best_score = np.asarray(best["chain"]), float(best["terminal_score"])
        source = "program"
    else:
        best_index = int(np.argmin(scores))
        best_chain, best_score = np.asarray(first[best_index]["prefix"]), float(scores[best_index])
        source = "first_bank"
    if carry_score < best_score:
        best_chain, best_score, source = carry.copy(), carry_score, "carry"
    count = (initial.CAP - per - meter.token_samples) // per
    if count < 1:
        raise ValueError("no joint refinement budget")
    refined, refined_score, generations = initial.cem(best_chain, [], count, np.random.default_rng(seed + 20),
                                                        predict, best_score, 4)
    before = {"name": "refined" if refined_score < best_score else source,
              "chain": refined.tolist(), "terminal_score": refined_score}
    comparison = refined.copy()
    comparison[:5] = 0
    comparison_score = float(predict(comparison[None])[1][0])
    abstained = comparison_score <= refined_score
    selected = ({"name": "zero_prefix", "chain": comparison.tolist(), "terminal_score": comparison_score}
                if abstained else before)
    return {"selected": selected, "precomparison": before, "comparison_score": comparison_score,
            "abstained": bool(abstained), "carry": carry.tolist(), "carry_score": carry_score,
            "first_bank": first, "prefix_scores": scores.tolist(), "first_response": evidence,
            "retained": retained, "programs": programs, "refinement": generations, "refine_samples": count,
            "structure": structure}
