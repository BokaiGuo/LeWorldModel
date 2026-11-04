"""Fixed-terminal compiler retaining all scored full chains before joint CEM."""
import numpy as np

from experiments.pc_budget_allocation_20261103_codex import allocate_samples
from experiments.spectral_reachability import pc_response_temporal_search_v1 as core

METHOD = "response_temporal_fixed_terminal_dev_v3"


def decide(arm, root, step, target, predict, meter, carry=None):
    if arm not in core.ARMS or step not in (0, 1, 2) or meter.token_samples:
        raise ValueError("invalid arm, step or nonempty meter")
    target = np.asarray(target, dtype=float)
    if target.shape != (2,) or not np.isfinite(target).all():
        raise ValueError("invalid target")
    if step == 0:
        if carry is not None:
            raise ValueError("initial decision cannot accept carry")
    else:
        carry = np.asarray(carry, dtype=float)
        if (carry.shape != (15 - 5 * step, 2) or not np.isfinite(carry).all()
                or np.any(np.abs(carry) > 1)):
            raise ValueError("invalid carry")
    if arm in ("smooth_cem", "action_cem"):
        from experiments.spectral_reachability import pc_response_temporal_feedback_v1
        return pc_response_temporal_feedback_v1.decide(arm, root, step, target, predict, meter, carry)

    horizon = 15 - 5 * step
    per = horizon // 5
    seed = 2027110400 + root * 100 + step * 10000
    best = {"chain": None, "score": float("inf"), "samples": 0}

    def tracked(actions):
        actions = np.asarray(actions, dtype=float)
        positions, scores = predict(actions)
        scores = np.asarray(scores, dtype=float)
        if actions.ndim != 3 or scores.shape != (len(actions),) or not np.isfinite(scores).all():
            raise ValueError("invalid prediction output")
        if actions.shape[1] == horizon:
            best["samples"] += len(actions)
            index = int(np.argmin(scores))
            if scores[index] < best["score"]:
                best["chain"] = actions[index].copy()
                best["score"] = float(scores[index])
        return positions, scores

    carry_score = float(tracked(carry[None])[1][0]) if carry is not None else None
    structure = {"svd": 0, "fd_action_samples": 0, "svd_seconds": 0.0}
    random_rng = np.random.default_rng(seed + 2) if arm == "matched_random" else None
    first = [{"name": "zero", "prefix": np.zeros((5, 2)).tolist()}]
    evidence = []
    for index, nominal in enumerate(core.NOMINALS):
        bank, info = core.response(nominal, [], tracked, structure, random_rng)
        evidence.append(info)
        first.extend({"name": f"nominal{index}/{item['name']}", "prefix": item["prefix"]} for item in bank)
    positions, scores = tracked(np.asarray([item["prefix"] for item in first]))
    programs, retained = [], []
    if step < 2:
        retained = core.retain(scores, positions)
        seeds = []
        for index in retained:
            prefix = np.asarray(first[index]["prefix"])
            candidates, info = core.response(core.NOMINALS[0], prefix, tracked, structure, random_rng)
            if step == 0:
                suffixes = [np.zeros((10, 2))] + [
                    np.concatenate((np.asarray(item["prefix"]), np.zeros((5, 2)))) for item in candidates]
            else:
                suffixes = [np.zeros((5, 2))] + [np.asarray(item["prefix"]) for item in candidates]
            joined = np.concatenate((np.broadcast_to(prefix, (len(suffixes), 5, 2)),
                                     np.asarray(suffixes)), axis=1)
            _, values = tracked(joined)
            selected = int(np.argmin(values))
            seeds.append({"index": index, "suffix": suffixes[selected], "score": float(values[selected]),
                          "seed_scores": values.tolist(), "successor_response": info})
        available = (core.CAP - per - meter.token_samples) // per
        counts = allocate_samples(3 * available // 4, 0, len(seeds), 1)
        if not counts or min(counts) < 13:
            raise ValueError("insufficient iterative suffix budget")
        for index, (item, count) in enumerate(zip(seeds, counts, strict=True)):
            chain, score, generations = core.cem(
                item["suffix"], first[item["index"]]["prefix"], count,
                np.random.default_rng(seed + 10 + index), tracked, item["score"], 4)
            programs.append({"parent_index": item["index"], "chain": chain.tolist(),
                             "terminal_score": score, "samples": count, "generations": generations,
                             "seed_scores": item["seed_scores"], "successor_response": item["successor_response"]})
    if best["chain"] is None:
        raise ValueError("no complete chain evaluated")
    refinement_start = {"chain": best["chain"].tolist(), "terminal_score": best["score"],
                        "full_samples_seen": best["samples"]}
    count = (core.CAP - per - meter.token_samples) // per
    if count < 1:
        raise ValueError("no joint refinement budget")
    _, _, generations = core.cem(best["chain"], [], count, np.random.default_rng(seed + 20),
                                  tracked, best["score"], 4)
    precomparison = {"chain": best["chain"].tolist(), "terminal_score": best["score"],
                     "full_samples_seen": best["samples"]}
    comparison = best["chain"].copy()
    comparison[:5] = 0
    comparison_score = float(tracked(comparison[None])[1][0])
    abstained = comparison_score <= precomparison["terminal_score"]
    selected = ({"name": "zero_prefix", "chain": comparison.tolist(), "terminal_score": comparison_score}
                if abstained else {"name": "global_scored_incumbent", **{k: precomparison[k]
                                    for k in ("chain", "terminal_score")}})
    return {"selected": selected, "precomparison": precomparison, "comparison_chain": comparison.tolist(),
            "comparison_score": comparison_score, "abstained": bool(abstained),
            "carry": None if carry is None else carry.tolist(), "carry_score": carry_score,
            "first_bank": first, "prefix_scores": scores.tolist(), "first_response": evidence,
            "retained": retained, "programs": programs, "refinement_start": refinement_start,
            "refinement": generations, "refine_samples": count, "structure": structure}
