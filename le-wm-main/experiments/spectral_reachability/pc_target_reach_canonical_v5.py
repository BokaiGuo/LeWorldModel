"""V5 response compiler with candidate-identity-stable CEM streams."""
import numpy as np

from experiments.pc_budget_allocation_20261103_codex import allocate_samples
from experiments.spectral_reachability import pc_response_temporal_search_v1 as core
from experiments.spectral_reachability import pc_target_reach_feedback_v4 as v4

METHOD = "target_reach_canonical_stream_dev_v5"
STRATEGIES = ("response", "position")


def decide(strategy, arm, root, step, target, predict, meter, carry=None):
    if strategy not in STRATEGIES or arm != "response" or step not in (0, 1, 2) or meter.token_samples:
        raise ValueError("invalid strategy, arm, step, or meter")
    target = np.asarray(target, dtype=float)
    if target.shape != (2,) or not np.isfinite(target).all():
        raise ValueError("invalid target")
    horizon, per = 15 - step * 5, 3 - step
    if step == 0:
        if carry is not None:
            raise ValueError("unexpected carry")
    else:
        carry = np.asarray(carry, dtype=float)
        if carry.shape != (horizon, 2) or not np.isfinite(carry).all() or np.any(np.abs(carry) > 1):
            raise ValueError("invalid carry")
    cap = min(core.CAP, meter.token_cap)
    best = {"chain": None, "score": float("inf"), "samples": 0}

    def tracked(actions):
        actions = np.asarray(actions, dtype=float)
        positions, scores = predict(actions)
        scores = np.asarray(scores, dtype=float)
        if actions.ndim != 3 or scores.shape != (len(actions),) or not np.isfinite(scores).all():
            raise ValueError("invalid evaluation")
        if actions.shape[1] == horizon:
            best["samples"] += len(actions)
            index = int(np.argmin(scores))
            if scores[index] < best["score"]:
                best["chain"], best["score"] = actions[index].copy(), float(scores[index])
        return positions, scores

    carry_score = float(tracked(carry[None])[1][0]) if carry is not None else None
    structure = {"svd": 0, "fd_action_samples": 0, "svd_seconds": 0.0}
    first = [{"name": "zero", "prefix": np.zeros((5, 2)).tolist()}]
    first_response = []
    for index, nominal in enumerate(core.NOMINALS):
        bank, info = v4.response(nominal, [], target, tracked, structure)
        first_response.append(info)
        first.extend({"name": f"nominal{index}/{item['name']}", "prefix": item["prefix"]} for item in bank)
    positions, scores = tracked(np.asarray([item["prefix"] for item in first]))
    successor, retained, programs = [], [], []
    seed = 2027110400 + root * 100 + step * 10000
    if step < 2:
        for item in first:
            candidates, info = v4.response(core.NOMINALS[0], item["prefix"], target, tracked, structure)
            successor.append({"candidates": candidates, **info})
        chosen = (v4.retain(scores, successor) if strategy == "response"
                  else core.retain(scores, positions))
        retained = sorted(chosen)
        seeds = []
        for index in retained:
            prefix = np.asarray(first[index]["prefix"])
            suffixes = [np.zeros((horizon - 5, 2))]
            for item in successor[index]["candidates"]:
                suffix = np.asarray(item["prefix"])
                if step == 0:
                    suffix = np.concatenate((suffix, np.zeros((5, 2))))
                suffixes.append(suffix)
            joined = np.concatenate((np.broadcast_to(prefix, (len(suffixes), 5, 2)), suffixes), axis=1)
            _, values = tracked(joined)
            chosen_suffix = int(np.argmin(values))
            seeds.append({"index": index, "suffix": suffixes[chosen_suffix], "score": float(values[chosen_suffix]),
                          "seed_scores": values.tolist()})
        available = (cap - per - meter.token_samples) // per
        counts = allocate_samples(3 * available // 4, 0, len(seeds), 1)
        if not counts or min(counts) < 13:
            raise ValueError("insufficient suffix budget")
        for item, count in zip(seeds, counts, strict=True):
            candidate_seed = seed + 100000 + item["index"]
            chain, score, generations = core.cem(item["suffix"], first[item["index"]]["prefix"], count,
                np.random.default_rng(candidate_seed), tracked, item["score"], 4)
            programs.append({"parent_index": item["index"], "chain": chain.tolist(), "terminal_score": score,
                "samples": count, "rng_seed": candidate_seed, "generations": generations,
                "seed_scores": item["seed_scores"]})
    if best["chain"] is None:
        raise ValueError("no complete chain")
    start = {"chain": best["chain"].tolist(), "terminal_score": best["score"], "full_samples_seen": best["samples"]}
    count = (cap - per - meter.token_samples) // per
    if count < 1:
        raise ValueError("no refinement budget")
    _, _, generations = core.cem(best["chain"], [], count, np.random.default_rng(seed + 20),
                                  tracked, best["score"], 4)
    before = {"chain": best["chain"].tolist(), "terminal_score": best["score"], "full_samples_seen": best["samples"]}
    comparison = best["chain"].copy()
    comparison[:5] = 0
    comparison_score = float(tracked(comparison[None])[1][0])
    abstained = comparison_score <= before["terminal_score"]
    selected = ({"name": "zero_prefix", "chain": comparison.tolist(), "terminal_score": comparison_score}
                if abstained else {"name": "global_scored_incumbent", "chain": before["chain"],
                                   "terminal_score": before["terminal_score"]})
    return {"selected": selected, "precomparison": before, "comparison_chain": comparison.tolist(),
        "comparison_score": comparison_score, "abstained": bool(abstained), "carry_score": carry_score,
        "first_bank": first, "prefix_scores": scores.tolist(), "first_response": first_response,
        "successor_responses": successor, "retained": retained, "programs": programs,
        "refinement_start": start, "refinement": generations, "refine_samples": count, "structure": structure,
        "retention_strategy": strategy, "rng_assignment": "root_step_candidate_bank_index"}
