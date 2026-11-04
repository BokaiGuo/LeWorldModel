"""V5-equivalent response-program development compiler with configurable action dimension."""
import time

import numpy as np

from experiments.pc_budget_allocation_20261103_codex import allocate_samples
from experiments.spectral_reachability import d2p_local_compiler as local
from experiments.spectral_reachability import pc_target_reach_feedback_v4 as v4

METHOD = "target_reach_action_dim_dev_v6"
CAP, PACKET, DAMPING = 1920, 5, .1


def response(nominal, prefix, target, predict, structure):
    nominal = np.asarray(nominal, dtype=float)
    dimension = nominal.shape[1]
    prefix = np.asarray(prefix, dtype=float).reshape(-1, dimension)
    axes = np.eye(PACKET * dimension).reshape(-1, PACKET, dimension)
    probes = np.concatenate((nominal[None], nominal[None] + .05*axes, nominal[None] - .05*axes))
    chains = np.concatenate((np.broadcast_to(prefix, (len(probes), len(prefix), dimension)), probes), axis=1)
    positions, _ = predict(chains)
    positions = np.asarray(positions, dtype=float)
    count = PACKET * dimension
    if positions.shape != (1 + 2*count, 2) or not np.isfinite(positions).all():
        raise ValueError("invalid response positions")
    matrix = ((positions[1:count+1] - positions[count+1:]) / .1).T
    started = time.monotonic()
    u, singular, right = np.linalg.svd(matrix, full_matrices=False)
    residual = target - positions[0]
    rank = int(np.sum(singular > 1e-10))
    contributions = singular**2 / (singular**2 + DAMPING) * (u.T @ residual)**2
    order = np.argsort(-contributions[:rank], kind="stable")
    geometry = local.local_geometry(matrix, residual, np.eye(2), np.eye(count), DAMPING)
    structure["svd_seconds"] += time.monotonic() - started
    structure["svd"] += 2
    structure["fd_action_samples"] += len(probes)
    bank = []
    for index in order:
        direction = right[index].reshape(PACKET, dimension)
        delta = direction * (.5 / np.max(np.abs(direction)))
        for sign in (-1, 1):
            bank.append({"name": f"mode{int(index)}_{sign:+d}",
                         "prefix": np.clip(nominal + sign*delta, -1, 1).tolist()})
    return bank, {"nominal": nominal.tolist(), "position": positions[0].tolist(), "matrix": matrix.tolist(),
                  "singular": singular.tolist(), "mode_contributions": contributions.tolist(),
                  "mode_order": order.tolist(), "rank": rank, "linear_reach_value": geometry["linear_reach_value"]}


def cem(base, prefix, count, rng, predict, initial_score, knots):
    base = np.asarray(base, dtype=float)
    if base.ndim != 2 or base.shape[1] < 1 or count < 1:
        raise ValueError("invalid CEM input")
    dimension = base.shape[1]
    prefix = np.asarray(prefix, dtype=float).reshape(-1, dimension)
    loc, scale = np.zeros((knots, dimension)), np.full((knots, dimension), .35)
    best_param, best_chain, best_score = loc.copy(), np.concatenate((prefix, base)), float(initial_score)
    generations = []
    for offset in range(0, count, 12):
        size = min(12, count-offset)
        proposals = loc + rng.normal(size=(size, knots, dimension))*scale
        proposals[0] = best_param
        x, at = np.linspace(0, len(base)-1, knots), np.arange(len(base))
        delta = np.stack([np.stack([np.interp(at, x, row[:, k]) for k in range(dimension)], axis=-1)
                          for row in proposals])
        suffix = np.clip(base[None]+delta, -1, 1)
        chains = np.concatenate((np.broadcast_to(prefix, (size, len(prefix), dimension)), suffix), axis=1)
        scores = np.asarray(predict(chains)[1], dtype=float)
        if scores.shape != (size,) or not np.isfinite(scores).all():
            raise ValueError("invalid CEM scores")
        index = int(np.argmin(scores))
        if scores[index] < best_score:
            best_score, best_chain, best_param = float(scores[index]), chains[index].copy(), proposals[index].copy()
        before_loc, before_scale = loc.copy(), scale.copy()
        if size > 1:
            elite = proposals[np.argsort(scores, kind="stable")[:min(4, size)]]
            loc, scale = elite.mean(axis=0), np.maximum(elite.std(axis=0), .07)
        generations.append({"size": size, "loc_before": before_loc.tolist(), "scale_before": before_scale.tolist(),
                            "loc_after": loc.tolist(), "scale_after": scale.tolist(), "best_score": best_score})
    return best_chain, best_score, generations


def decide(strategy, arm, root, step, target, predict, meter, nominals, carry=None):
    nominals = np.asarray(nominals, dtype=float)
    if (strategy not in ("response", "position") or arm != "response" or step not in (0, 1, 2)
            or meter.token_samples or nominals.ndim != 3 or nominals.shape[:2] != (2, PACKET)
            or nominals.shape[2] < 1 or not np.isfinite(nominals).all() or np.any(np.abs(nominals) > .95)):
        raise ValueError("invalid strategy, meter, or nominal probes")
    dimension = nominals.shape[2]
    target = np.asarray(target, dtype=float)
    if target.shape != (2,) or not np.isfinite(target).all():
        raise ValueError("invalid target")
    horizon, per = 15-step*PACKET, 3-step
    if step == 0:
        if carry is not None:
            raise ValueError("unexpected carry")
    else:
        carry = np.asarray(carry, dtype=float)
        if carry.shape != (horizon, dimension) or not np.isfinite(carry).all() or np.any(np.abs(carry)>1):
            raise ValueError("invalid carry")
    cap = min(CAP, meter.token_cap)
    best = {"chain": None, "score": float("inf"), "samples": 0}

    def tracked(actions):
        actions = np.asarray(actions, dtype=float)
        positions, scores = predict(actions)
        scores = np.asarray(scores, dtype=float)
        if actions.ndim != 3 or actions.shape[2] != dimension or scores.shape != (len(actions),) or not np.isfinite(scores).all():
            raise ValueError("invalid evaluation")
        if actions.shape[1] == horizon:
            best["samples"] += len(actions)
            index = int(np.argmin(scores))
            if scores[index] < best["score"]:
                best["chain"], best["score"] = actions[index].copy(), float(scores[index])
        return positions, scores

    carry_score = float(tracked(carry[None])[1][0]) if carry is not None else None
    structure = {"svd": 0, "fd_action_samples": 0, "svd_seconds": 0.0}
    first = [{"name": "zero", "prefix": np.zeros((PACKET, dimension)).tolist()}]
    first_response = []
    for index, nominal in enumerate(nominals):
        bank, info = response(nominal, [], target, tracked, structure)
        first_response.append(info)
        first.extend({"name": f"nominal{index}/{item['name']}", "prefix": item["prefix"]} for item in bank)
    positions, scores = tracked(np.asarray([item["prefix"] for item in first]))
    successor, retained, programs = [], [], []
    seed = 2027110400 + root*100 + step*10000
    if step < 2:
        for item in first:
            candidates, info = response(nominals[0], item["prefix"], target, tracked, structure)
            successor.append({"candidates": candidates, **info})
        if strategy == "response":
            chosen = v4.retain(scores, successor)
        else:
            from experiments.spectral_reachability import pc_response_temporal_search_v1 as core
            chosen = core.retain(scores, positions)
        retained = sorted(chosen)
        seeds = []
        for index in retained:
            prefix = np.asarray(first[index]["prefix"])
            suffixes = [np.zeros((horizon-PACKET, dimension))]
            for item in successor[index]["candidates"]:
                suffix = np.asarray(item["prefix"])
                if step == 0:
                    suffix = np.concatenate((suffix, np.zeros((PACKET, dimension))))
                suffixes.append(suffix)
            joined = np.concatenate((np.broadcast_to(prefix, (len(suffixes), PACKET, dimension)), suffixes), axis=1)
            _, values = tracked(joined)
            choice = int(np.argmin(values))
            seeds.append({"index": index, "suffix": suffixes[choice], "score": float(values[choice]),
                          "seed_scores": values.tolist()})
        available = (cap-per-meter.token_samples)//per
        counts = allocate_samples(3*available//4, 0, len(seeds), 1)
        if not counts or min(counts) < 13:
            raise ValueError("insufficient suffix budget")
        for item, count in zip(seeds, counts, strict=True):
            candidate_seed = seed+100000+item["index"]
            chain, score, generations = cem(item["suffix"], first[item["index"]]["prefix"], count,
                np.random.default_rng(candidate_seed), tracked, item["score"], 4)
            programs.append({"parent_index": item["index"], "chain": chain.tolist(), "terminal_score": score,
                "samples": count, "rng_seed": candidate_seed, "generations": generations,
                "seed_scores": item["seed_scores"]})
    if best["chain"] is None:
        raise ValueError("no complete chain")
    start = {"chain": best["chain"].tolist(), "terminal_score": best["score"], "full_samples_seen": best["samples"]}
    count = (cap-per-meter.token_samples)//per
    if count < 1:
        raise ValueError("no refinement budget")
    _, _, generations = cem(best["chain"], [], count, np.random.default_rng(seed+20), tracked, best["score"], 4)
    before = {"chain": best["chain"].tolist(), "terminal_score": best["score"], "full_samples_seen": best["samples"]}
    comparison = best["chain"].copy()
    comparison[:PACKET] = 0
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
