"""Goal-weighted response and successor-retention compiler, development v4."""
import time

import numpy as np

from experiments.pc_budget_allocation_20261103_codex import allocate_samples
from experiments.spectral_reachability import d2p_local_compiler as local
from experiments.spectral_reachability import pc_response_temporal_search_v1 as core

METHOD = "target_reach_successor_fixed_terminal_dev_v4"
DAMPING = .1


def response(nominal, prefix, target, predict, structure):
    """Generate both signs in task-ranked response modes; all probes are metered."""
    nominal = np.asarray(nominal, dtype=float)
    prefix = np.asarray(prefix, dtype=float).reshape(-1, 2)
    axes = np.eye(10).reshape(10, 5, 2)
    probes = np.concatenate((nominal[None], nominal[None] + .05 * axes, nominal[None] - .05 * axes))
    chains = np.concatenate((np.broadcast_to(prefix, (21, len(prefix), 2)), probes), axis=1)
    positions, _ = predict(chains)
    positions = np.asarray(positions, dtype=float)
    if positions.shape != (21, 2) or not np.isfinite(positions).all():
        raise ValueError("invalid response positions")
    matrix = ((positions[1:11] - positions[11:21]) / .1).T
    started = time.monotonic()
    u, singular, right = np.linalg.svd(matrix, full_matrices=False)
    residual = target - positions[0]
    rank = int(np.sum(singular > 1e-10))
    contributions = singular ** 2 / (singular ** 2 + DAMPING) * (u.T @ residual) ** 2
    order = np.argsort(-contributions[:rank], kind="stable")
    geometry = local.local_geometry(matrix, residual, np.eye(2), np.eye(10), DAMPING)
    structure["svd_seconds"] += time.monotonic() - started
    # One decomposition orders modes; local_geometry performs a second one.
    structure["svd"] += 2
    structure["fd_action_samples"] += 21
    bank = []
    for index in order:
        direction = right[index].reshape(5, 2)
        delta = direction * (.5 / np.max(np.abs(direction)))
        for sign in (-1, 1):
            bank.append({"name": f"mode{int(index)}_{sign:+d}",
                         "prefix": np.clip(nominal + sign * delta, -1, 1).tolist()})
    return bank, {"nominal": nominal.tolist(), "position": positions[0].tolist(), "matrix": matrix.tolist(),
                  "singular": singular.tolist(), "mode_contributions": contributions.tolist(),
                  "mode_order": order.tolist(), "rank": rank,
                  "linear_reach_value": geometry["linear_reach_value"]}


def retain(scores, responses, number=4):
    """Keep best immediate and local-reach candidates, then response diversity."""
    if not responses or len(scores) != len(responses):
        raise ValueError("missing successor response")
    matrices = np.stack([np.asarray(item["matrix"], dtype=float).reshape(-1) for item in responses])
    values = np.asarray([item["linear_reach_value"] for item in responses])
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(matrices).all() or not np.isfinite(values).all() or not np.isfinite(scores).all():
        raise ValueError("nonfinite retention input")
    chosen = [int(np.argmin(scores))]
    reach = int(np.argmin(values))
    if reach not in chosen and len(chosen) < min(number, len(scores)):
        chosen.append(reach)
    while len(chosen) < min(number, len(scores)):
        distance = np.linalg.norm(matrices[:, None] - matrices[chosen][None], axis=2).min(axis=1)
        distance[chosen] = -np.inf
        chosen.append(int(np.argmax(distance)))
    return chosen


def decide(arm, root, step, target, predict, meter, carry=None):
    if arm != "response" or step not in (0, 1, 2) or meter.token_samples:
        raise ValueError("invalid arm, step, or meter")
    target = np.asarray(target, dtype=float)
    if target.shape != (2,) or not np.isfinite(target).all():
        raise ValueError("invalid target")
    horizon = 15 - step * 5
    if step == 0:
        if carry is not None:
            raise ValueError("unexpected carry")
    else:
        carry = np.asarray(carry, dtype=float)
        if carry.shape != (horizon, 2) or not np.isfinite(carry).all() or np.any(np.abs(carry) > 1):
            raise ValueError("invalid carry")
    per = horizon // 5
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
        bank, info = response(nominal, [], target, tracked, structure)
        first_response.append(info)
        first.extend({"name": f"nominal{index}/{item['name']}", "prefix": item["prefix"]} for item in bank)
    _, scores = tracked(np.asarray([item["prefix"] for item in first]))
    successor, retained, programs = [], [], []
    if step < 2:
        for item in first:
            candidates, info = response(core.NOMINALS[0], item["prefix"], target, tracked, structure)
            successor.append({"candidates": candidates, **info})
        retained = retain(scores, successor)
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
            chosen = int(np.argmin(values))
            seeds.append({"index": index, "suffix": suffixes[chosen], "score": float(values[chosen]),
                          "seed_scores": values.tolist()})
        available = (cap - per - meter.token_samples) // per
        counts = allocate_samples(3 * available // 4, 0, len(seeds), 1)
        if not counts or min(counts) < 13:
            raise ValueError("insufficient suffix budget")
        seed = 2027110400 + root * 100 + step * 10000
        for index, (item, count) in enumerate(zip(seeds, counts, strict=True)):
            chain, score, generations = core.cem(item["suffix"], first[item["index"]]["prefix"], count,
                np.random.default_rng(seed + 10 + index), tracked, item["score"], 4)
            programs.append({"parent_index": item["index"], "chain": chain.tolist(), "terminal_score": score,
                             "samples": count, "generations": generations, "seed_scores": item["seed_scores"]})
    if best["chain"] is None:
        raise ValueError("no complete chain")
    start = {"chain": best["chain"].tolist(), "terminal_score": best["score"], "full_samples_seen": best["samples"]}
    count = (cap - per - meter.token_samples) // per
    if count < 1:
        raise ValueError("no refinement budget")
    seed = 2027110400 + root * 100 + step * 10000
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
            "comparison_score": comparison_score, "abstained": bool(abstained),
            "carry_score": carry_score, "first_bank": first, "prefix_scores": scores.tolist(),
            "first_response": first_response, "successor_responses": successor, "retained": retained,
            "programs": programs, "refinement_start": start, "refinement": generations,
            "refine_samples": count, "structure": structure}
