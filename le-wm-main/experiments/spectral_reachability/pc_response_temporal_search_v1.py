"""Development response-program compiler with temporal correlated CEM."""
import time

import numpy as np

from experiments.pc_budget_allocation_20261103_codex import allocate_samples

CAP = 1920
NOMINALS = (np.tile([0.25, 0.0], (5, 1)), np.tile([0.25, 0.5], (5, 1)))
ARMS = ("response", "matched_random", "smooth_cem", "action_cem")


def interpolate(parameters, length):
    """Linear interpolation in action time; final dimension is action coordinate."""
    values = np.asarray(parameters, dtype=float)
    if values.ndim != 3 or values.shape[2] != 2 or values.shape[1] < 2:
        raise ValueError("invalid knot parameters")
    x = np.linspace(0, length - 1, values.shape[1])
    at = np.arange(length)
    return np.stack([np.stack([np.interp(at, x, row[:, k]) for k in range(2)], axis=-1)
                     for row in values])


def cem(base, prefix, count, rng, predict, initial_score, knots):
    """Add bounded CEM perturbations to base, retaining the best scored chain."""
    base = np.asarray(base, dtype=float)
    prefix = np.asarray(prefix, dtype=float).reshape(-1, 2)
    if base.ndim != 2 or base.shape[1] != 2 or count < 1:
        raise ValueError("invalid CEM input")
    loc, scale = np.zeros((knots, 2)), np.full((knots, 2), .35)
    best_param = loc.copy()
    best_chain = np.concatenate((prefix, base))
    best_score = float(initial_score)
    generations = []
    for offset in range(0, count, 12):
        size = min(12, count - offset)
        proposals = loc + rng.normal(size=(size, knots, 2)) * scale
        proposals[0] = best_param
        delta = interpolate(proposals, len(base))
        suffix = np.clip(base[None] + delta, -1, 1)
        chains = np.concatenate((np.broadcast_to(prefix, (size, len(prefix), 2)), suffix), axis=1)
        scores = np.asarray(predict(chains)[1], dtype=float)
        if scores.shape != (size,) or not np.isfinite(scores).all():
            raise ValueError("invalid CEM scores")
        index = int(np.argmin(scores))
        if scores[index] < best_score:
            best_score = float(scores[index])
            best_chain = chains[index].copy()
            best_param = proposals[index].copy()
        before_loc, before_scale = loc.copy(), scale.copy()
        if size > 1:
            elite = proposals[np.argsort(scores, kind="stable")[:min(4, size)]]
            loc = elite.mean(axis=0)
            scale = np.maximum(elite.std(axis=0), .07)
        generations.append({"size": size, "loc_before": before_loc.tolist(),
                            "scale_before": before_scale.tolist(), "loc_after": loc.tolist(),
                            "scale_after": scale.tolist(), "best_score": best_score})
    return best_chain, best_score, generations


def response(nominal, prefix, predict, structure, random_rng):
    nominal, prefix = np.asarray(nominal), np.asarray(prefix).reshape(-1, 2)
    axes = np.eye(10).reshape(10, 5, 2)
    probes = np.concatenate((nominal[None], nominal[None] + .05 * axes,
                             nominal[None] - .05 * axes))
    chains = np.concatenate((np.broadcast_to(prefix, (21, len(prefix), 2)), probes), axis=1)
    position, _ = predict(chains)
    matrix = ((position[1:11] - position[11:21]) / .1).T
    started = time.monotonic()
    _, singular, right = np.linalg.svd(matrix, full_matrices=False)
    structure["svd_seconds"] += time.monotonic() - started
    structure["svd"] += 1
    structure["fd_action_samples"] += 21
    rank = int(np.sum(singular > 1e-10))
    directions = right[:rank]
    if random_rng is not None and rank:
        directions = np.linalg.qr(random_rng.normal(size=(10, rank)), mode="reduced")[0].T
    bank = []
    for index, direction in enumerate(directions):
        delta = direction.reshape(5, 2)
        delta *= .5 / np.max(np.abs(delta))
        for sign in (-1, 1):
            bank.append({"name": f"dir{index}_{sign:+d}",
                         "prefix": np.clip(nominal + sign * delta, -1, 1).tolist()})
    return bank, {"nominal": nominal.tolist(), "matrix": matrix.tolist(),
                  "singular": singular.tolist(), "directions": directions.tolist(), "rank": rank}


def retain(scores, positions, number=4):
    positions, scores = np.asarray(positions), np.asarray(scores)
    chosen = [int(np.argmin(scores))]
    while len(chosen) < min(number, len(scores)):
        distances = np.min(np.linalg.norm(positions[:, None] - positions[chosen][None], axis=2), axis=1)
        distances[chosen] = -np.inf
        chosen.append(int(np.argmax(distances)))
    return chosen


def decide(arm, root, target, predict, meter):
    if arm not in ARMS or meter.token_samples != 0:
        raise ValueError("invalid arm or nonempty meter")
    seed = 2027110400 + root * 100
    structure = {"svd": 0, "fd_action_samples": 0, "svd_seconds": 0.0}
    evidence, first, scores, programs = [], [], [], []
    if arm in ("smooth_cem", "action_cem"):
        zero = np.zeros((15, 2))
        initial = float(predict(zero[None])[1][0])
        count = (CAP - meter.token_samples) // 3
        chain, score, generations = cem(zero, [], count, np.random.default_rng(seed + 1), predict,
                                        initial, 4 if arm == "smooth_cem" else 15)
        selected = {"name": arm, "chain": chain.tolist(), "terminal_score": score}
        return {"selected": selected, "first_bank": [], "prefix_scores": [], "programs": [],
                "structure": structure, "generations": generations, "abstained": None}
    random_rng = np.random.default_rng(seed + 2) if arm == "matched_random" else None
    first.append({"name": "zero", "prefix": np.zeros((5, 2)).tolist()})
    for index, nominal in enumerate(NOMINALS):
        bank, info = response(nominal, [], predict, structure, random_rng)
        evidence.append(info)
        first.extend({"name": f"nominal{index}/{item['name']}", "prefix": item["prefix"]} for item in bank)
    positions, raw_scores = predict(np.asarray([item["prefix"] for item in first]))
    scores = raw_scores.tolist()
    selected_indices = retain(scores, positions)
    seeds = []
    for index in selected_indices:
        prefix = np.asarray(first[index]["prefix"])
        candidates, info = response(NOMINALS[0], prefix, predict, structure, random_rng)
        suffixes = [np.zeros((10, 2))] + [np.concatenate((np.asarray(item["prefix"]), np.zeros((5, 2))))
                                            for item in candidates]
        joined = np.concatenate((np.broadcast_to(prefix, (len(suffixes), 5, 2)), np.asarray(suffixes)), axis=1)
        _, values = predict(joined)
        best = int(np.argmin(values))
        seeds.append({"index": index, "suffix": suffixes[best], "score": float(values[best]),
                      "successor_response": info, "seed_scores": values.tolist()})
    available = (CAP - 3 - meter.token_samples) // 3
    counts = allocate_samples(3 * available // 4, 0, len(seeds), 1)
    if not counts or min(counts) < 13:
        raise ValueError("insufficient iterative suffix budget")
    for seed_index, (item, count) in enumerate(zip(seeds, counts, strict=True)):
        chain, score, generations = cem(item["suffix"], first[item["index"]]["prefix"], count,
                                        np.random.default_rng(seed + 10 + seed_index), predict, item["score"], 4)
        programs.append({"parent_index": item["index"], "chain": chain.tolist(), "terminal_score": score,
                         "samples": count, "generations": generations, "seed_scores": item["seed_scores"],
                         "successor_response": item["successor_response"]})
    selected = min(programs, key=lambda row: row["terminal_score"])
    refine_count = (CAP - 3 - meter.token_samples) // 3
    refined_chain, refined_score, refinement = cem(np.asarray(selected["chain"]), [], refine_count,
                                                     np.random.default_rng(seed + 20), predict,
                                                     selected["terminal_score"], 4)
    before = {"name": "refined" if refined_score < selected["terminal_score"] else "program",
              "chain": refined_chain.tolist(), "terminal_score": refined_score}
    comparison = refined_chain.copy()
    comparison[:5] = 0
    comparison_score = float(predict(comparison[None])[1][0])
    abstained = comparison_score <= refined_score
    chosen = ({"name": "zero_prefix", "chain": comparison.tolist(), "terminal_score": comparison_score}
              if abstained else before)
    return {"selected": chosen, "precomparison": before, "comparison_score": comparison_score,
            "abstained": abstained, "first_bank": first, "prefix_scores": scores,
            "retained": selected_indices, "first_response": evidence, "programs": programs,
            "refinement": refinement, "refine_samples": refine_count, "structure": structure}
