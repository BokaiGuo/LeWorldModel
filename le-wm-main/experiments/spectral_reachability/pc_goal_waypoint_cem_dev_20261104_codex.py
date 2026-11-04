"""Two-dimensional waypoint-CEM baseline for bounded E0 development."""
import time

import numpy as np

from experiments.spectral_reachability import pc_response_temporal_search_v1 as core


def plan(predict, meter, goal, seed):
    """Search goal-relative waypoints; retain all evaluated full-chain incumbents."""
    target = np.asarray(goal, dtype=np.float64)
    if target.shape != (2,) or not np.isfinite(target).all() or meter.token_samples:
        raise ValueError("expected two-dimensional goal and empty meter")
    started = time.monotonic()
    best, best_score = None, float("inf")

    def tracked(chains):
        nonlocal best, best_score
        raw = np.asarray(chains, dtype=np.float32)
        features, scores = predict(raw)
        if raw.shape[1] == 15:
            index = int(np.argmin(scores))
            if scores[index] < best_score:
                best, best_score = raw[index].copy(), float(scores[index])
        return features, scores

    zero = np.zeros((15, 2), np.float32)
    tracked(zero[None])
    position, _ = tracked(zero[None, :5])
    origin = np.asarray(position[0], float)
    delta = target - origin
    if np.linalg.norm(delta) < 1e-12:
        raise ValueError("degenerate waypoint geometry")
    perpendicular = np.array([-delta[1], delta[0]])
    waypoint_records = []
    for index, sign in enumerate((-1, 1)):
        waypoint = origin + .5 * delta + sign * 1.2 * perpendicular

        def waypoint_predict(actions):
            feature, _ = tracked(actions)
            return feature, np.linalg.norm(feature - waypoint, axis=1)

        prefix_zero = zero[:5]
        _, initial = waypoint_predict(prefix_zero[None])
        prefix, prefix_score, prefix_generations = core.cem(
            prefix_zero, [], 160, np.random.default_rng(seed + index), waypoint_predict,
            float(initial[0]), 4)
        suffix_zero = zero[5:]
        full = np.concatenate((prefix, suffix_zero))
        _, scores = tracked(full[None])
        chain, suffix_score, suffix_generations = core.cem(
            suffix_zero, prefix, 220, np.random.default_rng(seed + 10 + index), tracked,
            float(scores[0]), 4)
        waypoint_records.append({"waypoint": waypoint.tolist(), "prefix": prefix.tolist(),
            "prefix_waypoint_distance": prefix_score, "suffix_chain": chain.tolist(),
            "suffix_score": suffix_score, "prefix_generations": prefix_generations,
            "suffix_generations": suffix_generations})
    reserve = 6
    remaining = (1920 - meter.token_samples - reserve) // 3
    if remaining:
        core.cem(best.copy(), [], remaining, np.random.default_rng(seed + 20), tracked, best_score, 4)
    incumbent, pre_pair_score = best.copy(), best_score
    comparison = incumbent.copy()
    comparison[:5] = 0
    _, paired = tracked(np.stack((incumbent, comparison)))
    abstained = bool(np.array_equal(incumbent, comparison) or paired[1] <= paired[0])
    return {"method": "goal_waypoint_cem_dev", "waypoints": waypoint_records,
            "pre_pair_score": pre_pair_score, "incumbent": incumbent.tolist(),
            "paired_scores": paired.tolist(), "abstained": abstained,
            "selected": (comparison if abstained else incumbent).tolist(),
            "selected_score": float(paired[int(abstained)]),
            "refinement_samples": remaining, "wall_seconds": time.monotonic() - started}
