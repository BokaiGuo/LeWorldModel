"""Multi-chain refinement with an explicit iterative optimization reserve.

The caller supplies a frozen predictor returning (terminal features, task scores).
All actions passed to that predictor are exactly the recorded bounded float32 actions.
"""
from dataclasses import asdict, dataclass
import time

import numpy as np

from experiments.spectral_reachability import pc_shared_multistart_cem


def preserve_reachability_winner(chosen, reachability, count):
    """Reserve the best value if it is not already in the retained slots."""
    if reachability and len(chosen) < count:
        winner = min(reachability, key=lambda item: (item["value"], item["candidate"]))["candidate"]
        if winner not in chosen:
            chosen.append(winner)
    return chosen


@dataclass(frozen=True)
class Config:
    action_dim: int
    segment_length: int
    segments: int
    nominals: tuple
    model_packet_length: int = 5
    amplitudes: tuple = (0.25, 0.5)
    fd_step: float = 0.05
    max_modes: int = 1
    beam_width: int = 4
    greedy_slots: int = 2
    damping: float = 0.1
    token_cap: int = 1920
    seed: int = 0
    refine_batch: int = 12
    refine_elites: int = 4
    reachability_candidates: int = 4
    refine_starts: int = 4
    min_refine_samples: int = 24
    retention_policy: str = "response"
    successor_value: str = "ridge"
    direction_source: str = "spectral"


def retention_choices(scores, features, reachability, count, greedy_slots, seed):
    """Compute same-bank counterfactuals; response is inert with <2 probes."""
    scores = np.asarray(scores)
    features = np.asarray(features)
    order = np.argsort(scores, kind="stable")
    greedy = order[:count].tolist()
    fixed = order[:min(greedy_slots, count)].tolist()

    def fill(initial):
        chosen = initial.copy()
        while len(chosen) < count:
            separation = np.linalg.norm(features[:, None] - features[chosen][None], axis=2).min(axis=1)
            separation[chosen] = -np.inf
            chosen.append(int(np.argmax(separation)))
        return chosen

    diverse = fill(fixed)
    response = fixed.copy()
    if len(reachability) >= 2:
        preserve_reachability_winner(response, reachability, count)
    response = fill(response)
    available = [i for i in range(len(scores)) if i not in fixed]
    rng = np.random.default_rng(seed)
    random = fixed + rng.permutation(available)[:count-len(fixed)].tolist()
    return {"greedy": greedy, "diverse": diverse, "response": response, "random": random}


def compile_program(predict, target, config, carry=None, warm_starts=None):
    """Compile a bounded complete chain, or raise if the declared budget is inadequate."""
    c = config
    target = np.asarray(target, dtype=np.float64)
    if (target.ndim != 1 or not target.size or not np.isfinite(target).all()
            or min(c.action_dim, c.segment_length, c.segments, c.model_packet_length, c.max_modes, c.beam_width,
                   c.greedy_slots, c.token_cap, c.refine_batch, c.refine_elites,
                   c.reachability_candidates, c.refine_starts, c.min_refine_samples) < 1
            or c.greedy_slots > c.beam_width or c.refine_elites > c.refine_batch
            or c.min_refine_samples < 2 * c.refine_batch
            or c.retention_policy not in ("response", "greedy", "diverse", "random")
            or c.successor_value not in ("ridge", "probe_min")
            or c.direction_source not in ("spectral", "random")
            or not np.isfinite(c.fd_step) or c.fd_step <= 0 or c.damping <= 0
            or c.segment_length % c.model_packet_length):
        raise ValueError("invalid configuration or target")
    nominals = np.asarray(c.nominals, dtype=np.float64)
    if (nominals.ndim != 3 or nominals.shape[1:] != (c.segment_length, c.action_dim)
            or len(nominals) == 0 or not np.isfinite(nominals).all()
            or np.max(np.abs(nominals)) + c.fd_step > 1
            or not c.amplitudes or any(not np.isfinite(a) or a <= 0 for a in c.amplitudes)):
        raise ValueError("invalid nominal actions or amplitudes")
    horizon, dim = c.segment_length * c.segments, c.action_dim
    packets = horizon // c.model_packet_length
    reserve = 2 * packets
    min_start_samples = max(c.min_refine_samples, 1 + 2 * c.refine_batch)
    refine_reserve = c.refine_starts * min_start_samples * packets
    if refine_reserve + reserve >= c.token_cap:
        raise ValueError("insufficient budget for declared refinement reserve")
    if carry is not None:
        carry = np.asarray(carry, dtype=np.float32)
        if (carry.shape != (horizon, dim) or not np.isfinite(carry).all()
                or np.any(np.abs(carry) > 1)):
            raise ValueError("invalid carry")
    if warm_starts is None:
        warm_starts = np.empty((0, horizon, dim), dtype=np.float32)
    else:
        warm_starts = np.asarray(warm_starts, dtype=np.float32)
        if (warm_starts.ndim != 3 or warm_starts.shape[1:] != (horizon, dim)
                or len(warm_starts) >= c.refine_starts
                or not np.isfinite(warm_starts).all() or np.any(np.abs(warm_starts) > 1)):
            raise ValueError("invalid warm starts")
    used, calls, svds, svd_seconds = 0, 0, 0, 0.
    best_chain, best_score = None, float("inf")
    best_source = None
    full_samples = 0
    # Counts include every sample, including the baseline and final comparison.
    def evaluate(chains, source, final=False):
        nonlocal used, calls, best_chain, best_score, best_source, full_samples
        raw = np.asarray(chains, dtype=np.float32)
        if (raw.ndim != 3 or raw.shape[2] != dim or not len(raw) or raw.shape[1] % c.model_packet_length
                or raw.shape[1] > horizon or not np.isfinite(raw).all() or np.abs(raw).max() > 1):
            raise ValueError("invalid predictor actions")
        cost = len(raw) * (raw.shape[1] // c.model_packet_length)
        if used + cost > c.token_cap - (0 if final else reserve):
            raise RuntimeError("insufficient prediction-token budget")
        features, scores = predict(raw.copy())
        features, scores = np.asarray(features, dtype=np.float64), np.asarray(scores, dtype=np.float64)
        if (features.shape != (len(raw), len(target)) or scores.shape != (len(raw),)
                or not np.isfinite(features).all() or not np.isfinite(scores).all()):
            raise ValueError("invalid predictor output")
        used += cost
        calls += 1
        if raw.shape[1] == horizon and not final:
            full_samples += len(raw)
            index = int(np.argmin(scores))
            if scores[index] < best_score:
                best_chain, best_score, best_source = raw[index].copy(), float(scores[index]), source
        return features, scores

    started = time.monotonic()
    zero = np.zeros((horizon, dim), dtype=np.float32)
    if carry is None:
        evaluate(zero[None], "zero_full")
    else:
        _, initial_scores = evaluate(np.stack((zero, carry)), "zero_and_carry")
        best_source = ("zero_full", "carry")[int(np.argmin(initial_scores))]
    warm_seeds = []
    if len(warm_starts):
        _, warm_scores = evaluate(warm_starts, "global_warm_start")
        warm_seeds = [{"chain": chain.copy(), "score": float(score), "source": "global_warm_start"}
                      for chain, score in zip(warm_starts, warm_scores, strict=True)]
    beams = [{"chain": np.empty((0, dim), dtype=np.float32), "path": []}]
    layers = []
    axis = np.eye(c.segment_length * dim).reshape(-1, c.segment_length, dim)
    for depth in range(c.segments):
        candidates = []
        response_records = []
        for parent_index, parent in enumerate(beams):
            prefix = parent["chain"]
            for nominal_index, nominal in enumerate(nominals):
                probes = np.concatenate((nominal[None], nominal[None] + c.fd_step * axis,
                                         nominal[None] - c.fd_step * axis))
                joined = np.concatenate((np.broadcast_to(prefix, (len(probes), len(prefix), dim)),
                                         probes.astype(np.float32)), axis=1)
                features, _ = evaluate(joined, f"depth{depth}/probes")
                n = len(axis)
                matrix = ((features[1:n+1] - features[n+1:]) / (2*c.fd_step)).T
                svd_start = time.monotonic()
                u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
                svd_seconds += time.monotonic() - svd_start
                svds += 1
                contributions = (singular**2 / (singular**2 + c.damping)) * (u.T @ (target - features[0]))**2
                rank = int(np.sum(singular > 1e-10))
                modes = np.argsort(-contributions[:rank], kind="stable")[:c.max_modes]
                candidates.append({"chain": np.concatenate((prefix, nominal.astype(np.float32))),
                    "path": parent["path"] + [{"nominal": nominal_index, "mode": None, "scale": 0.}]})
                for mode in modes:
                    direction = vt[mode].reshape(c.segment_length, dim)
                    if c.direction_source == "random":
                        rng = np.random.default_rng(np.random.SeedSequence(
                            [c.seed, 2032700000, depth, parent_index, nominal_index, int(mode)]))
                        direction = rng.normal(size=(c.segment_length, dim))
                    denom = np.max(np.abs(direction))
                    for amplitude in c.amplitudes:
                        for sign in (-1, 1):
                            segment = np.clip(nominal + sign * amplitude * direction / denom, -1, 1).astype(np.float32)
                            candidates.append({"chain": np.concatenate((prefix, segment)),
                                "path": parent["path"] + [{"nominal": nominal_index, "mode": int(mode),
                                                           "scale": float(sign * amplitude)}]})
                response_records.append({"parent": parent_index, "nominal": nominal_index,
                    "matrix": matrix.tolist(), "singular": singular.tolist(),
                    "contributions": contributions.tolist(), "selected_modes": modes.tolist()})
        if not candidates:
            raise ValueError("empty candidate bank")
        values, scores = evaluate(np.stack([x["chain"] for x in candidates]), f"depth{depth}/candidates")
        for i, candidate in enumerate(candidates):
            candidate["score"] = float(scores[i])
        count = min(c.beam_width, len(candidates))
        # Preselect by the existing task score and feature diversity.  The
        # reachable-value calculation below never reads physical outcomes.
        greedy_reference = np.argsort(scores, kind="stable")[:min(c.greedy_slots, count)].tolist()
        shortlist = greedy_reference[:1]
        shortlist_count = min(c.reachability_candidates, len(candidates))
        while len(shortlist) < shortlist_count:
            separation = np.linalg.norm(values[:, None] - values[shortlist][None], axis=2).min(axis=1)
            separation[shortlist] = -np.inf
            # The second probe must be outside the already protected greedy
            # slots whenever such a candidate exists.
            if len(shortlist) == 1 and len(greedy_reference) < len(candidates):
                separation[greedy_reference] = -np.inf
            shortlist.append(int(np.argmax(separation)))
        reachability = []
        tail_length = horizon - (depth + 1) * c.segment_length
        if tail_length:
            # Orthonormal packet-constant action coordinates: fewer probes
            # than per-step axes, with an explicit reduced response space.
            tail_axis = []
            for offset in range(0, tail_length, c.model_packet_length):
                for action in range(dim):
                    direction = np.zeros((tail_length, dim), np.float64)
                    direction[offset:offset+c.model_packet_length, action] = 1 / np.sqrt(c.model_packet_length)
                    tail_axis.append(direction)
            tail_axis = np.stack(tail_axis)
            probe_count = 1 + 2 * len(tail_axis)
            probe_cost = probe_count * packets
            # Reserve a conservative upper bound for all later layers,
            # avoiding a reachability probe that would strand the search.
            max_modes = min(c.max_modes, len(target), c.segment_length * dim)
            max_candidates = 1 + 2 * len(c.amplitudes) * max_modes
            future_worst = sum(c.beam_width * len(nominals) * (1 + 2 * len(axis) + max_candidates)
                               * ((future_depth + 1) * c.segment_length // c.model_packet_length)
                               for future_depth in range(depth + 1, c.segments))
            available = max(0, (c.token_cap - reserve - refine_reserve - used - future_worst) // probe_cost)
            # One measurement cannot rank successor values. Do not charge
            # it to a "response selection" when no comparison is possible.
            probe_indices = shortlist[:min(len(shortlist), available)] if available >= 2 else []
            for index in probe_indices:
                prefix = candidates[index]["chain"]
                tail = np.zeros((tail_length, dim), np.float32)
                tails = np.concatenate((tail[None], tail[None] + c.fd_step * tail_axis,
                                        tail[None] - c.fd_step * tail_axis))
                chains = np.concatenate((np.broadcast_to(prefix, (probe_count, len(prefix), dim)),
                                         tails.astype(np.float32)), axis=1)
                response, probe_scores = evaluate(chains, f"depth{depth}/successor_reachability")
                n = len(tail_axis)
                matrix = ((response[1:n+1] - response[n+1:]) / (2 * c.fd_step)).T
                residual = target - response[0]
                # Solve in output space; the value is a local, unconstrained
                # heuristic, not a physical feasibility certificate.
                solution = matrix.T @ np.linalg.solve(
                    matrix @ matrix.T + c.damping * np.eye(len(target)), residual)
                error = residual - matrix @ solution
                ridge_value = float(error @ error + c.damping * (solution @ solution))
                value = ridge_value if c.successor_value == "ridge" else float(np.min(probe_scores))
                reachability.append({"candidate": index, "value": value, "tail_steps": tail_length,
                                     "ridge_value": ridge_value, "probe_min": float(np.min(probe_scores)),
                                     "matrix": matrix.tolist(), "tokens": probe_cost})
        alternatives = retention_choices(scores, values, reachability, count, c.greedy_slots, c.seed + depth)
        chosen = alternatives[c.retention_policy]
        layers.append({"depth": depth, "responses": response_records, "reachability": reachability,
                       "probe_shortlist": shortlist, "retention_alternatives": alternatives,
                       "response_added": sorted(set(alternatives["response"]) - set(alternatives["diverse"])),
                       "response_removed": sorted(set(alternatives["diverse"]) - set(alternatives["response"])),
                       "candidates": [{"chain": item["chain"].tolist(), "path": item["path"],
                                       "feature": values[i].tolist(), "score": float(scores[i])}
                                      for i, item in enumerate(candidates)],
                       "retained": chosen})
        beams = [candidates[i] for i in chosen]
    remaining = (c.token_cap - reserve - used) // packets
    generations, starts = [], []
    if remaining < min_start_samples:
        raise RuntimeError("insufficient budget for iterative refinement")
    if remaining:
        seeds = [{"chain": best_chain.copy(), "score": best_score, "source": best_source}]
        candidates_for_refine = warm_seeds + [
            dict(candidate, source="retained_final_beam")
            for candidate in sorted(beams, key=lambda item: item["score"])]
        for candidate in candidates_for_refine:
            if len(seeds) >= min(c.refine_starts, remaining // min_start_samples):
                break
            if not any(np.array_equal(candidate["chain"], seed["chain"]) for seed in seeds):
                seeds.append({"chain": candidate["chain"].copy(), "score": candidate["score"],
                              "source": candidate["source"]})
        def refinement_score(chains):
            return evaluate(chains, "full_refine")[1]

        refined = pc_shared_multistart_cem.optimize(
            refinement_score, np.empty((0, dim), np.float32),
            np.stack([item["chain"] for item in seeds]), remaining,
            seed=c.seed, batch_size=c.refine_batch, elites=c.refine_elites)
        starts = [dict(record, source=item["source"])
                  for record, item in zip(refined["starts"], seeds, strict=True)]
        generations = [dict(record, restart=i)
                       for i, start in enumerate(starts) for record in start["generations"]]
    incumbent, incumbent_source = best_chain.copy(), best_source
    comparison = incumbent.copy()
    comparison[:c.segment_length] = 0
    _, paired = evaluate(np.stack((incumbent, comparison)), "paired", final=True)
    identical = bool(np.array_equal(incumbent, comparison))
    abstained = bool(identical or paired[1] <= paired[0])
    return {"method": "recursive_reachability_v25_comparable_reachability_development", "config": asdict(c), "layers": layers,
            "incumbent": incumbent.tolist(), "incumbent_source": incumbent_source,
            "comparison": comparison.tolist(), "paired_scores": paired.tolist(), "abstained": abstained,
            "selected": (comparison if abstained else incumbent).tolist(),
            "selected_score": float(paired[int(abstained)]), "refinement": generations, "refinement_starts": starts,
            "resource": {"prediction_tokens": used, "prediction_calls": calls, "full_samples": full_samples,
                         "svd_calls": svds, "svd_seconds": svd_seconds,
                         "wall_seconds": time.monotonic()-started, "token_cap": c.token_cap}}
