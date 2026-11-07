"""Same-budget zero-reference retention for deadline-limited RGB search."""
import numpy as np
import torch

from experiments import pc_rgb_timed_prefix_search_20261107_codex as timed
from experiments.pc_input_permissions import RGB, RGBLeWM
from experiments.pc_rgb_bound_prefix_effect_20261107_codex import score_prefix_effects
from experiments.pc_scored_candidate_selection_20261107_codex import ScoredCandidate, select_scored

ARMS = ("lookahead", "centered", "effect", "shuffled")


def choose_retained(costs, strategy, seed):
    costs = np.asarray(costs)
    if costs.shape != (3, 3) or not np.isfinite(costs).all() or strategy not in ARMS:
        raise ValueError("invalid common costs or strategy")
    order = np.argsort(costs.min(axis=1), kind="stable")
    if strategy == "lookahead":
        return order[:2].tolist()
    if strategy == "centered":
        vector = costs[:, 1:] - costs[:, 1:].mean(axis=1, keepdims=True)
    else:
        vector = costs[:, :1] - costs[:, 1:]
        if strategy == "shuffled":
            rng = np.random.default_rng(seed)
            vector = np.stack([row[rng.permutation(2)] for row in vector])
    first = int(order[0])
    distances = np.square(vector - vector[first]).sum(axis=1)
    distances[first] = -np.inf
    return [first, int(np.argmax(distances))]


def search(score, parents, probes, tails, *, prefix, remaining, strategy, seed):
    parents, probes, tails = (np.asarray(a, dtype=np.float32) for a in (parents, probes, tails))
    if (parents.shape != (3, remaining, 2) or probes.shape != (2, remaining-prefix, 2)
            or tails.shape != probes.shape or prefix not in (5, 10)
            or remaining not in (10, 15, 20, 25) or prefix >= remaining
            or remaining % 5 or type(seed) is not int or strategy not in ARMS):
        raise ValueError("invalid candidates or clock")
    if any(not np.isfinite(x).all() or np.abs(x).max() > 1 for x in (parents, probes, tails)):
        raise ValueError("invalid raw actions")

    def checked(chains):
        values = np.asarray(score(chains))
        if values.shape != (len(chains),) or not np.isfinite(values).all():
            raise ValueError("invalid scorer")
        return values

    parent_cost = checked(parents)
    ref = np.zeros_like(probes[0])
    described = score_prefix_effects(checked, parents[:, :prefix], ref, probes, remaining_steps=remaining)
    probe_cost = np.column_stack((described["hold_cost"], described["probe_cost"]))
    kept = choose_retained(probe_cost, strategy, seed)
    tail_chains = np.concatenate((np.broadcast_to(parents[kept, None, :prefix], (2, 2, prefix, 2)),
                                  np.broadcast_to(tails[None], (2, *tails.shape))), axis=2).reshape(4, remaining, 2)
    tail_cost = checked(tail_chains)
    candidates = [ScoredCandidate("parent", i, float(x)) for i, x in enumerate(parent_cost)]
    candidates += [ScoredCandidate("probe", i, float(x)) for i, x in enumerate(probe_cost.ravel())]
    candidates += [ScoredCandidate("tail", i, float(x)) for i, x in enumerate(tail_cost)]
    best = select_scored(candidates)
    chosen = {"parent": parents, "probe": described["chains"].reshape(9, remaining, 2),
              "tail": tail_chains}[best.source][best.index].copy()
    return {"actions": chosen, "choice": best, "retained_prefixes": kept, "parent_cost": parent_cost,
            "probe_cost": probe_cost, "tail_cost": tail_cost, "effects": described["effects"],
            "model_queries": 16, "mode": "program", "prefix_steps": prefix}


def compile_rgb(boundary, current, *, scaler, old_tail, candidate_seed, strategy_seed, strategy, remaining_steps):
    if (not isinstance(boundary, RGBLeWM) or boundary.audit.track_id != RGB
            or set(current) != {"pixels", "goal", "action"} or strategy not in ARMS
            or remaining_steps not in (5, 10, 15, 20, 25)):
        raise ValueError("invalid RGB permission or clock")
    forward_calls = 0
    packet_steps = 0

    def score(chains):
        nonlocal forward_calls, packet_steps
        normal = np.asarray(scaler.transform(chains.reshape(-1, 2)), dtype=np.float32)
        normal = normal.reshape(1, len(chains), remaining_steps // 5, 10)
        value, _ = boundary.score(current, torch.as_tensor(normal, device=current["pixels"].device))
        forward_calls += 1
        packet_steps += len(chains) * (remaining_steps // 5)
        return value.detach().cpu().numpy().reshape(-1)

    if remaining_steps == 5:
        parents, _, _ = timed.build(candidate_seed, old_tail, 5, count=16)
        if old_tail is None:
            parents[0] = 0
        parents = parents[:, :5]
        costs = score(parents)
        if costs.shape != (16,) or not np.isfinite(costs).all():
            raise ValueError("invalid fallback score")
        index = int(np.argmin(costs))
        result = {"actions": parents[index].copy(), "choice": ScoredCandidate("fallback", index, float(costs[index])),
                  "retained_prefixes": [], "parent_cost": costs, "probe_cost": np.empty((0, 0)),
                  "tail_cost": np.empty(0), "effects": np.empty((0, 0)), "model_queries": 16,
                  "mode": "direct_fallback", "prefix_steps": None}
    else:
        prefix = min(10, remaining_steps - 5)
        parents, probes, tails = timed.build(candidate_seed, old_tail, prefix)
        result = search(score, parents[:, :remaining_steps], probes[:, :remaining_steps-prefix],
                        tails[:, :remaining_steps-prefix], prefix=prefix, remaining=remaining_steps,
                        strategy=strategy, seed=strategy_seed)
    result.update({"forward_calls": forward_calls, "model_packet_steps": packet_steps,
                   "remaining_steps": remaining_steps})
    return result
