"""Bounded fixed-prefix compiler using the audited RGB LeWM boundary.

This is an integration primitive, not a validated planning method. All input
arrays are raw bounded actions; goal/state truth is never accepted here.
"""
import numpy as np
import torch

from experiments.pc_input_permissions import RGBLeWM
from experiments.pc_scored_candidate_selection_20261107_codex import ScoredCandidate, select_scored


def compile_fixed_prefix(boundary, context, parents, probes, tail_bank, *, scaler,
                         prefix_steps, retain_count, strategy, max_queries, seed):
    """Score parents and shared probes, then query tails only for retained prefixes."""
    if not isinstance(boundary, RGBLeWM):
        raise ValueError("audited RGBLeWM boundary required")
    if set(context) != {"pixels", "goal", "action"}:
        raise ValueError("invalid planner observation")
    parents, probes, tail_bank = (np.asarray(x, dtype=np.float32) for x in
                                  (parents, probes, tail_bank))
    if parents.ndim != 3 or probes.ndim != 3 or tail_bank.ndim != 3:
        raise ValueError("time-by-action chains required")
    n, horizon, action_dim = parents.shape
    if (n < 2 or action_dim < 1 or horizon < 1 or
            type(prefix_steps) is not int or not 0 < prefix_steps < horizon or
            probes.shape[0] < 1 or tail_bank.shape[0] < 1 or
            probes.shape[1:] != (horizon - prefix_steps, action_dim) or
            tail_bank.shape[1:] != probes.shape[1:] or horizon % 5 != 0):
        raise ValueError("invalid fixed-prefix action clock")
    if type(retain_count) is not int or not 1 <= retain_count <= n:
        raise ValueError("invalid retention count")
    if strategy not in ("parent", "lookahead", "vector", "shuffled"):
        raise ValueError("unknown retention strategy")
    if any(not np.isfinite(a).all() or np.any(np.abs(a) > 1) for a in (parents, probes, tail_bank)):
        raise ValueError("nonfinite or illegal raw actions")
    needed = n + n * len(probes) + retain_count * len(tail_bank)
    if type(max_queries) is not int or max_queries < needed:
        raise ValueError("insufficient declared query budget")
    if not isinstance(seed, int):
        raise ValueError("explicit integer seed required")
    device = context["pixels"].device
    forward_calls = 0

    def score(raw):
        nonlocal forward_calls
        raw = np.asarray(raw, dtype=np.float32)
        norm = np.asarray(scaler.transform(raw.reshape(-1, action_dim)), dtype=np.float32).reshape(raw.shape)
        if not np.isfinite(norm).all():
            raise ValueError("nonfinite normalized action")
        chunks = []
        for start in range(0, len(raw), 32):
            t = torch.as_tensor(norm[start:start + 32].reshape(1, -1, horizon // 5, 5 * action_dim),
                                device=device)
            value, _ = boundary.score(context, t)
            values = value.detach().cpu().numpy().reshape(-1)
            if len(values) != len(t[0]) or not np.isfinite(values).all():
                raise ValueError("model returned invalid candidate costs")
            chunks.append(values)
            forward_calls += 1
        return np.concatenate(chunks)

    parent_cost = score(parents)
    prefix = parents[:, :prefix_steps]
    probe_chains = np.concatenate((np.broadcast_to(prefix[:, None], (n, len(probes), prefix_steps, action_dim)),
                                   np.broadcast_to(probes[None], (n, *probes.shape))), axis=2)
    probe_cost = score(probe_chains.reshape(-1, horizon, action_dim)).reshape(n, len(probes))
    if strategy == "parent":
        kept = np.argsort(parent_cost, kind="stable")[:retain_count].tolist()
    else:
        order = np.argsort(probe_cost.min(axis=1), kind="stable")
        if strategy == "lookahead":
            kept = order[:retain_count].tolist()
        else:
            vectors = probe_cost - probe_cost.mean(axis=1, keepdims=True)
            if strategy == "shuffled":
                rng = np.random.default_rng(seed)
                vectors = np.stack([v[rng.permutation(len(v))] for v in vectors])
            kept = [int(order[0])]
            while len(kept) < retain_count:
                distances = np.square(vectors[:, None] - vectors[kept][None]).sum(axis=2).min(axis=1)
                distances[kept] = -np.inf
                kept.append(int(np.argmax(distances)))
    tails = np.concatenate((np.broadcast_to(prefix[kept, None],
                                             (retain_count, len(tail_bank), prefix_steps, action_dim)),
                            np.broadcast_to(tail_bank[None], (retain_count, *tail_bank.shape))), axis=2)
    tails = tails.reshape(-1, horizon, action_dim)
    tail_cost = score(tails)
    candidates = [ScoredCandidate("parent", i, float(v)) for i, v in enumerate(parent_cost)]
    candidates += [ScoredCandidate("probe", i, float(v)) for i, v in enumerate(probe_cost.ravel())]
    candidates += [ScoredCandidate("tail", i, float(v)) for i, v in enumerate(tail_cost)]
    chosen = select_scored(candidates)
    selected = {"parent": parents, "probe": probe_chains.reshape(-1, horizon, action_dim),
                "tail": tails}[chosen.source][chosen.index].copy()
    return {"actions": selected, "choice": chosen, "retained_prefixes": kept,
            "parent_cost": parent_cost, "probe_cost": probe_cost, "tail_cost": tail_cost,
            "model_queries": needed, "forward_calls": forward_calls}
