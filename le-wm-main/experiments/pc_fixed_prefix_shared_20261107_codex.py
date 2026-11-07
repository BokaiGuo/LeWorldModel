"""Track-independent candidate construction and retention; no model inputs."""
import numpy as np

from experiments.pc_scored_candidate_selection_20261107_codex import ScoredCandidate, select_scored


def search(score, parents, probes, tails, *, prefix_steps, retain_count, strategy, budget, seed):
    parents, probes, tails = (np.asarray(x, dtype=np.float32) for x in (parents, probes, tails))
    if parents.ndim != 3 or probes.ndim != 3 or tails.ndim != 3:
        raise ValueError("expected action chains")
    n, horizon, width = parents.shape
    if (n < 2 or width < 1 or type(prefix_steps) is not int or not 0 < prefix_steps < horizon
            or horizon % 5 or not len(probes) or not len(tails)
            or probes.shape[1:] != (horizon - prefix_steps, width)
            or tails.shape[1:] != probes.shape[1:]):
        raise ValueError("invalid action shape or clock")
    if any(not np.isfinite(a).all() or np.any(np.abs(a) > 1) for a in (parents, probes, tails)):
        raise ValueError("illegal action")
    if type(retain_count) is not int or not 1 <= retain_count <= n:
        raise ValueError("invalid retain count")
    if strategy not in ("parent", "lookahead", "vector", "shuffled") or type(seed) is not int:
        raise ValueError("invalid strategy or seed")
    needed = n + n * len(probes) + retain_count * len(tails)
    if type(budget) is not int or budget < needed:
        raise ValueError("insufficient budget")

    def checked(batch):
        costs = np.asarray(score(batch))
        if costs.shape != (len(batch),) or not np.isfinite(costs).all():
            raise ValueError("invalid scorer output")
        return costs

    parent_cost = checked(parents)
    prefix = parents[:, :prefix_steps]
    bank = np.concatenate((np.broadcast_to(prefix[:, None], (n, len(probes), prefix_steps, width)),
                           np.broadcast_to(probes[None], (n, *probes.shape))), axis=2)
    probe_cost = checked(bank.reshape(-1, horizon, width)).reshape(n, len(probes))
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
                dist = np.square(vectors[:, None] - vectors[kept][None]).sum(axis=2).min(axis=1)
                dist[kept] = -np.inf
                kept.append(int(np.argmax(dist)))
    tail_chains = np.concatenate((np.broadcast_to(prefix[kept, None],
                                                  (retain_count, len(tails), prefix_steps, width)),
                                  np.broadcast_to(tails[None], (retain_count, *tails.shape))), axis=2)
    tail_chains = tail_chains.reshape(-1, horizon, width)
    tail_cost = checked(tail_chains)
    candidates = [ScoredCandidate("parent", i, float(v)) for i, v in enumerate(parent_cost)]
    candidates += [ScoredCandidate("probe", i, float(v)) for i, v in enumerate(probe_cost.ravel())]
    candidates += [ScoredCandidate("tail", i, float(v)) for i, v in enumerate(tail_cost)]
    best = select_scored(candidates)
    action = {"parent": parents, "probe": bank.reshape(-1, horizon, width), "tail": tail_chains}[best.source][best.index]
    return {"actions": action.copy(), "choice": best, "retained_prefixes": kept,
            "parent_cost": parent_cost, "probe_cost": probe_cost, "tail_cost": tail_cost,
            "model_queries": needed}
