"""CEM variant that includes an exactly representable incumbent in elite updates."""
import hashlib

import numpy as np

from experiments.pc_tworoom_vector_retention_posthoc_20261107_codex import retention

ARMS = ("baseline", "lookahead", "vector")


def search(score, *, remaining, old_tail, seed, arm):
    if remaining not in (5, 10, 15, 20, 25) or type(seed) is not int or arm not in ARMS:
        raise ValueError("invalid clock, seed or arm")
    blocks = remaining // 5
    rng = np.random.default_rng(seed)
    calls = []

    def evaluate(actions):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 3 or actions.shape[1:] != (remaining, 2):
            raise ValueError("wrong action shape")
        if not np.isfinite(actions).all() or (np.abs(actions) > 1).any():
            raise ValueError("invalid action")
        values = np.asarray(score(actions), dtype=np.float64)
        if values.shape != (len(actions),) or not np.isfinite(values).all():
            raise ValueError("invalid scores")
        calls.append(len(actions))
        return values

    old = None
    representable = False
    if old_tail is not None:
        old = np.asarray(old_tail, dtype=np.float32)
        if old.shape != (remaining, 2) or not np.isfinite(old).all() or (np.abs(old) > 1).any():
            raise ValueError("invalid old tail")
        old_blocks = old.reshape(blocks, 5, 2)
        representable = np.array_equal(old_blocks, np.broadcast_to(old_blocks[:, :1], old_blocks.shape))

    means = np.zeros((blocks, 2))
    std = np.ones_like(means)
    cem, cem_costs = [], []
    old_rank = None
    old_elite = False
    for iteration in range(3):
        samples = np.clip(means + rng.normal(size=(100, blocks, 2)) * std, -1, 1).astype(np.float32)
        samples[0] = means
        if iteration == 0 and representable:
            samples[0] = old_blocks[:, 0]
        actions = np.repeat(samples, 5, axis=1)
        if iteration == 0 and old is not None:
            actions[0] = old
        costs = evaluate(actions)
        elite_idx = np.argsort(costs, kind="stable")
        if iteration == 0 and old is not None:
            old_rank = int(np.flatnonzero(elite_idx == 0)[0] + 1)
            if not representable:
                elite_idx = elite_idx[elite_idx != 0]
            old_elite = bool(representable and old_rank <= 10)
        elites = samples[elite_idx[:10]]
        means, std = elites.mean(axis=0), np.maximum(elites.std(axis=0), 1e-5)
        cem.append(actions)
        cem_costs.append(costs)
    cem, cem_costs = np.concatenate(cem), np.concatenate(cem_costs)
    rand = np.repeat(rng.uniform(-1, 1, (300, blocks, 2)).astype(np.float32), 5, axis=1)
    rand_costs = evaluate(rand)
    all_actions, all_costs = [cem, rand], [cem_costs, rand_costs]
    retained = []
    if arm == "baseline" or remaining == 5:
        extra = np.repeat(rng.uniform(-1, 1, (80, blocks, 2)).astype(np.float32), 5, axis=1)
        all_actions.append(extra)
        all_costs.append(evaluate(extra))
    else:
        prefix = min(10, remaining - 5)
        parents, parent_costs = [], []
        for actions, costs in ((cem, cem_costs), (rand, rand_costs)):
            count = 0
            for index in np.argsort(costs, kind="stable"):
                if all(not np.array_equal(actions[index, :prefix], x[:prefix]) for x in parents):
                    parents.append(actions[index])
                    parent_costs.append(costs[index])
                    count += 1
                    if count == 4:
                        break
            if count != 4:
                raise ValueError("insufficient distinct prefixes")
        parents = np.asarray(parents)
        n_tail_blocks = (remaining - prefix) // 5

        def bank():
            suffix = np.repeat(rng.uniform(-1, 1, (8, n_tail_blocks, 2)).astype(np.float32), 5, axis=1)
            return np.concatenate((np.broadcast_to(parents[:, None, :prefix], (8, 8, prefix, 2)),
                                   np.broadcast_to(suffix[None], (8, 8, remaining-prefix, 2))), axis=2)

        a, b = bank(), bank()
        a_actions = a.reshape(64, remaining, 2)
        a_cost = evaluate(a_actions)
        retained = retention(a_cost.reshape(8, 8), np.asarray(parent_costs), seed + 100000)[arm]
        b_actions = b[retained].reshape(16, remaining, 2)
        b_cost = evaluate(b_actions)
        all_actions.extend((a_actions, b_actions))
        all_costs.extend((a_cost, b_cost))
    actions, costs = np.concatenate(all_actions), np.concatenate(all_costs)
    if len(actions) != 680 or sum(calls) != 680:
        raise ValueError("budget accounting mismatch")
    best = int(np.argmin(costs))
    return {"actions": actions[best].copy(), "cost": float(costs[best]), "winner": best,
            "all_scores": costs, "retained": retained, "query_batches": calls,
            "model_queries": len(costs), "model_packet_steps": len(costs) * blocks,
            "actions_sha256": hashlib.sha256(np.ascontiguousarray(actions[best]).tobytes()).hexdigest(),
            "old_tail_included": old is not None, "old_tail_representable": representable,
            "old_tail_rank": old_rank, "old_tail_elite": old_elite}
