"""Sequential finite-budget effect search; only queried B costs enter later choices."""
import itertools
import json
import time
from pathlib import Path

import numpy as np

from experiments import pc_exact_action_identity_retention_dev_20261106_codex as base
from experiments import pc_exact_descriptor_combination_proposal_20261107_codex as prior
from experiments import pc_exact_goal_relative_bridge_20261107_codex as geom

PROTOCOL = geom.ROOT.parent / "docs/PC_EXACT_SEQUENTIAL_EFFECT_SEARCH_PROTOCOL_20261107_CODEX.md"
OUTPUT = geom.ROOT / "results/pc_exact_sequential_effect_search_20261107_codex.json"
ARMS = ("static", "sequential", "shuffled", "row_mean", "random")


def tasks():
    return [{"start": [-.5, y, t], "goal": [.5, gy]}
            for y, t, gy in itertools.product((-.28, -.08, .08, .28),
                                               (-np.pi / 8, 0., np.pi / 8), (-.3, .3))]


def next_pair(a, b, a_cost, history, step, mode):
    """Returns one new pair using only A and explicitly queried per-arm history."""
    a, b, a_cost = np.asarray(a), np.asarray(b), np.asarray(a_cost)
    if a.shape != (8, 10, 2) or b.shape != (64, 10, 2) or a_cost.shape != (8, 8):
        raise ValueError("bad candidate shapes")
    if mode not in ("descriptor", "row_mean") or not all(np.isfinite(x).all() for x in (a, b, a_cost)):
        raise ValueError("bad predictor input")
    if step != len(history) or step < 0 or step >= 8:
        raise ValueError("invalid history length")
    seen = set()
    for idx, cost in history:
        if idx in seen or not 0 <= idx < 512 or not np.isfinite(cost):
            raise ValueError("invalid history")
        seen.add(idx)
    values = np.empty((8, 64))
    for i in range(8):
        current = [(idx % 64, cost) for idx, cost in history if idx // 64 == i]
        tails = np.concatenate((a, np.asarray([b[j] for j, _ in current]).reshape(-1, 10, 2)))
        costs = np.concatenate((a_cost[i], np.asarray([c for _, c in current])))
        distance = ((b[:, None] - tails[None]) ** 2).sum(axis=(2, 3))
        if step % 2:
            values[i] = -distance.min(axis=1)
        elif mode == "row_mean":
            values[i] = costs.mean()
        else:
            near = np.argsort(distance, axis=1, kind="stable")[:, :3]
            values[i] = costs[near].mean(axis=1)
    values.ravel()[list(seen)] = np.inf
    return int(np.argmin(values))


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    paths = [Path(__file__), PROTOCOL, Path(prior.__file__), Path(base.__file__), Path(geom.__file__),
             Path(base.exact.__file__), Path(base.exact.wall.__file__), Path(base.exact.wall.physics.__file__)]
    dependencies = {str(p): base.sha(p) for p in paths}
    rows = []
    print("Sequential action-effect query development; no unqueried B scores exposed.", flush=True)
    for root, task in enumerate(tasks()):
        offsets = np.random.default_rng(2740000 + root).uniform(-2, 2, 80)
        chains = np.stack([geom.make_chain(task["start"], task["goal"], o)[0] for o in offsets])
        if chains.shape != (80, 15, 2) or not np.isfinite(chains).all() or np.abs(chains).max() > 1 + 1e-12:
            raise ValueError("invalid actions")
        pool, a, b = chains[:8], chains[8:16, 5:], chains[16:, 5:]
        def score(actions):
            return np.linalg.norm(base.exact.endpoints(task, actions) - np.asarray(task["goal"]), axis=1)
        parent = score(pool)
        a_chains = np.concatenate((np.broadcast_to(pool[:, None, :5], (8, 8, 5, 2)),
                                   np.broadcast_to(a[None], (8, 8, 10, 2))), axis=2).reshape(64, 15, 2)
        ac = score(a_chains).reshape(8, 8)
        common = np.r_[parent, ac.ravel()]
        common_chains = np.concatenate((pool, a_chains))
        best_common = int(np.argmin(common))
        pred = prior.proposal_predictions(a, b, ac, parent, 2741000 + root)
        fixed = {"static": prior.choose(pred["descriptor"]),
                 "random": np.random.default_rng(2742000 + root).choice(512, 8, replace=False).tolist()}
        rng = np.random.default_rng(2741000 + root)
        scrambled = np.stack([row[rng.permutation(8)] for row in ac])
        outcomes = {}
        for arm in ARMS:
            history = []
            begin = time.perf_counter()
            for k in range(8):
                if arm in fixed:
                    idx = fixed[arm][k]
                else:
                    costs = scrambled if arm == "shuffled" else ac
                    mode = "row_mean" if arm == "row_mean" else "descriptor"
                    idx = next_pair(a, b, costs, history, k, mode)
                action = np.concatenate((pool[idx // 64, :5], b[idx % 64]))
                value = float(score(action[None])[0])
                history.append((int(idx), value))
            elapsed = time.perf_counter() - begin
            if len({idx for idx, _ in history}) != 8:
                raise ValueError("duplicate candidate query")
            q = int(np.argmin([v for _, v in history]))
            idx, val = history[q]
            use_q = val < common[best_common]
            action = np.concatenate((pool[idx // 64, :5], b[idx % 64])) if use_q else common_chains[best_common]
            outcomes[arm] = {"history": history, "score": val if use_q else float(common[best_common]),
                             "chain": action.tolist(), "origin": "queried_B" if use_q else "common",
                             "query_and_selection_seconds": elapsed}
        rows.append({"root": root, "task": task, "offsets": offsets.tolist(), "parent_cost": parent.tolist(),
                     "a_cost": ac.tolist(), "outcomes": outcomes, "per_arm_model_steps": 1200})
    if dependencies != {p: base.sha(p) for p in dependencies}:
        raise ValueError("dependency drift")
    summary = {}
    for arm in ARMS:
        scores = [r["outcomes"][arm]["score"] for r in rows]
        summary[arm] = {"successes": sum(x < .05 for x in scores), "mean_distance": float(np.mean(scores))}
    success = [r["outcomes"]["sequential"]["score"] < .05 for r in rows]
    for arm in ("static", "shuffled", "row_mean"):
        other = [r["outcomes"][arm]["score"] < .05 for r in rows]
        summary[arm]["sequential_only"] = sum(x and not y for x, y in zip(success, other, strict=True))
        summary[arm]["comparator_only"] = sum(y and not x for x, y in zip(success, other, strict=True))
    passed = all(summary[a]["sequential_only"] >= 2 and summary[a]["comparator_only"] <= 1
                 for a in ("static", "shuffled", "row_mean"))
    passed &= summary["sequential"]["successes"] >= summary["random"]["successes"]
    with OUTPUT.open("x") as stream:
        json.dump({"dependencies": dependencies, "rows": rows, "summary": summary,
                   "development_exit": bool(passed),
                   "scope": "exact state and dynamics; exposed family; not RGB-only or independent"},
                  stream, separators=(",", ":"), allow_nan=False)
    print(json.dumps({"summary": summary, "development_exit": bool(passed)}, indent=2))


if __name__ == "__main__":
    run()
