"""Equal-budget full-chain temporal-block CEM on exposed exact-wall tasks."""
import json
import time
from pathlib import Path

import numpy as np

from experiments import pc_exact_action_identity_retention_dev_20261106_codex as prior

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT.parent / "docs/PC_EXACT_BLOCK_CHAIN_EXPLORATION_PROTOCOL_20261107_CODEX.md"
OUTPUT = ROOT / "results/pc_exact_block_chain_exploration_20261107_codex.json"


def block_chains(blocks):
    blocks = np.asarray(blocks)
    if blocks.ndim != 3 or blocks.shape[1:] != (3, 2) or not np.isfinite(blocks).all():
        raise ValueError("expected finite three-block actions")
    if np.any(np.abs(blocks) > 1):
        raise ValueError("out of bounds")
    return np.repeat(blocks, 5, axis=1)


def block_cem(task, seed):
    rng = np.random.default_rng(seed)
    mean = np.zeros((3, 2))
    scale = np.full((3, 2), 1 / 3)
    actions, costs = [], []
    goal = np.asarray(task["goal"])
    for _ in range(32):
        candidates = np.clip(mean + rng.normal(size=(12, 3, 2)) * scale, -1, 1)
        candidates[0] = mean
        chains = block_chains(candidates)
        values = np.linalg.norm(prior.exact.endpoints(task, chains) - goal, axis=1)
        actions.extend(chains.tolist())
        costs.extend(values.tolist())
        elite = candidates[np.argsort(values, kind="stable")[:4]]
        mean = elite.mean(axis=0)
        scale = np.maximum(elite.std(axis=0), 1e-5)
    return {"actions": actions, "costs": costs, "best": int(np.argmin(costs))}


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    dependencies = {str(p): prior.sha(p) for p in
                    (Path(__file__), PROTOCOL, Path(prior.__file__), Path(prior.panel.__file__),
                     Path(prior.exact.__file__), Path(prior.exact.wall.__file__),
                     Path(prior.exact.wall.physics.__file__))}
    tasks = prior.panel.tasks()
    if len(tasks) != 24:
        raise ValueError("panel drift")
    print("Compare fixed-budget whole-chain CEM with and without five-step temporal blocks.", flush=True)
    rows = []
    for root, task in enumerate(tasks):
        results = {}
        for name in ("step", "block"):
            started = time.perf_counter()
            result = (prior.optimize(task, [], np.zeros((15, 2)), 384, 2711000 + root)
                      if name == "step" else block_cem(task, 2712000 + root))
            result["seconds"] = time.perf_counter() - started
            actions = np.asarray(result["actions"])
            costs = np.asarray(result["costs"])
            if actions.shape != (384, 15, 2) or costs.shape != (384,):
                raise ValueError("budget mismatch")
            recost = np.linalg.norm(prior.exact.endpoints(task, actions) - task["goal"], axis=1)
            np.testing.assert_allclose(costs, recost, atol=1e-12, rtol=0)
            padded = np.pad(actions, ((0, 0), (0, 1), (0, 0))) * prior.exact.wall.HIGH
            trajectory, _ = prior.exact.wall.rollout(task["start"], padded)
            result["crossed"] = np.any(trajectory[:, 1:16, 0] > 0, axis=1).tolist()
            result["tokens"] = 1152
            results[name] = result
        rows.append({"root": root, "task": task, "arms": results})
        print(f"root {root}: step={min(results['step']['costs']):.4f}, "
              f"block={min(results['block']['costs']):.4f}", flush=True)
    if dependencies != {path: prior.sha(path) for path in dependencies}:
        raise ValueError("dependency drift")
    with OUTPUT.open("x") as stream:
        json.dump({"dependencies": dependencies, "rows": rows,
                   "scope": "exposed exact dynamics development, no RGB or independent evidence"},
                  stream, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    run()
