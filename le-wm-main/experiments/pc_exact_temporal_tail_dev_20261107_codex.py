"""Compare temporally coherent suffix search with sealed per-step CEM."""
import json
import time
from pathlib import Path

import numpy as np

from experiments import pc_exact_action_identity_retention_dev_20261106_codex as prior

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/pc_exact_prefix_coverage_cause_dev_20261106_codex.json"
OUTPUT = ROOT / "results/pc_exact_temporal_tail_dev_20261107_codex.json"
PROTOCOL = ROOT.parent / "docs/PC_EXACT_TEMPORAL_TAIL_DEV_PROTOCOL_20261107_CODEX.md"


def optimize(task, prefix, initial_tail, seed, samples=384):
    """CEM over shared two-step offsets around a fixed legal starting tail."""
    prefix = np.asarray(prefix, dtype=np.float64)
    initial = np.asarray(initial_tail, dtype=np.float64)
    if prefix.shape != (5, 2) or initial.shape != (10, 2):
        raise ValueError("expected fixed five-step prefix and ten-step suffix")
    if not np.isfinite(initial).all() or np.max(np.abs(initial)) > 1:
        raise ValueError("invalid initial tail")
    rng = np.random.default_rng(seed)
    center = np.zeros((5, 2))
    scale = np.full_like(center, 1 / 3)
    actions, costs = [], []
    goal = np.asarray(task["goal"])
    for start in range(0, samples, 12):
        count = min(12, samples - start)
        offsets = np.clip(center + rng.normal(size=(count, 5, 2)) * scale, -1, 1)
        offsets[0] = center
        tails = np.clip(initial[None] + np.repeat(offsets, 2, axis=1), -1, 1)
        chains = np.concatenate((np.broadcast_to(prefix, (count, 5, 2)), tails), axis=1)
        values = np.linalg.norm(prior.exact.endpoints(task, chains) - goal, axis=1)
        if not np.isfinite(values).all():
            raise ValueError("nonfinite cost")
        actions.extend(chains.tolist())
        costs.extend(values.tolist())
        if count > 1:
            elite = offsets[np.argsort(values, kind="stable")[:min(4, count)]]
            center = elite.mean(axis=0)
            scale = np.maximum(elite.std(axis=0), 1e-5)
    return {"actions": actions, "costs": costs, "best": int(np.argmin(costs))}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    prior_data = json.loads(prior.OUTPUT.read_text())
    source = json.loads(SOURCE.read_text())
    if prior_data["dependencies"] != {p: prior.sha(p) for p in prior_data["dependencies"]}:
        raise ValueError("prior dependencies drift")
    if source["dependencies"] != {p: prior.sha(p) for p in source["dependencies"]}:
        raise ValueError("coverage dependencies drift")
    paths = [Path(__file__), PROTOCOL, SOURCE, prior.OUTPUT, Path(prior.__file__),
             Path(prior.exact.__file__), Path(prior.exact.wall.__file__),
             Path(prior.exact.wall.physics.__file__)]
    deps = {str(p): prior.sha(p) for p in paths}
    if len(source["rows"]) != 24 or len(prior_data["rows"]) != 24:
        raise ValueError("unexpected panel")
    print("Compare fixed-prefix temporal versus per-step CEM; oracle is diagnostic only.", flush=True)
    rows = []
    for old, baseline in zip(prior_data["rows"], source["rows"], strict=True):
        if old["root"] != baseline["root"] or old["task"] != baseline["task"]:
            raise ValueError("root mismatch")
        start_time = time.perf_counter()
        pool = np.asarray(old["pool"], dtype=np.float64)
        temporal = [optimize(old["task"], chain[:5], chain[5:], 261113000 + old["root"] * 8 + i)
                    for i, chain in enumerate(pool)]
        for i, result in enumerate(temporal):
            np.testing.assert_array_equal(result["actions"][0], pool[i])
            np.testing.assert_array_equal(baseline["pool_longer"][i]["actions"][0], pool[i])
        rows.append({"root": old["root"], "task": old["task"],
                     "selected_lookahead_prefixes": old["selected"]["lookahead"],
                     "temporal": temporal, "elapsed_seconds": time.perf_counter() - start_time})
        print(f"root={old['root']} complete", flush=True)
    if deps != {p: prior.sha(p) for p in deps}:
        raise ValueError("dependency drift")
    with OUTPUT.open("x") as file:
        json.dump({"dependencies": deps, "rows": rows,
                   "scope": "exposed exact wall development; no descriptor or RGB claim"},
                  file, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()
