"""Post-hoc prefix intervention on sealed exact-wall candidates."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_exact_action_identity_retention_dev_20261106_codex as prior
from experiments import pc_exact_temporal_tail_dev_20261107_codex as trial

PROTOCOL = trial.ROOT.parent / "docs/PC_WALL_CROSSING_TAIL_INTERVENTION_PROTOCOL_20261107_CODEX.md"
OUTPUT = trial.ROOT / "results/pc_wall_crossing_tail_intervention_20261107_codex.json"


def choose(task, banks):
    """Evaluator-only: select earliest observed crossing, then original terminal cost."""
    options = []
    for arm, searches in banks:
        for prefix_id, search in enumerate(searches):
            chains = np.asarray(search["actions"], dtype=np.float64)
            costs = np.asarray(search["costs"], dtype=np.float64)
            padded = np.pad(chains, ((0, 0), (0, 1), (0, 0))) * prior.exact.wall.HIGH
            trajectories, _ = prior.exact.wall.rollout(task["start"], padded)
            flags = trajectories[:, 1:15, 0] > 0
            for index in np.flatnonzero(flags.any(axis=1)):
                step = int(np.argmax(flags[index]) + 1)
                options.append((step, float(costs[index]), arm, prefix_id, int(index),
                                chains[index], trajectories[index, step].copy()))
    if not options:
        return None
    return min(options, key=lambda item: item[:5])


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    baseline, temporal = json.loads(trial.SOURCE.read_text()), json.loads(trial.OUTPUT.read_text())
    for source in (baseline, temporal):
        if source["dependencies"] != {p: prior.sha(p) for p in source["dependencies"]}:
            raise ValueError("source drift")
    paths = (Path(__file__), PROTOCOL, trial.SOURCE, trial.OUTPUT, Path(prior.__file__),
             Path(prior.exact.__file__), Path(prior.exact.wall.__file__), Path(prior.exact.wall.physics.__file__))
    dependencies = {str(p): prior.sha(p) for p in paths}
    print("Post-hoc exact crossing-prefix vs same-duration zero-prefix with equal tail budget.", flush=True)
    rows = []
    for b, t in zip(baseline["rows"], temporal["rows"], strict=True):
        if b["root"] != t["root"] or b["task"] != t["task"]:
            raise ValueError("root mismatch")
        candidate = choose(t["task"], (("per_step", b["pool_longer"]), ("temporal", t["temporal"])))
        if candidate is None:
            rows.append({"root": t["root"], "eligible": False})
            continue
        step, initial_cost, arm, prefix_id, idx, chain, successor = candidate
        tail = chain[step:]
        seed = 261114000 + t["root"]
        crossed = prior.optimize(t["task"], chain[:step], tail, 384, seed)
        zero = prior.optimize(t["task"], np.zeros((step, 2)), tail, 384, seed)
        np.testing.assert_array_equal(crossed["actions"][0], chain)
        rows.append({"root": t["root"], "task": t["task"], "eligible": True,
                     "selection": {"arm": arm, "prefix_index": prefix_id, "candidate_index": idx,
                                   "crossing_step": step, "initial_chain": chain.tolist(),
                                   "initial_cost": initial_cost, "prefix_successor": successor.tolist()},
                     "crossed": crossed, "zero": zero})
        print(f"root={t['root']} prefix={step}, equal-budget tail optimization complete", flush=True)
    if dependencies != {p: prior.sha(p) for p in dependencies}:
        raise ValueError("dependency drift")
    with OUTPUT.open("x") as stream:
        json.dump({"dependencies": dependencies, "rows": rows,
                   "scope": "post-hoc state-selected diagnostic, not deployable planner"},
                  stream, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()
