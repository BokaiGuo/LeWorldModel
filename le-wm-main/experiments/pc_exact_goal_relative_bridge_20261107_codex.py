"""Fixed-budget goal-relative open-loop waypoint programs for exact-wall development."""
import json
import time
from pathlib import Path

import numpy as np

from experiments import pc_exact_block_chain_exploration_20261107_codex as baseline

ROOT = baseline.ROOT
PROTOCOL = ROOT.parent / "docs/PC_EXACT_GOAL_RELATIVE_BRIDGE_PROTOCOL_20261107_CODEX.md"
OUTPUT = ROOT / "results/pc_exact_goal_relative_bridge_20261107_codex.json"


def make_chain(start, goal, offset):
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    if start.shape != (3,) or goal.shape != (2,) or not np.isfinite(np.r_[start, goal, offset]).all():
        raise ValueError("finite start pose, goal and offset required")
    delta = goal - start[:2]
    length = float(np.linalg.norm(delta))
    if length <= 1e-12:
        raise ValueError("distinct start and goal required")
    lateral = np.array([-delta[1], delta[0]])
    waypoint = (start[:2] + goal) / 2 + offset * lateral
    position, heading = start[:2].copy(), float(start[2])
    actions = []
    high = baseline.prior.exact.wall.HIGH
    for point in (waypoint, goal):
        displacement = point - position
        distance = float(np.linalg.norm(displacement))
        if distance <= 1e-12:
            continue
        angle = float(np.arctan2(displacement[1], displacement[0]))
        turn = float((angle - heading + np.pi) % (2 * np.pi) - np.pi)
        turns = int(np.ceil(abs(turn) / high[1]))
        if turns:
            actions.extend([[0., turn / turns / high[1]]] * turns)
        steps = int(np.ceil(distance / high[0]))
        actions.extend([[distance / steps / high[0], 0.]] * steps)
        position, heading = point, angle
    required = len(actions)
    return np.asarray((actions + [[0., 0.]] * 15)[:15]), waypoint, required


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    old = json.loads(baseline.OUTPUT.read_text())
    if old["dependencies"] != {p: baseline.prior.sha(p) for p in old["dependencies"]}:
        raise ValueError("baseline dependency changed")
    dependencies = {str(p): baseline.prior.sha(p) for p in
                    (Path(__file__), PROTOCOL, baseline.OUTPUT, Path(baseline.__file__),
                     Path(baseline.prior.exact.__file__), Path(baseline.prior.exact.wall.__file__),
                     Path(baseline.prior.exact.wall.physics.__file__))}
    rows = []
    print("Goal-relative waypoint programs vs sealed equal-budget step CEM; exact development only.", flush=True)
    for row in old["rows"]:
        root, task = row["root"], row["task"]
        started = time.perf_counter()
        offsets = np.random.default_rng(2713000 + root).uniform(-2., 2., 384)
        proposals = [make_chain(task["start"], task["goal"], offset) for offset in offsets]
        chains = np.stack([p[0] for p in proposals])
        goals = np.asarray([p[1] for p in proposals])
        required = np.asarray([p[2] for p in proposals])
        costs = np.linalg.norm(baseline.prior.exact.endpoints(task, chains) - task["goal"], axis=1)
        elapsed = time.perf_counter() - started
        rows.append({"root": root, "task": task, "offsets": offsets.tolist(), "waypoints": goals.tolist(),
                     "required_steps": required.tolist(), "actions": chains.tolist(), "costs": costs.tolist(),
                     "best": int(np.argmin(costs)), "tokens": 1152, "seconds": elapsed})
        print(f"root {root}: distance={costs.min():.6f}, clipped={int((required > 15).sum())}", flush=True)
    if dependencies != {path: baseline.prior.sha(path) for path in dependencies}:
        raise ValueError("dependency changed")
    with OUTPUT.open("x") as stream:
        json.dump({"dependencies": dependencies, "rows": rows,
                   "scope": "exposed exact state input, no descriptor or RGB-only evidence"},
                  stream, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()
