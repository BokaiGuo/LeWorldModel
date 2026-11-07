"""Three-segment geometric proposals on the sealed exact-wall development panel."""
import json
import time
from pathlib import Path

import numpy as np

from experiments import pc_exact_goal_relative_bridge_20261107_codex as prior

ROOT = prior.ROOT
PROTOCOL = ROOT.parent / "docs/PC_EXACT_TWO_WAYPOINT_BRIDGE_PROTOCOL_20261107_CODEX.md"
OUTPUT = ROOT / "results/pc_exact_two_waypoint_bridge_20261107_codex.json"
sha = prior.baseline.prior.sha
wall = prior.baseline.prior.exact.wall


def make_chain(start, goal, offset):
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    if start.shape != (3,) or goal.shape != (2,) or not np.isfinite(np.r_[start, goal, offset]).all():
        raise ValueError("finite start pose, goal and offset required")
    delta = goal - start[:2]
    if np.linalg.norm(delta) <= 1e-12:
        raise ValueError("distinct start and goal required")
    lateral = np.array([-delta[1], delta[0]])
    points = [start[:2] + fraction * delta + offset * lateral for fraction in (.25, .75)]
    position, heading = start[:2].copy(), float(start[2])
    actions = []
    for point in [*points, goal]:
        displacement = point - position
        distance = float(np.linalg.norm(displacement))
        if distance <= 1e-12:
            continue
        angle = float(np.arctan2(displacement[1], displacement[0]))
        turn = float((angle - heading + np.pi) % (2 * np.pi) - np.pi)
        turns = int(np.ceil(abs(turn) / wall.HIGH[1]))
        if turns:
            actions.extend([[0., turn / turns / wall.HIGH[1]]] * turns)
        steps = int(np.ceil(distance / wall.HIGH[0]))
        actions.extend([[distance / steps / wall.HIGH[0], 0.]] * steps)
        position, heading = point, angle
    return np.asarray((actions + [[0., 0.]] * 15)[:15]), np.asarray(points), len(actions)


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    old = json.loads(prior.OUTPUT.read_text())
    if old["dependencies"] != {p: sha(p) for p in old["dependencies"]}:
        raise ValueError("source dependency changed")
    paths = (Path(__file__), PROTOCOL, prior.OUTPUT, Path(prior.__file__),
             Path(prior.baseline.prior.exact.__file__), Path(wall.__file__), Path(wall.physics.__file__))
    dependencies = {str(p): sha(p) for p in paths}
    rows = []
    if len(old["rows"]) != 24:
        raise ValueError("unexpected task count")
    for row in old["rows"]:
        task, root = row["task"], row["root"]
        started = time.perf_counter()
        proposals = [make_chain(task["start"], task["goal"], a) for a in row["offsets"]]
        chains = np.stack([a for a, _, _ in proposals])
        points = np.stack([b for _, b, _ in proposals])
        required = np.asarray([n for _, _, n in proposals])
        costs = np.linalg.norm(prior.baseline.prior.exact.endpoints(task, chains) - task["goal"], axis=1)
        elapsed = time.perf_counter() - started
        best = int(np.argmin(costs))
        rows.append({"root": root, "task": task, "actions": chains.tolist(), "points": points.tolist(),
                     "required_steps": required.tolist(), "costs": costs.tolist(), "best": best,
                     "evaluations": len(chains), "seconds": elapsed})
        print(f"root {root}: two={costs[best]:.5f} one={row['costs'][row['best']]:.5f}", flush=True)
    if dependencies != {p: sha(p) for p in dependencies}:
        raise ValueError("dependency changed")
    with OUTPUT.open("x") as stream:
        json.dump({"dependencies": dependencies, "rows": rows,
                   "scope": "exposed exact-state development, no RGB or descriptor claim"},
                  stream, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    run()
