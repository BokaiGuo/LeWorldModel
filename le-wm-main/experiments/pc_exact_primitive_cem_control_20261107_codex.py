"""Fixed-budget turn/drive CEM control for exposed exact-wall development."""
import json
import time
from pathlib import Path

import numpy as np

from experiments import pc_exact_goal_relative_bridge_20261107_codex as bridge

ROOT = bridge.ROOT
PROTOCOL = ROOT.parent / "docs/PC_EXACT_PRIMITIVE_CEM_CONTROL_PROTOCOL_20261107_CODEX.md"
OUTPUT = ROOT / "results/pc_exact_primitive_cem_control_20261107_codex.json"
AUDIT = ROOT / "results/pc_exact_primitive_cem_control_audit_20261107_codex.json"
prior = bridge.baseline.prior


def chain_from_parameters(start, goal, params):
    start, goal, params = (np.asarray(x, dtype=float) for x in (start, goal, params))
    if (start.shape != (3,) or goal.shape != (2,) or params.shape != (4,)
            or not np.isfinite(np.r_[start, goal, params]).all()
            or np.any(np.abs(params) > 1)):
        raise ValueError("invalid inputs")
    delta = goal - start[:2]
    distance = float(np.linalg.norm(delta))
    if distance <= 1e-12:
        raise ValueError("distinct start and goal required")
    bearing = float(np.arctan2(delta[1], delta[0]))
    high = prior.exact.wall.HIGH
    heading = float(start[2])
    actions = []
    for angle_param, distance_param in ((params[0], params[1]), (params[2], params[3])):
        angle = bearing + float(angle_param) * np.pi / 2
        turn = (angle - heading + np.pi) % (2 * np.pi) - np.pi
        count = int(np.ceil(abs(turn) / high[1]))
        if count:
            actions.extend([[0., float(turn / count / high[1])]] * count)
        length = float((distance_param + 1) * distance)
        count = int(np.ceil(length / high[0]))
        if count:
            actions.extend([[float(length / count / high[0]), 0.]] * count)
        heading = angle
    required = len(actions)
    return np.asarray((actions + [[0., 0.]] * 15)[:15]), required


def sample_round(rng, mean=None, scale=None, count=128):
    if mean is None:
        return rng.uniform(-1., 1., (count, 4))
    return np.clip(mean + rng.normal(size=(count, 4)) * scale, -1., 1.)


def main():
    if OUTPUT.exists() or AUDIT.exists():
        raise FileExistsError("experiment output exists")
    old = json.loads(bridge.OUTPUT.read_text())
    if old["dependencies"] != {p: prior.sha(p) for p in old["dependencies"]}:
        raise ValueError("sealed bridge dependency changed")
    paths = (Path(__file__), PROTOCOL, bridge.OUTPUT, Path(bridge.__file__),
             Path(prior.exact.__file__), Path(prior.exact.wall.__file__),
             Path(prior.exact.wall.physics.__file__))
    dependencies = {str(p): prior.sha(p) for p in paths}
    rows = []
    for oldrow in old["rows"]:
        task, root = oldrow["task"], oldrow["root"]
        seed = 2717000 + root
        rng = np.random.default_rng(seed)
        params_all, actions_all, costs_all, steps_all, blocked_all = [], [], [], [], []
        mean = scale = None
        started = time.perf_counter()
        for iteration in range(5):
            params = sample_round(rng, mean, scale, 128 if iteration == 0 else 64)
            decoded = [chain_from_parameters(task["start"], task["goal"], p) for p in params]
            actions = np.stack([item[0] for item in decoded])
            required = [item[1] for item in decoded]
            padded = np.pad(actions, ((0, 0), (0, 1), (0, 0))) * prior.exact.wall.HIGH
            trajectory, blocked = prior.exact.wall.rollout(task["start"], padded)
            costs = np.linalg.norm(trajectory[:, 15, :2] - task["goal"], axis=1)
            elite = params[np.argsort(costs, kind="stable")[:16]]
            mean, scale = elite.mean(axis=0), np.maximum(elite.std(axis=0), .08)
            params_all.extend(params.tolist())
            actions_all.extend(actions.tolist())
            costs_all.extend(costs.tolist())
            steps_all.extend(required)
            blocked_all.extend(blocked[:, :15].tolist())
        elapsed = time.perf_counter() - started
        rows.append({"root": root, "task": task, "seed": seed, "parameters": params_all,
                     "actions": actions_all, "costs": costs_all, "required_steps": steps_all,
                     "blocked": blocked_all, "best": int(np.argmin(costs_all)),
                     "samples": len(costs_all), "seconds": elapsed})
        print(f"root {root}: primitive={min(costs_all):.5f}, bridge={min(oldrow['costs']):.5f}", flush=True)
    if dependencies != {p: prior.sha(p) for p in dependencies}:
        raise ValueError("dependency changed")
    with OUTPUT.open("x") as stream:
        json.dump({"dependencies": dependencies, "rows": rows,
                   "scope": "exposed exact-state development; not RGB or descriptor evidence"},
                  stream, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()
