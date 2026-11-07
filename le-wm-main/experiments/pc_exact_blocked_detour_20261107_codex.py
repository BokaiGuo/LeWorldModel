"""Fixed-budget predicted-collision detours in the exposed exact-wall panel."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_exact_prefix_successor_composition_20261107_codex as base

ROOT = base.ROOT
SOURCE = base.prior.OUTPUT
PROTOCOL = ROOT.parent / "docs/PC_EXACT_BLOCKED_DETOUR_PROTOCOL_20261107_CODEX.md"
OUTPUT = ROOT / "results/pc_exact_blocked_detour_20261107_codex.json"


def generate(start, goal, offsets):
    start, goal, offsets = (np.asarray(x, dtype=float) for x in (start, goal, offsets))
    if (start.shape != (3,) or goal.shape != (2,) or offsets.ndim != 1 or len(offsets) == 0
            or not np.isfinite(np.r_[start, goal, offsets]).all()
            or np.linalg.norm(goal - start[:2]) <= 1e-12):
        raise ValueError("invalid task or offsets")
    n = len(offsets)
    delta = goal - start[:2]
    lateral = np.array([-delta[1], delta[0]])
    waypoints = (start[:2] + goal)[None, :] / 2 + offsets[:, None] * lateral
    state = np.broadcast_to(start, (n, 3)).copy()
    destination = np.broadcast_to(goal, (n, 2))
    waypoint_done = np.zeros(n, dtype=bool)
    detouring = np.zeros(n, dtype=bool)
    detour_target = np.zeros((n, 2))
    detour_count = np.zeros(n, dtype=int)
    side = np.where(np.arange(n) % 2 == 0, 1., -1.)
    radius = np.where((np.arange(n) // 2) % 2 == 0, .3, .6)
    actions, traces, triggers = [], [state.copy()], []
    for _ in range(15):
        reached_detour = detouring & (np.linalg.norm(state[:, :2] - detour_target, axis=1) < 1e-9)
        detouring[reached_detour] = False
        waypoint_done |= (~detouring) & (np.linalg.norm(state[:, :2] - waypoints, axis=1) < 1e-9)
        target = np.where(detouring[:, None], detour_target,
                          np.where(waypoint_done[:, None], destination, waypoints))
        diff = target - state[:, :2]
        distance = np.linalg.norm(diff, axis=1)
        bearing = np.arctan2(diff[:, 1], diff[:, 0])
        turn = (bearing - state[:, 2] + np.pi) % (2 * np.pi) - np.pi
        action = np.zeros((n, 2))
        rotating = (distance > 1e-9) & (np.abs(turn) > 1e-9)
        action[rotating, 1] = np.clip(turn[rotating] / base.wall.HIGH[1], -1., 1.)
        moving = (distance > 1e-9) & ~rotating
        action[moving, 0] = np.minimum(distance[moving] / base.wall.HIGH[0], 1.)
        displacement = action[:, :1] * base.wall.HIGH[0] * np.stack(
            (np.cos(state[:, 2]), np.sin(state[:, 2])), axis=1)
        next_pos, _ = base.wall.physics.step_batch(state[:, :2], displacement)
        blocked = moving & (np.linalg.norm(next_pos - state[:, :2], axis=1) < 1e-9)
        # The detour target is generated from the predicted movement response only.
        normal = np.stack((-np.sin(state[:, 2]), np.cos(state[:, 2])), axis=1)
        detour_target[blocked] = next_pos[blocked] + side[blocked, None] * radius[blocked, None] * normal[blocked]
        detouring[blocked] = True
        detour_count += blocked
        state[:, :2] = next_pos
        state[:, 2] += action[:, 1] * base.wall.HIGH[1]
        actions.append(action)
        traces.append(state.copy())
        triggers.append(blocked.copy())
    return (np.stack(actions, axis=1), np.stack(traces, axis=1),
            np.stack(triggers, axis=1), detour_count)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source = json.loads(SOURCE.read_text())
    if source["dependencies"] != {p: base.sha(p) for p in source["dependencies"]}:
        raise ValueError("source drift")
    deps = {str(p): base.sha(p) for p in (Path(__file__), PROTOCOL, SOURCE, Path(base.__file__),
                                           Path(base.wall.__file__), Path(base.wall.physics.__file__))}
    rows = []
    print("Testing fixed-budget model-predicted collision detours versus sealed waypoint chains.", flush=True)
    for original in source["rows"]:
        task, root = original["task"], original["root"]
        actions, trace, triggers, counts = generate(task["start"], task["goal"], original["offsets"])
        if actions.shape != (384, 15, 2) or np.abs(actions).max() > 1 + 1e-12:
            raise ValueError("bad action pool")
        # Validation replay is not included in the deployment budget.
        replay, _ = base.rollout_n(task["start"], actions)
        np.testing.assert_allclose(trace, replay, atol=1e-12, rtol=0)
        costs = np.linalg.norm(trace[:, -1, :2] - task["goal"], axis=1)
        chosen = int(np.argmin(costs))
        old_best = float(original["costs"][original["best"]])
        rows.append({"root": root, "task": task, "actions": actions.tolist(),
                     "costs": costs.tolist(), "chosen": chosen, "trigger_counts": counts.tolist(),
                     "trigger_steps": triggers.tolist(), "old_best": old_best,
                     "new_best": float(costs[chosen]), "model_steps": 384 * 15})
        print(f"root={root} old={old_best:.5f} new={costs[chosen]:.5f}", flush=True)
    if deps != {p: base.sha(p) for p in deps} or len(rows) != 24:
        raise ValueError("dependency or panel drift")
    with OUTPUT.open("x") as f:
        json.dump({"dependencies": deps, "rows": rows,
                   "scope": "exposed exact-state development; not RGB or descriptor evidence"},
                  f, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()
