"""Budget-matched successor-conditioned composition on exposed exact-wall roots."""
import json
import time
from pathlib import Path

import numpy as np

from experiments import pc_exact_goal_relative_bridge_20261107_codex as prior

ROOT = prior.ROOT
PROTOCOL = ROOT.parent / "docs/PC_EXACT_PREFIX_SUCCESSOR_COMPOSITION_PROTOCOL_20261107_CODEX.md"
OUTPUT = ROOT / "results/pc_exact_prefix_successor_composition_20261107_codex.json"
wall = prior.baseline.prior.exact.wall
sha = prior.baseline.prior.sha


def rollout_n(start, chains):
    start, chains = np.asarray(start, dtype=float), np.asarray(chains, dtype=float)
    if (start.shape != (3,) or chains.ndim != 3 or chains.shape[2] != 2
            or chains.shape[1] < 1 or not np.isfinite(start).all() or not np.isfinite(chains).all()
            or np.any(np.abs(chains) > 1 + 1e-12)):
        raise ValueError("finite bounded actions and start pose required")
    state = np.broadcast_to(start, (len(chains), 3)).copy()
    states, blocked = [state.copy()], []
    for t in range(chains.shape[1]):
        action = chains[:, t] * wall.HIGH
        displacement = action[:, :1] * np.stack((np.cos(state[:, 2]), np.sin(state[:, 2])), axis=1)
        state[:, :2], event = wall.physics.step_batch(state[:, :2], displacement)
        state[:, 2] += action[:, 1]
        states.append(state.copy())
        blocked.append(event)
    return np.stack(states, axis=1), np.stack(blocked, axis=1)


def route(start, points):
    position, heading = start[:2].copy(), float(start[2])
    actions, segment_counts = [], []
    for point in points:
        displacement = point - position
        distance = float(np.linalg.norm(displacement))
        begin = len(actions)
        if distance > 1e-12:
            angle = float(np.arctan2(displacement[1], displacement[0]))
            turn = float((angle - heading + np.pi) % (2 * np.pi) - np.pi)
            turns = int(np.ceil(abs(turn) / wall.HIGH[1]))
            if turns:
                actions.extend([[0., turn / turns / wall.HIGH[1]]] * turns)
            steps = int(np.ceil(distance / wall.HIGH[0]))
            actions.extend([[distance / steps / wall.HIGH[0], 0.]] * steps)
            position, heading = point.copy(), angle
        segment_counts.append(len(actions) - begin)
    return actions, segment_counts


def suffix_from_successor(successor, waypoint, goal, waypoint_finished):
    state, waypoint, goal = (np.asarray(x, dtype=float) for x in (successor, waypoint, goal))
    if (state.shape != (3,) or waypoint.shape != (2,) or goal.shape != (2,)
            or not np.isfinite(np.r_[state, waypoint, goal]).all()
            or not isinstance(waypoint_finished, (bool, np.bool_))):
        raise ValueError("finite successor, waypoint, goal and stage flag required")
    actions, counts = route(state, [goal] if waypoint_finished else [waypoint, goal])
    return np.asarray((actions + [[0., 0.]] * 10)[:10]), counts


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    sealed = json.loads(prior.OUTPUT.read_text())
    if sealed["dependencies"] != {p: sha(p) for p in sealed["dependencies"]}:
        raise ValueError("prior dependency drift")
    paths = (Path(__file__), PROTOCOL, prior.OUTPUT, Path(prior.__file__),
             Path(wall.__file__), Path(wall.physics.__file__))
    dependencies = {str(p): sha(p) for p in paths}
    if len(sealed["rows"]) != 24:
        raise ValueError("unexpected panel")
    print("Exact-state development: compare budget-matched predicted-successor composition.", flush=True)
    rows = []
    for row in sealed["rows"]:
        root, task = row["root"], row["task"]
        start, goal = np.asarray(task["start"], float), np.asarray(task["goal"], float)
        t0 = time.perf_counter()
        originals = [prior.make_chain(start, goal, offset) for offset in row["offsets"][:256]]
        base_actions = np.stack([item[0] for item in originals])
        base_trace, base_blocked = rollout_n(start, base_actions)
        base_cost = np.linalg.norm(base_trace[:, -1, :2] - goal, axis=1)
        np.testing.assert_allclose(base_cost, np.asarray(row["costs"][:256]), atol=1e-12, rtol=0)
        base_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        prefixes = base_actions[:192, :5]
        prefix_trace, prefix_blocked = rollout_n(start, prefixes)
        successor = prefix_trace[:, -1]
        tails, stage_counts = [], []
        for state, (_, waypoint, _old_required) in zip(successor, originals[:192], strict=True):
            _, counts = route(start, [waypoint])
            tail, _ = suffix_from_successor(state, waypoint, goal, counts[0] <= 5)
            tails.append(tail)
            stage_counts.append(counts[0])
        chains = np.concatenate((prefixes, np.stack(tails)), axis=1)
        trace, blocked = rollout_n(start, chains)
        np.testing.assert_allclose(trace[:, 5], successor, atol=1e-12, rtol=0)
        costs = np.linalg.norm(trace[:, -1, :2] - goal, axis=1)
        seconds = time.perf_counter() - t0
        np.testing.assert_allclose(trace[:, -1, :2], prior.baseline.prior.exact.endpoints(task, chains),
                                   atol=1e-12, rtol=0)
        b, c = int(np.argmin(base_cost)), int(np.argmin(costs))
        rows.append({"root": root, "task": task, "base_index": b, "base_cost": float(base_cost[b]),
                     "base_action": base_actions[b].tolist(), "base_blocked_steps": int(base_blocked[b].sum()),
                     "base_seconds": base_seconds, "prefix_actions": prefixes.tolist(),
                     "prefix_successors": successor.tolist(), "prefix_blocked": prefix_blocked.tolist(),
                     "waypoint_stage_steps": stage_counts, "actions": chains.tolist(), "costs": costs.tolist(),
                     "chosen": c, "blocked_steps": int(blocked[c].sum()), "seconds": seconds,
                     "baseline_model_steps": 256 * 15, "composed_model_steps": 192 * (5 + 15)})
        print(f"root={root} baseline={base_cost[b]:.5f} composition={costs[c]:.5f}", flush=True)
    if dependencies != {p: sha(p) for p in dependencies}:
        raise ValueError("dependency changed")
    with OUTPUT.open("x") as stream:
        json.dump({"dependencies": dependencies, "rows": rows,
                   "scope": "exposed exact-state development; not RGB or descriptor evidence"},
                  stream, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()
