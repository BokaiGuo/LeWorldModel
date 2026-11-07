"""Post-hoc trigger assignment intervention; not a deployable blind planner."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_exact_blocked_detour_20261107_codex as trial

ROOT = trial.ROOT
PROTOCOL = ROOT.parent / "docs/PC_EXACT_DETOUR_SHUFFLED_TRIGGER_PROTOCOL_20261107_CODEX.md"
OUTPUT = ROOT / "results/pc_exact_detour_shuffled_trigger_20261107_codex.json"


def permutation(n, root):
    if n < 4 or n % 4 or root < 0:
        raise ValueError("invalid candidate count or root")
    rng = np.random.default_rng(20261107 + root)
    order = np.arange(n)
    for group in range(4):
        group_ids = np.arange(group, n, 4)
        order[group_ids] = rng.permutation(group_ids)
    return order


def generate(start, goal, offsets, schedule):
    start, goal, offsets = (np.asarray(x, dtype=float) for x in (start, goal, offsets))
    schedule = np.asarray(schedule)
    n = len(offsets)
    if (start.shape != (3,) or goal.shape != (2,) or offsets.ndim != 1
            or n == 0 or schedule.shape != (n, 15) or schedule.dtype != bool
            or not np.isfinite(np.r_[start, goal, offsets]).all()
            or np.linalg.norm(goal - start[:2]) <= 1e-12):
        raise ValueError("invalid task or schedule")
    delta = goal - start[:2]
    waypoints = (start[:2] + goal)[None, :] / 2 + offsets[:, None] * np.array([-delta[1], delta[0]])
    state = np.broadcast_to(start, (n, 3)).copy()
    done = np.zeros(n, dtype=bool)
    detouring = np.zeros(n, dtype=bool)
    detour_target = np.zeros((n, 2))
    side = np.where(np.arange(n) % 2 == 0, 1., -1.)
    radius = np.where((np.arange(n) // 2) % 2 == 0, .3, .6)
    actions, traces = [], [state.copy()]
    for step in range(15):
        detouring &= np.linalg.norm(state[:, :2] - detour_target, axis=1) >= 1e-9
        done |= ~detouring & (np.linalg.norm(state[:, :2] - waypoints, axis=1) < 1e-9)
        target = np.where(detouring[:, None], detour_target, np.where(done[:, None], goal, waypoints))
        diff = target - state[:, :2]
        distance = np.linalg.norm(diff, axis=1)
        turn = (np.arctan2(diff[:, 1], diff[:, 0]) - state[:, 2] + np.pi) % (2 * np.pi) - np.pi
        action = np.zeros((n, 2))
        rotating = (distance > 1e-9) & (np.abs(turn) > 1e-9)
        action[rotating, 1] = np.clip(turn[rotating] / trial.base.wall.HIGH[1], -1., 1.)
        moving = (distance > 1e-9) & ~rotating
        action[moving, 0] = np.minimum(distance[moving] / trial.base.wall.HIGH[0], 1.)
        normal = np.stack((-np.sin(state[:, 2]), np.cos(state[:, 2])), axis=1)
        displacement = action[:, :1] * trial.base.wall.HIGH[0] * np.stack(
            (np.cos(state[:, 2]), np.sin(state[:, 2])), axis=1)
        next_pos, _ = trial.base.wall.physics.step_batch(state[:, :2], displacement)
        mask = schedule[:, step]
        detour_target[mask] = next_pos[mask] + side[mask, None] * radius[mask, None] * normal[mask]
        detouring[mask] = True
        state[:, :2] = next_pos
        state[:, 2] += action[:, 1] * trial.base.wall.HIGH[1]
        actions.append(action)
        traces.append(state.copy())
    return np.stack(actions, axis=1), np.stack(traces, axis=1)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source = json.loads(trial.SOURCE.read_text())
    observed = json.loads(trial.OUTPUT.read_text())
    for obj in (source, observed):
        if obj["dependencies"] != {p: trial.base.sha(p) for p in obj["dependencies"]}:
            raise ValueError("dependency drift")
    deps = {str(p): trial.base.sha(p) for p in (
        Path(__file__), PROTOCOL, trial.SOURCE, trial.OUTPUT, Path(trial.__file__),
        Path(trial.base.__file__), Path(trial.base.wall.__file__), Path(trial.base.wall.physics.__file__))}
    rows = []
    print("Testing candidate-specific blocked trigger timing against within-class reassignment.", flush=True)
    for old, original in zip(observed["rows"], source["rows"], strict=True):
        if old["root"] != original["root"] or old["task"] != original["task"]:
            raise ValueError("task mismatch")
        task = original["task"]
        schedule = np.asarray(old["trigger_steps"], dtype=bool)
        actions, _ = generate(task["start"], task["goal"], original["offsets"], schedule)
        np.testing.assert_allclose(actions, old["actions"], atol=1e-12, rtol=0)
        order = permutation(len(schedule), original["root"])
        shuffled = schedule[order]
        actions, trace = generate(task["start"], task["goal"], original["offsets"], shuffled)
        if actions.shape != (384, 15, 2) or np.max(np.abs(actions)) > 1 + 1e-12:
            raise ValueError("invalid action pool")
        for group in range(4):
            np.testing.assert_array_equal(np.sort(order[group::4]), np.arange(group, 384, 4))
            np.testing.assert_array_equal(shuffled[group::4].sum(axis=0), schedule[group::4].sum(axis=0))
        replay, _ = trial.base.rollout_n(task["start"], actions)
        np.testing.assert_allclose(trace, replay, atol=1e-12, rtol=0)
        costs = np.linalg.norm(trace[:, -1, :2] - task["goal"], axis=1)
        chosen = int(np.argmin(costs))
        rows.append({"root": original["root"], "permutation": order.tolist(),
                     "actions": actions.tolist(), "costs": costs.tolist(), "chosen": chosen,
                     "shuffled_cost": float(costs[chosen]), "blocked_cost": old["new_best"],
                     "model_steps": 384 * 15, "identity_reconstruction_steps": 384 * 15,
                     "verification_replay_steps": 384 * 15})
        print(f"root={original['root']} shuffled={costs[chosen]:.5f} blocked={old['new_best']:.5f}", flush=True)
    if len(rows) != 24 or deps != {p: trial.base.sha(p) for p in deps}:
        raise ValueError("panel or dependency drift")
    with OUTPUT.open("x") as f:
        json.dump({"dependencies": deps, "rows": rows,
                   "scope": "exposed exact-state diagnostic; schedule is not free or deployable"},
                  f, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()
