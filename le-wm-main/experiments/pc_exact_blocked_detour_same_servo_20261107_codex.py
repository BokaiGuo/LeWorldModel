"""Post-hoc same-controller, no-detour control for the sealed detour run."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_exact_blocked_detour_20261107_codex as trial

ROOT = trial.ROOT
PROTOCOL = ROOT.parent / "docs/PC_EXACT_BLOCKED_DETOUR_SAME_SERVO_PROTOCOL_20261107_CODEX.md"
OUTPUT = ROOT / "results/pc_exact_blocked_detour_same_servo_20261107_codex.json"


def no_detour(start, goal, offsets):
    start, goal, offsets = (np.asarray(x, float) for x in (start, goal, offsets))
    if (start.shape != (3,) or goal.shape != (2,) or offsets.ndim != 1 or not len(offsets)
            or not np.isfinite(np.r_[start, goal, offsets]).all()
            or np.linalg.norm(goal - start[:2]) <= 1e-12):
        raise ValueError("invalid task")
    n = len(offsets)
    diff = goal - start[:2]
    waypoints = (start[:2] + goal)[None, :] / 2 + offsets[:, None] * np.array([-diff[1], diff[0]])
    state = np.broadcast_to(start, (n, 3)).copy()
    finished = np.zeros(n, dtype=bool)
    actions, trajectory = [], [state.copy()]
    for _ in range(15):
        finished |= np.linalg.norm(state[:, :2] - waypoints, axis=1) < 1e-9
        delta = np.where(finished[:, None], goal, waypoints) - state[:, :2]
        distance = np.linalg.norm(delta, axis=1)
        turn = (np.arctan2(delta[:, 1], delta[:, 0]) - state[:, 2] + np.pi) % (2*np.pi) - np.pi
        action = np.zeros((n, 2))
        rotating = (distance > 1e-9) & (np.abs(turn) > 1e-9)
        action[rotating, 1] = np.clip(turn[rotating] / trial.base.wall.HIGH[1], -1., 1.)
        moving = (distance > 1e-9) & ~rotating
        action[moving, 0] = np.minimum(distance[moving] / trial.base.wall.HIGH[0], 1.)
        displacement = action[:, :1] * trial.base.wall.HIGH[0] * np.stack(
            (np.cos(state[:, 2]), np.sin(state[:, 2])), axis=1)
        state[:, :2], _ = trial.base.wall.physics.step_batch(state[:, :2], displacement)
        state[:, 2] += action[:, 1] * trial.base.wall.HIGH[1]
        actions.append(action)
        trajectory.append(state.copy())
    return np.stack(actions, axis=1), np.stack(trajectory, axis=1)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    detour, original = json.loads(trial.OUTPUT.read_text()), json.loads(trial.SOURCE.read_text())
    if detour["dependencies"] != {p: trial.base.sha(p) for p in detour["dependencies"]}:
        raise ValueError("detour dependency drift")
    dependencies = {str(p): trial.base.sha(p) for p in (Path(__file__), PROTOCOL, trial.OUTPUT, trial.SOURCE)}
    rows = []
    print("Post-hoc same-servo control, detour enabled versus disabled.", flush=True)
    for saved, source in zip(detour["rows"], original["rows"], strict=True):
        if saved["root"] != source["root"] or saved["task"] != source["task"]:
            raise ValueError("task mismatch")
        actions, trace = no_detour(source["task"]["start"], source["task"]["goal"], source["offsets"])
        detour_actions = np.asarray(saved["actions"])
        triggers = np.asarray(saved["trigger_steps"], dtype=bool)
        for i in range(len(actions)):
            where = np.flatnonzero(triggers[i])
            through = int(where[0] + 1) if len(where) else 15
            np.testing.assert_allclose(actions[i, :through], detour_actions[i, :through], atol=1e-12, rtol=0)
        replay, _ = trial.base.rollout_n(source["task"]["start"], actions)
        np.testing.assert_allclose(trace, replay, atol=1e-12, rtol=0)
        costs = np.linalg.norm(trace[:, -1, :2] - source["task"]["goal"], axis=1)
        idx = int(np.argmin(costs))
        rows.append({"root": saved["root"], "actions": actions.tolist(), "costs": costs.tolist(),
                     "chosen": idx, "control_cost": float(costs[idx]), "detour_cost": saved["new_best"],
                     "model_steps": 384 * 15})
        print(f"root={saved['root']} no_detour={costs[idx]:.5f} detour={saved['new_best']:.5f}", flush=True)
    if len(rows) != 24 or dependencies != {p: trial.base.sha(p) for p in dependencies}:
        raise ValueError("panel or dependency drift")
    with OUTPUT.open("x") as f:
        json.dump({"dependencies": dependencies, "rows": rows,
                   "scope": "post-hoc same-servo control on exposed exact-state tasks"},
                  f, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()
