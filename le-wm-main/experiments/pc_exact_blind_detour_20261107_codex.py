"""One fixed-time blind detour versus collision-triggered detours."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_exact_blocked_detour_20261107_codex as trial
from experiments import pc_exact_blocked_detour_same_servo_20261107_codex as control

ROOT = trial.ROOT
PROTOCOL = ROOT.parent / "docs/PC_EXACT_BLIND_DETOUR_PROTOCOL_20261107_CODEX.md"
OUTPUT = ROOT / "results/pc_exact_blind_detour_20261107_codex.json"


def generate(start, goal, offsets):
    start, goal, offsets = (np.asarray(x, dtype=float) for x in (start, goal, offsets))
    if (start.shape != (3,) or goal.shape != (2,) or offsets.ndim != 1
            or not len(offsets) or not np.isfinite(np.r_[start, goal, offsets]).all()
            or np.linalg.norm(goal - start[:2]) <= 1e-12):
        raise ValueError("invalid task")
    n = len(offsets)
    delta = goal - start[:2]
    waypoints = (start[:2] + goal)[None, :] / 2 + offsets[:, None] * np.array([-delta[1], delta[0]])
    state = np.broadcast_to(start, (n, 3)).copy()
    done = np.zeros(n, dtype=bool)
    detouring = np.zeros(n, dtype=bool)
    target_detour = np.zeros((n, 2))
    side = np.where(np.arange(n) % 2 == 0, 1., -1.)
    radius = np.where((np.arange(n) // 2) % 2 == 0, .3, .6)
    actions, traces = [], [state.copy()]
    for step in range(15):
        detouring &= np.linalg.norm(state[:, :2] - target_detour, axis=1) >= 1e-9
        done |= ~detouring & (np.linalg.norm(state[:, :2] - waypoints, axis=1) < 1e-9)
        target = np.where(detouring[:, None], target_detour, np.where(done[:, None], goal, waypoints))
        diff = target - state[:, :2]
        distance = np.linalg.norm(diff, axis=1)
        turn = (np.arctan2(diff[:, 1], diff[:, 0]) - state[:, 2] + np.pi) % (2 * np.pi) - np.pi
        action = np.zeros((n, 2))
        rotating = (distance > 1e-9) & (np.abs(turn) > 1e-9)
        action[rotating, 1] = np.clip(turn[rotating] / trial.base.wall.HIGH[1], -1., 1.)
        moving = (distance > 1e-9) & ~rotating
        action[moving, 0] = np.minimum(distance[moving] / trial.base.wall.HIGH[0], 1.)
        displacement = action[:, :1] * trial.base.wall.HIGH[0] * np.stack(
            (np.cos(state[:, 2]), np.sin(state[:, 2])), axis=1)
        state[:, :2], _ = trial.base.wall.physics.step_batch(state[:, :2], displacement)
        state[:, 2] += action[:, 1] * trial.base.wall.HIGH[1]
        if step == 4:
            normal = np.stack((-np.sin(state[:, 2]), np.cos(state[:, 2])), axis=1)
            target_detour = state[:, :2] + side[:, None] * radius[:, None] * normal
            detouring[:] = True
        actions.append(action)
        traces.append(state.copy())
    return np.stack(actions, axis=1), np.stack(traces, axis=1)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    old, source, baseline = (json.loads(p.read_text()) for p in (trial.OUTPUT, trial.SOURCE, control.OUTPUT))
    for artifact in (old, baseline):
        if artifact["dependencies"] != {p: trial.base.sha(p) for p in artifact["dependencies"]}:
            raise ValueError("dependency drift")
    deps = {str(p): trial.base.sha(p) for p in (
        Path(__file__), PROTOCOL, trial.OUTPUT, trial.SOURCE, control.OUTPUT,
        Path(trial.base.__file__), Path(trial.base.wall.__file__), Path(trial.base.wall.physics.__file__))}
    rows = []
    print("Testing fixed step-5 blind detours against blocked-triggered detours.", flush=True)
    for blocked, original, plain in zip(old["rows"], source["rows"], baseline["rows"], strict=True):
        if blocked["root"] != original["root"] or plain["root"] != original["root"]:
            raise ValueError("root mismatch")
        actions, trace = generate(original["task"]["start"], original["task"]["goal"], original["offsets"])
        if actions.shape != (384, 15, 2) or np.abs(actions).max() > 1 + 1e-12:
            raise ValueError("invalid actions")
        np.testing.assert_allclose(actions[:, :5], np.asarray(plain["actions"])[:, :5], atol=1e-12, rtol=0)
        replay, _ = trial.base.rollout_n(original["task"]["start"], actions)
        np.testing.assert_allclose(trace, replay, atol=1e-12, rtol=0)
        costs = np.linalg.norm(trace[:, -1, :2] - original["task"]["goal"], axis=1)
        chosen = int(np.argmin(costs))
        rows.append({"root": original["root"], "actions": actions.tolist(), "costs": costs.tolist(),
                     "chosen": chosen, "blind_cost": float(costs[chosen]),
                     "blocked_cost": blocked["new_best"], "plain_cost": plain["control_cost"],
                     "model_steps": 384 * 15, "verification_replay_steps": 384 * 15})
        print(f"root={original['root']} blind={costs[chosen]:.5f} blocked={blocked['new_best']:.5f}", flush=True)
    if len(rows) != 24 or deps != {p: trial.base.sha(p) for p in deps}:
        raise ValueError("panel or dependency drift")
    with OUTPUT.open("x") as f:
        json.dump({"dependencies": deps, "rows": rows, "scope": "exposed exact-state development"},
                  f, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()
