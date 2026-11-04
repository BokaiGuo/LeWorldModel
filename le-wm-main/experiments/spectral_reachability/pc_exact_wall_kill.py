"""Fixed analytical detour pilot; no learned model or physical environment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

START = np.array([-0.5, 0.0])
GOAL = np.array([0.5, 0.0])
RADIUS = 0.3
HORIZON = 12
ANGLES = np.arange(16) * 2 * np.pi / 16
ACTIONS = np.vstack((RADIUS * np.column_stack((np.cos(ANGLES), np.sin(ANGLES))), [[0.0, 0.0]]))


def step_batch(states, actions):
    states, actions = np.asarray(states, dtype=float), np.asarray(actions, dtype=float)
    if states.shape != actions.shape or states.shape[-1] != 2:
        raise ValueError("states and actions must have matching [...,2] shapes")
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError("nonfinite state or action")
    if np.any(np.linalg.norm(actions, axis=-1) > RADIUS + 1e-12):
        raise ValueError("action outside disk")
    end = states + actions
    dx = actions[..., 0]
    # Include endpoints to prevent penetrating the wall via x==0.
    crosses = ((states[..., 0] <= 0) & (end[..., 0] >= 0) |
               (states[..., 0] >= 0) & (end[..., 0] <= 0)) & (np.abs(dx) > 1e-15)
    fraction = np.divide(-states[..., 0], dx, out=np.zeros_like(dx), where=np.abs(dx) > 1e-15)
    intersection_y = states[..., 1] + fraction * actions[..., 1]
    blocked = crosses & (np.abs(intersection_y) < 1.0)
    return np.where(blocked[..., None], states, end), blocked


def project(actions):
    norm = np.linalg.norm(actions, axis=-1, keepdims=True)
    return actions * np.minimum(1.0, RADIUS / np.maximum(norm, 1e-30))


def rollout_batch(programs):
    programs = np.asarray(programs, dtype=float)
    if programs.ndim != 3 or programs.shape[1:] != (HORIZON, 2):
        raise ValueError("program must be [n,12,2]")
    state = np.repeat(START[None], len(programs), axis=0)
    rejected = np.zeros(len(programs), dtype=int)
    for t in range(HORIZON):
        state, blocked = step_batch(state, programs[:, t])
        rejected += blocked
    return np.linalg.norm(state - GOAL, axis=1), rejected


def greedy():
    state = START.copy()
    selected = []
    for _ in range(HORIZON):
        next_states, _ = step_batch(np.repeat(state[None], len(ACTIONS), axis=0), ACTIONS)
        idx = int(np.argmin(np.linalg.norm(next_states - GOAL, axis=1)))
        selected.append(ACTIONS[idx])
        state = next_states[idx]
    return np.asarray(selected), {"transitions": HORIZON * len(ACTIONS), "scores": HORIZON * len(ACTIONS)}


def cem():
    rng = np.random.default_rng(20261102)
    mean, std = np.zeros((HORIZON, 2)), np.full((HORIZON, 2), 0.3)
    best_cost, best = float("inf"), None
    for _ in range(8):
        proposals = project(rng.normal(mean, std, (256, HORIZON, 2)))
        costs, _ = rollout_batch(proposals)
        index = int(np.argmin(costs))
        if costs[index] < best_cost:
            best_cost, best = float(costs[index]), proposals[index].copy()
        elites = proposals[np.argsort(costs, kind="stable")[:32]]
        mean, std = elites.mean(axis=0), np.maximum(elites.std(axis=0), 1e-3)
    return best, {"transitions": 256 * 8 * HORIZON, "scores": 256 * 8}


def beam():
    states = START[None]
    sequences = np.empty((1, 0, 2))
    transitions = 0
    for _ in range(HORIZON):
        n = len(states)
        current = np.repeat(states, len(ACTIONS), axis=0)
        actions = np.tile(ACTIONS, (n, 1))
        after, _ = step_batch(current, actions)
        transitions += len(after)
        paths = np.concatenate((np.repeat(sequences, len(ACTIONS), axis=0), actions[:, None]), axis=1)
        indices = np.argsort(np.linalg.norm(after - GOAL, axis=1), kind="stable")[:20]
        states, sequences = after[indices], paths[indices]
    return sequences[0], {"transitions": transitions, "scores": transitions}


def witness():
    state = START.copy()
    actions = []
    for target in (np.array([0.0, 1.1]), GOAL):
        displacement = target - state
        count = int(np.ceil(np.linalg.norm(displacement) / RADIUS))
        action = displacement / count
        for _ in range(count):
            actions.append(action.copy())
            state, blocked = step_batch(state, action)
            if bool(blocked):
                raise RuntimeError("privileged witness hit the wall")
    if len(actions) > HORIZON:
        raise RuntimeError("witness exceeds planning horizon")
    actions += [np.zeros(2)] * (HORIZON - len(actions))
    return np.array(actions), {"transitions": len(actions), "scores": 0}


def run():
    results = {}
    for name, method in (("greedy", greedy), ("cem", cem), ("beam", beam), ("witness", witness)):
        program, ledger = method()
        state = START.copy()
        trajectory = [state.tolist()]
        blocked_steps = []
        for i, action in enumerate(program):
            state, blocked = step_batch(state, action)
            trajectory.append(state.tolist())
            if blocked:
                blocked_steps.append(i)
        errors = np.linalg.norm(np.asarray(trajectory) - GOAL, axis=1)
        results[name] = {"actions": program.tolist(), "trajectory": trajectory,
                         "terminal_error": float(errors[-1]), "maximum_goal_distance": float(errors.max()),
                         "blocked_steps": blocked_steps, "ledger": ledger}
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    results = run()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"scope": "analytical one-task development; not D2P or physical outcome",
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "results": results}
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: {"terminal_error": v["terminal_error"],
                          "maximum_goal_distance": v["maximum_goal_distance"],
                          "ledger": v["ledger"]} for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
