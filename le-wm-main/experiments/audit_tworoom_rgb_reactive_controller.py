"""Pure RGB reactive TwoRoom control ceiling.

This controller uses only the rendered current frame and rendered goal frame.
It repeatedly re-observes after one action pulse and steers toward a visible
door waypoint when the two red markers are on opposite sides of the wall.
It is deliberately not a world-model method; it separates the action/control
interface from frozen-LeWM action ranking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from experiments.run_live_tworoom_planning import TwoRoomGoalAliasWrapper


def _red_centroid(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    mask = (
        (image[..., 0] > 160)
        & (image[..., 0] > image[..., 1] * 1.5)
        & (image[..., 0] > image[..., 2] * 1.5)
    )
    yy, xx = np.nonzero(mask)
    if len(xx) == 0:
        return np.asarray([image.shape[1] / 2.0, image.shape[0] / 2.0], dtype=np.float64)
    return np.asarray([xx.mean(), yy.mean()], dtype=np.float64)


def _longest_true_run(row: np.ndarray, valid: np.ndarray | None = None) -> tuple[int, int]:
    if valid is not None:
        row = row & valid
    best = (0, 0)
    start: int | None = None
    for index, value in enumerate(np.concatenate([row, [False]])):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start > best[1] - best[0]:
                best = (start, index)
            start = None
    return best


def _door_point(image: np.ndarray) -> tuple[np.ndarray, bool]:
    image = np.asarray(image)
    dark = image.mean(axis=-1) < 50.0
    height, width = dark.shape
    cx, cy = width // 2, height // 2
    vertical_line = dark[:, max(0, cx - 3):min(width, cx + 4)].mean(axis=1)
    horizontal_line = dark[max(0, cy - 3):min(height, cy + 4), :].mean(axis=0)
    vertical = bool(vertical_line.mean() >= horizontal_line.mean())
    vertical_dark = dark[:, max(0, cx - 3):min(width, cx + 4)].mean(axis=1) > 0.5
    horizontal_dark = dark[max(0, cy - 3):min(height, cy + 4), :].mean(axis=0) > 0.5
    vertical_open = ~vertical_dark
    horizontal_open = ~horizontal_dark
    v_valid = np.zeros_like(vertical_dark)
    h_valid = np.zeros_like(horizontal_dark)
    v_indices = np.flatnonzero(vertical_dark)
    h_indices = np.flatnonzero(horizontal_dark)
    if len(v_indices):
        v_valid[v_indices.min():v_indices.max() + 1] = True
    if len(h_indices):
        h_valid[h_indices.min():h_indices.max() + 1] = True
    vs, ve = _longest_true_run(vertical_open, v_valid)
    hs, he = _longest_true_run(horizontal_open, h_valid)
    if vertical:
        return np.asarray([cx, (vs + ve) / 2.0], dtype=np.float64), True
    return np.asarray([(hs + he) / 2.0, cy], dtype=np.float64), False


def _action(current: np.ndarray, goal: np.ndarray) -> tuple[np.ndarray, str]:
    source = _red_centroid(current)
    target = _red_centroid(goal)
    door, vertical = _door_point(current)
    side_gap = abs(float(target[0] - source[0])) if vertical else abs(float(target[1] - source[1]))
    waypoint = target
    phase = "goal"
    if side_gap > 35.0 and np.linalg.norm(source - door) > 18.0:
        waypoint = door
        phase = "door"
    delta = waypoint - source
    norm = float(np.linalg.norm(delta))
    if norm < 1.0e-6:
        return np.zeros(2, dtype=np.float32), phase
    return (delta / norm).astype(np.float32), phase


def run(output: Path, seed: int, roots: int, max_steps: int, pulse_steps: int) -> dict:
    env = TwoRoomGoalAliasWrapper(gym.make("swm/TwoRoom-v1", render_mode="rgb_array"))
    rows = []
    try:
        for index in range(int(roots)):
            root_seed = int(seed + index)
            _, info = env.reset(seed=root_seed)
            goal = np.asarray(info["goal"], dtype=np.uint8).copy()
            phases = []
            terminated = False
            for _ in range(int(max_steps)):
                action, phase = _action(np.asarray(env.render()), goal)
                phases.append(phase)
                for _ in range(max(1, int(pulse_steps))):
                    _, _, terminated, truncated, _ = env.step(action)
                    if terminated or truncated:
                        break
                if terminated:
                    break
            rows.append({
                "seed": root_seed,
                "success": int(terminated),
                "steps": len(phases),
                "door_pulses": int(sum(item == "door" for item in phases)),
                "goal_pulses": int(sum(item == "goal" for item in phases)),
            })
            print(
                f"[tworoom-rgb-reactive] root={index + 1}/{roots} seed={root_seed} "
                f"success={int(terminated)} steps={len(phases)}", flush=True,
            )
    finally:
        env.close()
    report = {
        "schema": "tworoom_rgb_reactive_controller.v0.1",
        "protocol": {
            "roots": int(roots),
            "seed_start": int(seed),
            "max_steps": int(max_steps),
            "pulse_steps": int(pulse_steps),
            "runtime_input": "current RGB frame and fixed rendered RGB goal frame",
            "simulator_state_access": False,
            "learned_model_used": False,
        },
        "metrics": {
            "root_count": len(rows),
            "success_count": int(sum(row["success"] for row in rows)),
            "success_rate": float(np.mean([row["success"] for row in rows])),
        },
        "rows": rows,
        "claim_boundary": "RGB reactive control ceiling only; not learned world-model or universal planning evidence",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20268000)
    parser.add_argument("--roots", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--pulse-steps", type=int, default=1)
    args = parser.parse_args()
    report = run(args.output, args.seed, args.roots, args.max_steps, args.pulse_steps)
    print(json.dumps(report["metrics"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
