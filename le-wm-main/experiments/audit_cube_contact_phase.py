"""Isolate the Cube contact-phase bottleneck with RGB-only control.

This is a mechanism audit, not planning evidence.  The controller uses only
RGB centroids for XY approach and a predeclared action schedule.  Simulator
contact, effector/block height, and XY distances are evaluator-only labels.
The paired comparison tests whether the historical ``descend -> XY approach
-> close`` ordering is responsible for the missing contact, versus
``descend -> XY approach -> descend -> close``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401

from experiments.audit_cube_rgb_observability import (
    _goal_position,
    _target_red_centroid,
    _largest_effector_centroid,
)
from experiments.run_live_cube_rgb_macro_ranker import _rgb_approach_action


def _make_env() -> gym.Env:
    return gym.make(
        "swm/OGBCube-v0", ob_type="pixels", render_mode="rgb_array",
        terminate_at_goal=False,
    )


def _label(info: dict[str, Any], env: gym.Env) -> dict[str, float]:
    contact = float(np.max(np.asarray(
        info.get("proprio/gripper_contact", [0.0]), dtype=np.float64,
    )))
    effector = np.asarray(info.get("proprio/effector_pos", [np.nan] * 3), dtype=np.float64).reshape(-1)
    block = np.asarray(info.get("privileged/block_0_pos", [np.nan] * 3), dtype=np.float64).reshape(-1)
    goal = np.asarray(_goal_position(env), dtype=np.float64).reshape(-1)
    return {
        "contact": contact,
        "effector_z": float(effector[2]) if effector.size >= 3 else float("nan"),
        "block_z": float(block[2]) if block.size >= 3 else float("nan"),
        "z_gap": float(effector[2] - block[2]) if effector.size >= 3 and block.size >= 3 else float("nan"),
        "block_goal_xy": float(np.linalg.norm(block[:2] - goal[:2])) if block.size >= 2 and goal.size >= 2 else float("nan"),
    }


def _run(seed: int, task_id: int, mode: str, descent_steps: int,
         approach_steps: int, post_descend_steps: int, close_steps: int,
         approach_gain: float, approach_threshold: float) -> dict[str, Any]:
    env = _make_env()
    try:
        obs, info = env.reset(seed=int(seed), options={"task_id": int(task_id), "render_goal": True})
        trace: list[dict[str, Any]] = []

        def step(action: np.ndarray, phase: str) -> bool:
            nonlocal obs, info
            obs, _reward, terminated, truncated, info = env.step(action)
            image = np.asarray(obs, dtype=np.uint8)
            block = _target_red_centroid(image)[:2]
            effector = _largest_effector_centroid(image)[:2]
            trace.append({
                "phase": phase,
                "rgb_xy_error": float(np.linalg.norm(block - effector)),
                **_label(info, env),
            })
            return bool(terminated or truncated)

        for _ in range(int(descent_steps)):
            action = np.zeros(5, dtype=np.float32)
            action[2] = -1.0
            if step(action, "pre_descend"):
                break
        if not any(row["phase"] == "pre_descend" for row in trace):
            return {"seed": int(seed), "task_id": int(task_id), "mode": mode, "trace": trace, "success": False}
        for _ in range(int(approach_steps)):
            image = np.asarray(obs, dtype=np.uint8)
            block = _target_red_centroid(image)[:2]
            effector = _largest_effector_centroid(image)[:2]
            if approach_threshold > 0.0 and float(np.linalg.norm(block - effector)) <= approach_threshold:
                break
            if step(_rgb_approach_action(image, gain=approach_gain), "rgb_approach"):
                break
        if mode == "post_contact":
            for _ in range(int(post_descend_steps)):
                action = np.zeros(5, dtype=np.float32)
                action[2] = -1.0
                if step(action, "post_descend"):
                    break
            for _ in range(int(close_steps)):
                action = np.zeros(5, dtype=np.float32)
                action[4] = 1.0
                if step(action, "post_close"):
                    break
        else:
            # Historical phase ordering: close is represented by the macro
            # bank immediately after the XY approach, with no second descent.
            for _ in range(int(close_steps)):
                action = np.zeros(5, dtype=np.float32)
                action[4] = 1.0
                if step(action, "legacy_close"):
                    break
        return {
            "seed": int(seed), "task_id": int(task_id), "mode": mode,
            "trace": trace,
            "success": bool(any(row["contact"] > 0.5 for row in trace)),
            "max_contact": float(max(row["contact"] for row in trace)),
            "min_z_gap": float(min(row["z_gap"] for row in trace if np.isfinite(row["z_gap"]))),
            "final_rgb_xy_error": float(trace[-1]["rgb_xy_error"]),
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261901)
    parser.add_argument("--roots", type=int, default=8)
    parser.add_argument("--tasks", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--descent-steps", type=int, default=15)
    parser.add_argument("--approach-steps", type=int, default=16)
    parser.add_argument("--post-descend-steps", type=int, default=6)
    parser.add_argument("--close-steps", type=int, default=5)
    parser.add_argument("--approach-gain", type=float, default=1.0)
    parser.add_argument("--approach-threshold", type=float, default=0.0)
    args = parser.parse_args()
    if args.roots < 1 or any(x < 0 for x in (args.descent_steps, args.post_descend_steps, args.close_steps)):
        raise ValueError("roots positive and phase lengths non-negative")
    if not 0.0 < args.approach_gain <= 1.0 or args.approach_threshold < 0.0:
        raise ValueError("approach gain must be in (0, 1] and threshold non-negative")
    rows = []
    for i in range(args.roots):
        seed = args.seed + 1000 + i
        task = int(args.tasks[i % len(args.tasks)])
        for mode in ("legacy", "post_contact"):
            print(f"[contact-phase] root={i} mode={mode}: RGB approach and contact schedule")
            rows.append(_run(seed, task, mode, args.descent_steps, args.approach_steps,
                             args.post_descend_steps, args.close_steps,
                             args.approach_gain, args.approach_threshold))
    summary = {}
    for mode in ("legacy", "post_contact"):
        subset = [r for r in rows if r["mode"] == mode]
        summary[mode] = {
            "roots": len(subset),
            "contact_rate": float(np.mean([r["success"] for r in subset])),
            "mean_max_contact": float(np.mean([r["max_contact"] for r in subset])),
            "mean_min_z_gap": float(np.mean([r["min_z_gap"] for r in subset])),
            "mean_final_rgb_xy_error": float(np.mean([r["final_rgb_xy_error"] for r in subset])),
        }
    payload = {
        "schema": "cube_contact_phase_audit.v1",
        "claim_boundary": "mechanism-only RGB phase-order audit; evaluator labels are not controller inputs",
        "protocol": {key: (str(value) if isinstance(value, Path) else value)
                     for key, value in vars(args).items()},
        "summary": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
