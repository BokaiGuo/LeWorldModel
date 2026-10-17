"""Online RGB-only Cube control using the calibrated short macro ranker.

The ranker is fitted on calibration-only branch points and executes a fixed
20-step, four-direction macro bank.  At runtime it receives only current RGB,
previous RGB, the rendered RGB goal, and the previous action.  Simulator state
is used only for evaluator labels.  This runner is deliberately separate from
the frozen-LeWM CEM runner: it tests whether the positive offline
goal-conditioned macro ranking can transfer to a closed-loop controller.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

# Configure headless MuJoCo before the environment package is imported.  The
# CUDA runtime is checked by torch after import, while EGL keeps RGB rendering
# available on the GPU host.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401 - registers environments
import torch

from experiments.audit_cube_macro_action_ranking import (
    _branch_macro_effects,
    _features,
    _fit,
    _macro_bank,
    _predict,
)
from experiments.audit_cube_rgb_observability import _make_env
from experiments.audit_cube_rgb_observability import (
    _goal_position,
    _largest_effector_centroid,
    _largest_red_centroid,
)


def _rgb_approach_action(image: np.ndarray) -> np.ndarray:
    """Move the visible end-effector toward the visible physical cube."""
    block = _largest_red_centroid(image)[:2]
    effector = _largest_effector_centroid(image)[:2]
    direction = np.asarray(block - effector, dtype=np.float32)
    norm = float(np.linalg.norm(direction))
    if norm < 1.0e-6:
        direction[:] = 0.0
    else:
        direction /= max(norm, 1.0)
    action = np.zeros(5, dtype=np.float32)
    action[:2] = np.clip(direction, -1.0, 1.0)
    return action


def _approach_observation(image: np.ndarray) -> dict[str, Any]:
    """Return RGB-only approach geometry and explicit validity flags."""
    block = _largest_red_centroid(image)[:2]
    effector = _largest_effector_centroid(image)[:2]
    block_valid = bool(np.linalg.norm(block) > 0.0)
    effector_valid = bool(np.linalg.norm(effector) > 0.0)
    error = block - effector
    return {
        "block_xy": block.astype(np.float32).tolist(),
        "effector_xy": effector.astype(np.float32).tolist(),
        "approach_error_xy": error.astype(np.float32).tolist(),
        "approach_error_norm": float(np.linalg.norm(error))
        if block_valid and effector_valid else None,
        "block_valid": block_valid,
        "effector_valid": effector_valid,
    }


def _capture_rgb_approach_branch(*, seed: int, task_id: int, approach_steps: int) -> dict[str, Any]:
    """Collect a calibration branch using only the frozen RGB approach law."""
    env = _make_env()
    try:
        obs, info = env.reset(
            seed=int(seed), options={"task_id": int(task_id), "render_goal": True},
        )
        goal_image = np.asarray(env.unwrapped._cur_goal_rendered, dtype=np.uint8).copy()
        previous = np.asarray(obs, dtype=np.uint8).copy()
        previous_action = np.zeros(5, dtype=np.float32)
        approach_trace: list[dict[str, Any]] = []
        for _ in range(int(approach_steps)):
            before = np.asarray(obs, dtype=np.uint8).copy()
            approach_trace.append(_approach_observation(before))
            action = _rgb_approach_action(before)
            obs, _reward, terminated, truncated, info = env.step(action)
            previous = before
            previous_action = action.copy()
            if terminated or truncated:
                raise RuntimeError("RGB approach terminated before calibration branch")
        raw = env.unwrapped
        return {
            "seed": int(seed),
            "task_id": int(task_id),
            "variation_values": copy.deepcopy(raw.variation_space.value),
            "state": np.concatenate([raw.data.qpos.copy(), raw.data.qvel.copy()]).tolist(),
            "current_image": np.asarray(obs, dtype=np.uint8).copy(),
            "previous_image": previous,
            "goal_image": goal_image,
            "previous_action": previous_action,
            "goal": _goal_position(env)[:2].copy(),
            "approach_trace": approach_trace,
        }
    finally:
        env.close()


def _episode(
    *,
    seed: int,
    task_id: int,
    bank: np.ndarray,
    fit: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: str,
    action_block: int,
    macro_cycles: int,
    approach_steps: int,
) -> dict[str, Any]:
    env = _make_env()
    try:
        obs, info = env.reset(
            seed=int(seed), options={"task_id": int(task_id), "render_goal": True},
        )
        goal = np.asarray(env.unwrapped._cur_goal_rendered, dtype=np.uint8).copy()
        previous = np.asarray(obs, dtype=np.uint8).copy()
        previous_action = np.zeros(5, dtype=np.float32)
        approach_trace: list[dict[str, Any]] = []
        for _ in range(int(approach_steps)):
            before = np.asarray(obs, dtype=np.uint8).copy()
            approach_trace.append(_approach_observation(before))
            action = _rgb_approach_action(before)
            obs, _reward, terminated, truncated, info = env.step(action)
            previous = before
            previous_action = action.copy()
            if terminated or truncated:
                break
        if terminated or truncated:
            success = np.asarray(info.get("success", 0), dtype=np.float64).reshape(-1)
            return {"seed": int(seed), "task_id": int(task_id), "selected_macro_indices": [],
                    "success": bool(success[-1] > 0.5) if len(success) else False,
                    "final_success_info": success.tolist(), "steps": int(approach_steps),
                    "approach_trace": approach_trace, "zero_macro_selected": False}
        selected: list[int] = []
        score_trace: list[list[float]] = []
        terminated = False
        truncated = False
        for _cycle in range(int(macro_cycles)):
            branch = {
                "current_image": np.asarray(obs, dtype=np.uint8).copy(),
                "previous_image": previous.copy(),
                "goal_image": goal.copy(),
                "previous_action": previous_action.copy(),
            }
            scores = np.linalg.norm(
                _predict(_features([branch], bank, device), fit), axis=1,
            )
            score_trace.append(scores.astype(np.float64).tolist())
            index = int(np.argmin(scores))
            selected.append(index)
            for action in bank[index]:
                for _ in range(int(action_block)):
                    before = np.asarray(obs, dtype=np.uint8).copy()
                    next_obs, _reward, terminated, truncated, info = env.step(action)
                    previous = before
                    previous_action = np.asarray(action, dtype=np.float32).copy()
                    obs = next_obs
                    if terminated or truncated:
                        break
                if terminated or truncated:
                    break
            if terminated or truncated:
                break
        success = np.asarray(info.get("success", 0), dtype=np.float64).reshape(-1)
        time_value = np.asarray(info.get("time", 0)).reshape(-1)
        return {
            "seed": int(seed),
            "task_id": int(task_id),
            "selected_macro_indices": selected,
            "success": bool(success[-1] > 0.5) if len(success) else False,
            "final_success_info": success.tolist(),
            "steps": int(time_value[-1]) if len(time_value) else 0,
            "approach_trace": approach_trace,
            "score_trace": score_trace,
            "zero_macro_selected": bool(any(index == 0 for index in selected)),
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261900)
    parser.add_argument("--calibration-roots", type=int, default=40)
    parser.add_argument("--test-roots", type=int, default=8)
    parser.add_argument("--tasks", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--macro-cycles", type=int, default=1)
    parser.add_argument("--approach-steps", type=int, default=20)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.calibration_roots < 8 or args.test_roots < 1:
        raise ValueError("need at least 8 calibration roots and one test root")
    if args.approach_steps < 1 or args.action_block < 1 or args.macro_cycles < 1:
        raise ValueError("approach-steps, action-block, and macro-cycles must be positive")
    bank = _macro_bank(length=20, direction_count=4, approach_steps=4)
    calibration = [
        _capture_rgb_approach_branch(
            seed=args.seed + i, task_id=int(args.tasks[i % len(args.tasks)]),
            approach_steps=args.approach_steps,
        )
        for i in range(args.calibration_roots)
    ]
    features = _features(calibration, bank, args.device)
    targets = torch.as_tensor(
        np.concatenate([
            # Reuse the audit's evaluator-only labels during calibration.
            # This branch is never used by the runtime policy.
            _branch_macro_effects(branch, bank)
            for branch in calibration
        ], axis=0),
        device=args.device, dtype=torch.float32,
    )
    fit = _fit(features, targets)
    test = [
        _episode(
            seed=args.seed + 1000 + i,
            task_id=int(args.tasks[i % len(args.tasks)]),
            bank=bank, fit=fit, device=args.device,
            action_block=args.action_block, macro_cycles=args.macro_cycles,
            approach_steps=args.approach_steps,
        )
        for i in range(args.test_roots)
    ]
    payload = {
        "schema": "live_cube_rgb_macro_ranker.v1",
        "protocol": {
            "device": args.device,
            "calibration_roots": args.calibration_roots,
            "test_roots": args.test_roots,
            "root_disjoint": True,
            "runtime_input": "current RGB, previous RGB, rendered RGB goal, previous action",
            "runtime_state_access": False,
            "bank": "20-step four-direction short macro",
            "action_block": args.action_block,
            "macro_cycles": args.macro_cycles,
            "approach_steps": args.approach_steps,
            "fit": "calibration roots only",
        },
        "test": test,
        "aggregate": {
            "roots": len(test),
            "success_rate": float(np.mean([row["success"] for row in test])),
            "selected_macro_histogram": np.bincount(
                [index for row in test for index in row["selected_macro_indices"]],
                minlength=len(bank),
            ).astype(int).tolist(),
            "zero_macro_rate": float(np.mean([
                row["zero_macro_selected"] for row in test
            ])),
            "approach_valid_rate": float(np.mean([
                all(item["block_valid"] and item["effector_valid"]
                    for item in row["approach_trace"])
                for row in test
            ])),
        },
        "claim_boundary": (
            "RGB-only macro-ranker online transfer diagnostic; no universal "
            "world-model or causal certificate claim"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
