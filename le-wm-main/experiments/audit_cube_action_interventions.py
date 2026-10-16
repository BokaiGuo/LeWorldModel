"""Strict paired action-to-task interventions for OGB-Cube.

This diagnostic separates an action-support/interface failure from a frozen
world-model ranking failure.  Each condition is evaluated from the identical
task, visual variation, and simulator state.  Privileged values are used only
after the action block to measure end-effector motion, cube motion, contact,
and goal-distance change; they never construct an observation or select an
action for a learned planner.

The intervention bank deliberately contains both task-agnostic controls and a
short approach/contact/push sequence.  The latter is a mechanism diagnostic,
not a planning result, because it uses the known cube-to-goal direction only
to test whether the environment/action interface can produce an effect.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401 - registers swm environments


def _make_env() -> gym.Env:
    return gym.make(
        "swm/OGBCube-v0",
        ob_type="states",
        render_mode=None,
        terminate_at_goal=False,
    )


def _vec(info: dict[str, Any], key: str, width: int) -> np.ndarray:
    value = np.asarray(info[key], dtype=np.float64).reshape(-1)
    if value.size != width:
        raise ValueError(f"{key} expected {width} values, got {value.shape}")
    return value


def _cube_position(info: dict[str, Any]) -> np.ndarray:
    return _vec(info, "privileged/block_0_pos", 3)


def _effector_position(info: dict[str, Any]) -> np.ndarray:
    return _vec(info, "proprio/effector_pos", 3)


def _contact(info: dict[str, Any]) -> float:
    return float(np.max(np.asarray(info["proprio/gripper_contact"], dtype=np.float64)))


def _goal_position(env: gym.Env) -> np.ndarray:
    raw = env.unwrapped
    target_block = int(getattr(raw, "_target_block", 0))
    return np.asarray(raw.cur_task_info["goal_xyzs"][target_block], dtype=np.float64)


def _action_bank(
    initial_cube: np.ndarray,
    initial_effector: np.ndarray,
    goal: np.ndarray,
) -> dict[str, list[list[float]]]:
    """Return normalized control sequences, not privileged planner actions."""
    zero = [0.0] * 5
    unit = {
        "x_plus": [1.0, 0.0, 0.0, 0.0, 0.0],
        "x_minus": [-1.0, 0.0, 0.0, 0.0, 0.0],
        "y_plus": [0.0, 1.0, 0.0, 0.0, 0.0],
        "y_minus": [0.0, -1.0, 0.0, 0.0, 0.0],
        "z_plus": [0.0, 0.0, 1.0, 0.0, 0.0],
        "z_minus": [0.0, 0.0, -1.0, 0.0, 0.0],
        "yaw_plus": [0.0, 0.0, 0.0, 1.0, 0.0],
        "gripper_close": [0.0, 0.0, 0.0, 0.0, -1.0],
    }
    direction = goal[:2] - initial_cube[:2]
    norm = float(np.linalg.norm(direction))
    direction = direction / norm if norm > 1e-8 else np.array([1.0, 0.0])
    push = [float(direction[0]), float(direction[1]), 0.0, 0.0, 0.0]
    xy_direction = initial_cube[:2] - initial_effector[:2]
    xy_norm = float(np.linalg.norm(xy_direction))
    xy_direction = xy_direction / xy_norm if xy_norm > 1e-8 else np.array([1.0, 0.0])
    xy_align = [float(xy_direction[0]), float(xy_direction[1]), 0.0, 0.0, 0.0]

    def repeat(action: list[float], count: int) -> list[list[float]]:
        return [list(action) for _ in range(count)]

    # The default Cube reset places the end effector around z=0.275 m while
    # the cube rests at z=0.02 m.  Five control steps are not enough to reach
    # contact; use a predeclared 15-step approach so the probe tests contact
    # dynamics rather than an unreachable action budget.
    approach = repeat([0.0, 0.0, -1.0, 0.0, 0.0], 15)
    close = repeat(unit["gripper_close"], 5)
    push_sequence = repeat(push, 10)
    return {
        "zero": repeat(zero, 15),
        "single_axis_x_plus": repeat(unit["x_plus"], 15),
        "single_axis_y_plus": repeat(unit["y_plus"], 15),
        "single_axis_z_minus": repeat(unit["z_minus"], 8),
        "approach_only": approach,
        "approach_close": approach + close,
        "approach_close_push_goal_direction": approach + close + push_sequence,
        "approach_open_push_goal_direction": approach + repeat(zero, 3) + push_sequence,
        "align_descend_close_push": repeat(xy_align, 5) + approach + close + push_sequence,
    }


def _capture_initial_condition(seed: int, task_id: int) -> dict[str, Any]:
    env = _make_env()
    try:
        _, info = env.reset(seed=seed, options={"task_id": task_id})
        raw = env.unwrapped
        return {
            "seed": int(seed),
            "task_id": int(task_id),
            "variation_values": copy.deepcopy(raw.variation_space.value),
            "state": np.concatenate([raw.data.qpos.copy(), raw.data.qvel.copy()]).tolist(),
            "initial_cube": _cube_position(info).tolist(),
            "initial_effector": _effector_position(info).tolist(),
            "goal": _goal_position(env).tolist(),
        }
    finally:
        env.close()


def _run_condition(
    env: gym.Env,
    initial: dict[str, Any],
    actions: list[list[float]],
) -> dict[str, Any]:
    _, info = env.reset(
            seed=int(initial["seed"]),
            options={
                "task_id": int(initial["task_id"]),
                "variation_values": copy.deepcopy(initial["variation_values"]),
                "state": np.asarray(initial["state"], dtype=np.float64),
            },
        )
    start_cube = _cube_position(info)
    start_effector = _effector_position(info)
    goal = np.asarray(initial["goal"], dtype=np.float64)
    cube_history = [start_cube.copy()]
    effector_history = [start_effector.copy()]
    contact_history = [_contact(info)]
    for action in actions:
        if not np.all(np.isfinite(action)) or np.any(np.abs(np.asarray(action)) > 1.0 + 1e-8):
            raise ValueError(f"invalid normalized action {action}")
        _, _, _, _, info = env.step(np.asarray(action, dtype=np.float32))
        cube_history.append(_cube_position(info))
        effector_history.append(_effector_position(info))
        contact_history.append(_contact(info))
    cubes = np.asarray(cube_history)
    effectors = np.asarray(effector_history)
    contacts = np.asarray(contact_history)
    start_distance = float(np.linalg.norm(start_cube - goal))
    final_distance = float(np.linalg.norm(cubes[-1] - goal))
    return {
            "n_control_steps": len(actions),
            "start_cube": start_cube.tolist(),
            "final_cube": cubes[-1].tolist(),
            "cube_displacement_m": float(np.linalg.norm(cubes[-1] - start_cube)),
            "cube_max_displacement_m": float(np.max(np.linalg.norm(cubes - start_cube, axis=1))),
            "start_effector": start_effector.tolist(),
            "final_effector": effectors[-1].tolist(),
            "effector_displacement_m": float(np.linalg.norm(effectors[-1] - start_effector)),
            "contact_max": float(np.max(contacts)),
            "contact_steps": int(np.sum(contacts > 1e-5)),
            "start_goal_distance_m": start_distance,
            "final_goal_distance_m": final_distance,
            "goal_distance_change_m": final_distance - start_distance,
            "cube_history": cubes.tolist(),
            "effector_history": effectors.tolist(),
            "contact_history": contacts.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--tasks", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--seeds-per-task", type=int, default=2)
    args = parser.parse_args()
    if args.seeds_per_task < 1:
        raise ValueError("seeds-per-task must be positive")
    print("Running strict paired Cube action interventions: interface/contact support, not planner success.")
    records: list[dict[str, Any]] = []
    for task_id in args.tasks:
        for offset in range(args.seeds_per_task):
            initial = _capture_initial_condition(args.seed + 100 * int(task_id) + offset, int(task_id))
            bank = _action_bank(
                np.asarray(initial["initial_cube"], dtype=np.float64),
                np.asarray(initial["initial_effector"], dtype=np.float64),
                np.asarray(initial["goal"], dtype=np.float64),
            )
            env = _make_env()
            try:
                for condition, actions in bank.items():
                    result = _run_condition(env, initial, actions)
                    records.append(
                        {
                            "task_id": int(task_id),
                            "task_name": f"task{task_id}",
                            "seed": int(initial["seed"]),
                            "condition": condition,
                            "initial": initial,
                            "result": result,
                        }
                    )
                    print(
                        f"task={task_id} seed={initial['seed']} condition={condition} "
                        f"cube_disp={result['cube_displacement_m']:.6g} "
                        f"contact_steps={result['contact_steps']} "
                        f"goal_delta={result['goal_distance_change_m']:.6g}"
                    )
            finally:
                env.close()
    payload = {
        "schema": "cube_action_interventions.v1",
        "protocol": {
            "environment": "swm/OGBCube-v0",
            "observation_for_action": "none; raw physics intervention only",
            "privileged_measurements": "post_step evaluator only",
            "strict_initial_state_pairing": True,
            "task_ids": [int(x) for x in args.tasks],
            "seeds_per_task": int(args.seeds_per_task),
            "normalized_action_bounds": [-1.0, 1.0],
            "control_timestep_s": 0.05,
            "physics_timestep_s": 0.002,
            "physics_steps_per_control_step": 25,
            "claim_boundary": "action support/contact mechanism diagnostic; not a learned-planner result",
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=lambda value: value.tolist() if isinstance(value, np.ndarray) else str(value),
        )
        + "\n"
    )
    print(f"wrote {args.output} records={len(records)}")


if __name__ == "__main__":
    main()
