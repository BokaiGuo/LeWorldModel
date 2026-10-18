"""Privileged ceiling for a pick-and-place Cube action primitive.

The existing four-direction Cube bank closes the gripper and then applies a
planar push.  OGB-Cube's task oracle instead grasps, lifts, transports in XY,
lowers, and releases.  This audit tests that action-interface hypothesis with
the same RGB prelude and evaluator-only oracle selection.  It is not RGB
planning or learned-dynamics evidence.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from experiments.audit_cube_action_interventions import (
    _capture_initial_condition,
    _cube_position,
)
from experiments.audit_cube_planner_ceiling import (
    _capture_rgb_decision_state,
    _make_rgb_env,
    _oracle_for_bank,
    _reset_from_initial,
)
from experiments.run_live_cube_rgb_macro_ranker import _fit_rgb_pixel_action_mapping
from experiments.run_cube_rgb_visual_servo import (
    _capture_initial,
    _fit_rgb_action_map,
)


def _pick_place_bank(
    *, length: int = 50, direction_count: int = 8,
    close_steps: int = 5, lift_steps: int = 8,
    transport_steps: int = 20, lower_steps: int = 8,
    release_steps: int = 5,
) -> np.ndarray:
    durations = close_steps + lift_steps + transport_steps + lower_steps + release_steps
    if length < durations or direction_count not in (4, 8):
        raise ValueError("length is too short or direction_count is invalid")
    angles = np.arange(direction_count, dtype=np.float32) * (2.0 * np.pi / direction_count)
    directions = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)
    return _pick_place_bank_for_directions(
        directions, length=length, close_steps=close_steps, lift_steps=lift_steps,
        transport_steps=transport_steps, lower_steps=lower_steps,
        release_steps=release_steps,
    )


def _pick_place_bank_for_directions(
    directions: np.ndarray, *, length: int = 50, close_steps: int = 5,
    lift_steps: int = 15, transport_steps: int = 20, lower_steps: int = 8,
    release_steps: int = 5,
) -> np.ndarray:
    """Build pick-place candidates for a supplied XY direction set."""
    durations = close_steps + lift_steps + transport_steps + lower_steps + release_steps
    if length < durations:
        raise ValueError("length is too short for the pick-place primitive")
    directions = np.asarray(directions, dtype=np.float32).reshape(-1, 2)
    norms = np.linalg.norm(directions, axis=1)
    if len(directions) == 0 or not np.isfinite(directions).all() or np.any(norms < 1.0e-8):
        raise ValueError("directions must be finite nonzero XY vectors")
    directions = directions / norms[:, None]
    bank = [np.zeros((length, 5), dtype=np.float32)]
    for direction in directions:
        sequence = np.zeros((length, 5), dtype=np.float32)
        cursor = 0
        sequence[cursor:cursor + close_steps, 4] = 1.0
        cursor += close_steps
        sequence[cursor:cursor + lift_steps, 2] = 1.0
        cursor += lift_steps
        sequence[cursor:cursor + transport_steps, :2] = direction
        cursor += transport_steps
        sequence[cursor:cursor + lower_steps, 2] = -1.0
        cursor += lower_steps
        sequence[cursor:cursor + release_steps, 4] = -1.0
        bank.append(sequence)
    return np.stack(bank, axis=0)


def _goal_relative_bank(
    decision_state: dict[str, Any], *, length: int = 95,
    transport_magnitude: float = 0.2, transport_steps: int = 50,
) -> np.ndarray:
    """Privileged one-candidate bank aligned with cube-to-goal world geometry."""
    env = _make_rgb_env()
    try:
        _obs, info = _reset_from_initial(env, decision_state)
        cube = np.asarray(_cube_position(info), dtype=np.float64)[:2]
        goal = np.asarray(decision_state["goal"], dtype=np.float64)[:2]
        direction = goal - cube
        bank = _pick_place_bank_for_directions(
            direction.reshape(1, 2), length=length, close_steps=5,
            lift_steps=15, transport_steps=int(transport_steps),
            lower_steps=15, release_steps=8,
        )
        transport_start = 5 + 15
        bank[:, transport_start:transport_start + int(transport_steps), :2] *= float(
            transport_magnitude
        )
        return bank
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261901)
    parser.add_argument("--roots", type=int, default=3)
    parser.add_argument("--tasks", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--approach-calibration-roots", type=int, default=8)
    parser.add_argument("--descent-steps", type=int, default=15)
    parser.add_argument("--approach-steps", type=int, default=32)
    parser.add_argument("--goal-tolerance-m", type=float, default=0.05)
    parser.add_argument("--goal-transport-magnitude", type=float, default=0.2)
    parser.add_argument("--transport-steps", type=int, default=20)
    args = parser.parse_args()
    if not np.isfinite(args.goal_transport_magnitude) or args.goal_transport_magnitude <= 0.0:
        raise ValueError("goal-transport-magnitude must be finite and positive")
    if args.transport_steps < 1:
        raise ValueError("transport-steps must be positive")
    bank_length = max(50, 5 + 8 + args.transport_steps + 8 + 5)
    bank = _pick_place_bank(length=bank_length, transport_steps=args.transport_steps)
    mapping, probe = _fit_rgb_pixel_action_mapping(
        seed=args.seed, tasks=args.tasks, roots=args.approach_calibration_roots,
        descent_steps=args.descent_steps, probe_steps=4,
    )
    calibration_initials = [
        _capture_initial(args.seed + index, int(args.tasks[index % len(args.tasks)]))
        for index in range(args.approach_calibration_roots)
    ]
    approach_model, approach_report = _fit_rgb_action_map(
        calibration_initials, 4, args.descent_steps, "cpu", 1.0e-2, "bilinear",
    )
    rows = []
    for index in range(args.roots):
        task = int(args.tasks[index % len(args.tasks)])
        initial = _capture_initial_condition(args.seed + 1000 + index, task)
        decision = _capture_rgb_decision_state(
            initial, mapping, args.descent_steps, args.approach_steps,
            approach_model=approach_model,
        )
        result = _oracle_for_bank(decision, bank, args.goal_tolerance_m, 1)
        goal_relative = _oracle_for_bank(
            decision,
            _goal_relative_bank(
                decision,
                length=max(95, 5 + 15 + args.transport_steps + 15 + 8),
                transport_magnitude=args.goal_transport_magnitude,
                transport_steps=args.transport_steps,
            ),
            args.goal_tolerance_m, 1,
        )
        rows.append({
            "root_index": index, "seed": int(initial["seed"]), "task_id": task,
            "oracle": result,
            "goal_relative_oracle": goal_relative,
        })
        print(
            f"root={index} oracle_distance={result['oracle']['final_goal_distance_m']:.6g} "
            f"reachable={result['reachable']} goal_relative_distance={goal_relative['oracle']['final_goal_distance_m']:.6g} "
            f"goal_relative_reachable={goal_relative['reachable']} contact={result['oracle']['contact_max']:.4g}"
        )
    payload = {
        "schema": "cube_pick_place_ceiling.v1",
        "claim_boundary": "privileged pick-and-place candidate ceiling; not RGB planning or learned-dynamics evidence",
        "protocol": {
            "root_disjoint": True,
            "candidate_selection": "privileged post-hoc final goal distance",
            "runtime_policy_input": "none; evaluator-only oracle",
            "prelude": "RGB Jacobian descent and approach, then frozen decision state",
            "bank": "8-direction close-lift-transport-lower-release",
            "bank_shape": list(bank.shape),
            "descent_steps": args.descent_steps,
            "approach_steps": args.approach_steps,
            "approach_calibration_roots": args.approach_calibration_roots,
            "goal_transport_magnitude": float(args.goal_transport_magnitude),
            "transport_steps": int(args.transport_steps),
            "rgb_probe_diagnostics": probe,
            "approach_chart_diagnostics": approach_report,
        },
        "rows": rows,
        "aggregate": {
            "roots": len(rows),
            "reachability_rate": float(np.mean([r["oracle"]["reachable"] for r in rows])),
            "mean_oracle_distance_m": float(np.mean([
                r["oracle"]["oracle"]["final_goal_distance_m"] for r in rows
            ])),
            "goal_relative_reachability_rate": float(np.mean([
                r["goal_relative_oracle"]["reachable"] for r in rows
            ])),
            "goal_relative_mean_oracle_distance_m": float(np.mean([
                r["goal_relative_oracle"]["oracle"]["final_goal_distance_m"] for r in rows
            ])),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
