"""Verified Contact-Ranking Ladder for PushT.

This is a privileged *mechanism* experiment, not a learned visual planner.
It answers three nested questions on identical contact roots and an identical
declared candidate bank:

1. Do mirrored +torque/-torque/zero pulses have a measurable same-state
   orientation contrast?
2. Does a real-successor two-pulse oracle have a feasible short-horizon path?
3. What non-privileged runtime boundary must a later RGB ranker obey?

Simulator state and successor rollouts are used only by the oracle/audit.  A
future deployable ranker may consume only RGB history and declared actions;
that interface is recorded explicitly in every artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch

import stable_worldmodel  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.audit_pusht_successor_oracle_ceiling import successor as oracle_successor
from experiments.run_pusht_mode_aware_shared_candidate import (
    angle_error,
    boundary_inward_normal,
    candidate_actions,
    make_boundary_contact_roots,
    pose_cost,
)

gym.logger.min_level = 40


def non_privileged_rgb_runtime_boundary() -> dict[str, Any]:
    """Declare the only permitted interface for a future RGB ranker."""

    return {
        "runtime_successor_access": False,
        "runtime_privileged_state_access": False,
        "runtime_observation": "RGB history and declared actions only",
        "status": "future_interface_only_not_evaluated_here",
    }


def validate_declared_oracle_bank(actions: np.ndarray, *, expected_count: int) -> None:
    """Fail closed when the runtime bank differs from the declared oracle bank."""

    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[0] != int(expected_count):
        actual = int(actions.shape[0]) if actions.ndim >= 1 else 0
        raise ValueError(
            f"declared {int(expected_count)}-action oracle bank required; got {actual} candidates"
        )


def resolve_device(requested: str) -> torch.device:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    torch.zeros(8, device=device).sum().item()
    return device


def paired_torque_actions(
    env: gym.Env | Any,
    state: np.ndarray,
    *,
    inward_magnitude: float,
    tangent_magnitude: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return matched inward +tangent/-tangent/zero actions from one state."""

    if inward_magnitude < 0.0 or tangent_magnitude <= 0.0:
        raise ValueError("inward magnitude must be non-negative and tangent positive")
    normal = np.asarray(boundary_inward_normal(env, np.asarray(state, dtype=np.float64)), dtype=np.float64)
    normal /= max(float(np.linalg.norm(normal)), 1.0e-12)
    tangent = np.asarray([-normal[1], normal[0]], dtype=np.float64)
    plus = float(inward_magnitude) * normal + float(tangent_magnitude) * tangent
    minus = float(inward_magnitude) * normal - float(tangent_magnitude) * tangent
    if np.max(np.abs(np.concatenate((plus, minus)))) > 1.0 + 1.0e-8:
        raise ValueError("paired torque action exceeds PushT action bounds")
    return plus, minus, np.zeros(2, dtype=np.float64)


def audit_same_state_pair(
    source: np.ndarray,
    goal: np.ndarray,
    *,
    plus: np.ndarray,
    minus: np.ndarray,
    zero: np.ndarray,
    successor_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    angle_error_fn: Callable[[float, float], float],
) -> dict[str, Any]:
    """Evaluate matched pulse effects from exactly the same source state."""

    source = np.asarray(source, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    before = float(angle_error_fn(float(source[4]), float(goal[4])))
    candidates = {"plus": plus, "minus": minus, "zero": zero}
    outcomes: dict[str, dict[str, Any]] = {}
    for name, action in candidates.items():
        after = np.asarray(successor_fn(source.copy(), np.asarray(action, dtype=np.float64)), dtype=np.float64)
        after_error = float(angle_error_fn(float(after[4]), float(goal[4])))
        outcomes[name] = {
            "action": np.asarray(action, dtype=np.float64).tolist(),
            "after_angle_error": after_error,
            "angle_improvement": float(before - after_error),
            "after_xy": float(np.linalg.norm(after[2:4] - goal[2:4])),
        }
    best_member = min(outcomes, key=lambda name: outcomes[name]["after_angle_error"])
    return {
        "same_source_state": True,
        "before_angle_error": before,
        "outcomes": outcomes,
        "plus_minus_contrast": float(
            outcomes["plus"]["angle_improvement"] - outcomes["minus"]["angle_improvement"]
        ),
        "best_member": best_member,
        **non_privileged_rgb_runtime_boundary(),
    }


def select_two_pulse_oracle(
    state: np.ndarray,
    actions: np.ndarray,
    *,
    successor_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    cost_fn: Callable[[np.ndarray], float],
) -> dict[str, Any]:
    """Exhaustively choose the best real-successor ordered action pair."""

    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[0] < 1:
        raise ValueError("actions must be a non-empty [candidate, action_dim] array")
    best: dict[str, Any] | None = None
    for first_index, first in enumerate(actions):
        after_first = np.asarray(successor_fn(np.asarray(state, dtype=np.float64).copy(), first), dtype=np.float64)
        for second_index, second in enumerate(actions):
            after_second = np.asarray(successor_fn(after_first.copy(), second), dtype=np.float64)
            terminal_cost = float(cost_fn(after_second))
            record = {
                "first_index": int(first_index),
                "second_index": int(second_index),
                "terminal_cost": terminal_cost,
                "after_first": after_first,
                "after_second": after_second,
            }
            if best is None or terminal_cost < float(best["terminal_cost"]):
                best = record
    assert best is not None
    return best


def _real_successor(
    env: gym.Env,
    state: np.ndarray,
    goal: np.ndarray,
    action: np.ndarray,
    pulse_steps: int,
) -> np.ndarray:
    return oracle_successor(env, state, goal, action, pulse_steps)


def run_episode(env: gym.Env, root: dict[str, np.ndarray], args: argparse.Namespace) -> dict[str, Any]:
    """Run paired mechanism audits and a real-successor two-pulse oracle."""

    current = np.asarray(root["state"], dtype=np.float64).copy()
    goal = np.asarray(root["goal"], dtype=np.float64).copy()
    trace: list[dict[str, Any]] = []
    pair_audits: list[dict[str, Any]] = []
    for macro_step in range(int(args.max_steps)):
        if pose_cost(current, goal) < float(args.stop_cost):
            break
        plus, minus, zero = paired_torque_actions(
            env, current,
            inward_magnitude=float(args.inward_magnitude),
            tangent_magnitude=float(args.probe_tangent_magnitude),
        )
        pair_audit = audit_same_state_pair(
            current, goal, plus=plus, minus=minus, zero=zero,
            successor_fn=lambda state, action: _real_successor(
                env, state, goal, action, int(args.pulse_steps)
            ),
            angle_error_fn=angle_error,
        )
        pair_audits.append(pair_audit)
        actions = candidate_actions(
            env, current, tuple(args.tangent_magnitudes), float(args.inward_magnitude),
            goal=goal, boundary_action_frame=True,
            contact_normal_magnitudes=tuple(args.contact_normal_magnitudes),
        )
        validate_declared_oracle_bank(actions, expected_count=int(args.expected_candidate_count))
        choice = select_two_pulse_oracle(
            current, actions,
            successor_fn=lambda state, action: _real_successor(
                env, state, goal, action, int(args.pulse_steps)
            ),
            cost_fn=lambda state: pose_cost(state, goal),
        )
        current = np.asarray(choice["after_second"], dtype=np.float64)
        trace.append({
            "macro_step": int(macro_step),
            "candidate_count": int(len(actions)),
            "first_index": int(choice["first_index"]),
            "second_index": int(choice["second_index"]),
            "terminal_cost": float(choice["terminal_cost"]),
            "angle_after": float(angle_error(current[4], goal[4])),
            "xy_after": float(np.linalg.norm(current[2:4] - goal[2:4])),
            "pair_plus_minus_contrast": float(pair_audit["plus_minus_contrast"]),
            "pair_best_member": str(pair_audit["best_member"]),
        })
    final_xy = float(np.linalg.norm(current[2:4] - goal[2:4]))
    final_angle = float(angle_error(current[4], goal[4]))
    return {
        "initial_angle": float(angle_error(root["state"][4], goal[4])),
        "final_angle": final_angle,
        "final_xy": final_xy,
        "pose_success": bool(final_xy < 20.0 and final_angle < np.pi / 9.0),
        "pair_audits": pair_audits,
        "trace": trace,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261061)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--pulse-steps", type=int, default=2)
    parser.add_argument("--stop-cost", type=float, default=1.0)
    parser.add_argument("--inward-magnitude", type=float, default=0.08)
    parser.add_argument("--probe-tangent-magnitude", type=float, default=0.12)
    # Seven tangent values times eight normal values, plus the explicit
    # separation candidate, give the declared 113-action extended oracle bank.
    parser.add_argument("--tangent-magnitudes", type=float, nargs="+", default=[0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.24])
    parser.add_argument("--contact-normal-magnitudes", type=float, nargs="+", default=[0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.16])
    parser.add_argument("--expected-candidate-count", type=int, default=113)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    try:
        roots = make_boundary_contact_roots(env, int(args.episodes), int(args.seed) + 2000)
        rows = [run_episode(env, root, args) for root in roots]
    finally:
        env.close()
    contrasts = [
        audit["plus_minus_contrast"]
        for row in rows for audit in row["pair_audits"]
    ]
    report = {
        "schema": "pusht_verified_contact_ranking_ladder.v1",
        "claim_boundary": (
            "Privileged same-state pulse and real-successor oracle mechanism audit only; "
            "it makes no learned-planner, frozen-LeWM, RGB-planning, or universal-world-model claim."
        ),
        "protocol": {
            "environment": "swm/PushT-v1",
            "root_protocol": "boundary_contact",
            "root_seed": int(args.seed) + 2000,
            "episodes": len(rows),
            "max_two_pulse_macros": int(args.max_steps),
            "pulse_steps": int(args.pulse_steps),
            "candidate_bank": "boundary-frame inward/tangent/normal bank shared with existing successor oracle",
            "candidate_bank_config": {
                "inward_magnitude": float(args.inward_magnitude),
                "tangent_magnitudes": list(args.tangent_magnitudes),
                "contact_normal_magnitudes": list(args.contact_normal_magnitudes),
                "expected_candidate_count": int(args.expected_candidate_count),
            },
            "oracle_runtime_successor_access": True,
            "resolved_device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "later_rgb_runtime_boundary": non_privileged_rgb_runtime_boundary(),
        "metrics": {
            "pose_success": float(np.mean([row["pose_success"] for row in rows])),
            "mean_final_angle": float(np.mean([row["final_angle"] for row in rows])),
            "mean_final_xy": float(np.mean([row["final_xy"] for row in rows])),
            "mean_pair_plus_minus_contrast": float(np.mean(contrasts)) if contrasts else 0.0,
            "pair_audit_count": len(contrasts),
        },
        "rows": rows,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({"output": str(args.output), "metrics": report["metrics"], "claim_boundary": report["claim_boundary"]}, indent=2))


if __name__ == "__main__":
    main()
