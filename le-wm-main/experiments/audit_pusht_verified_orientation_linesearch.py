"""Active observed-effect orientation line search for PushT.

This is a deliberately narrow privileged diagnostic.  It does not query a
counterfactual successor: each signed tangent pulse is executed, the current
state is re-observed, and the next sign is chosen from the realized angle
error.  Its purpose is to test whether the remaining PushT failure is a
wrong-sign/stiction interface problem or an intrinsic within-contact pose
closure boundary.

It is not evidence for frozen RGB/LeWM causal dynamics.  A positive result
only establishes a physical-interface ceiling for an observed-effect policy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

import stable_worldmodel  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_pusht_mode_aware_shared_candidate import (
    angle_error,
    boundary_inward_normal,
    make_boundary_contact_roots,
    physical_contact_geometry,
)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    # Keep the experiment's device contract explicit even though MuJoCo/Pymunk
    # stepping is CPU-side: candidate tensors and the preflight allocation use
    # the requested device, and --device cuda never silently falls back.
    torch.zeros(4, device=device).sum().item()
    return device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--cycles", type=int, default=20)
    parser.add_argument("--probe-steps", type=int, default=1)
    parser.add_argument("--commit-steps", type=int, default=2)
    parser.add_argument("--normal-magnitude", type=float, default=0.10)
    parser.add_argument("--tangent-magnitude", type=float, default=0.16)
    parser.add_argument("--xy-tolerance", type=float, default=22.0)
    parser.add_argument("--xy-restore-steps", type=int, default=3)
    parser.add_argument("--xy-restore-magnitude", type=float, default=0.06)
    return parser.parse_args()


def safe_step(env: gym.Env, action: np.ndarray) -> np.ndarray:
    obs, _, terminated, truncated, _ = env.step(np.asarray(action, dtype=np.float32))
    if terminated or truncated:
        return np.asarray(obs["state"], dtype=np.float64).copy()
    return np.asarray(obs["state"], dtype=np.float64).copy()


def signed_contact_action(
    env: gym.Env, state: np.ndarray, sign: float,
    normal_magnitude: float, tangent_magnitude: float,
) -> np.ndarray:
    normal = boundary_inward_normal(env, state)
    tangent = np.asarray([-normal[1], normal[0]], dtype=np.float64)
    action = normal_magnitude * normal + float(sign) * tangent_magnitude * tangent
    return np.clip(action, -1.0, 1.0)


def restore_xy(
    env: gym.Env, state: np.ndarray, target_xy: np.ndarray,
    normal_magnitude: float, max_steps: int,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Use observed XY progress to restore the translation tube.

    The action sign is selected from the desired block displacement and then
    re-evaluated after every pulse.  This remains an observed-effect policy:
    no counterfactual successor is queried.
    """
    current = np.asarray(state, dtype=np.float64).copy()
    records: list[dict[str, float]] = []
    for _ in range(max(0, int(max_steps))):
        delta = np.asarray(target_xy, dtype=np.float64) - current[2:4]
        before_error = float(np.linalg.norm(delta))
        if before_error <= 1.0e-6:
            break
        normal = boundary_inward_normal(env, current)
        tangent = np.asarray([-normal[1], normal[0]], dtype=np.float64)
        directions = [normal, -normal, tangent, -tangent]
        # Geometry supplies only an ordering for the probe bank; admission is
        # based on the observed XY successor, not on the geometric proxy.
        directions.sort(key=lambda direction: -float(np.dot(direction, delta)))
        # Every restoration candidate retains an inward lease.  Pure tangent
        # probes were able to reduce XY residual only by separating from the
        # object, so they are no longer admissible restoration actions.
        tangent_scale = float(normal_magnitude)
        action_bank = [
            float(normal_magnitude) * normal,
            -float(normal_magnitude) * normal,
            float(normal_magnitude) * normal + tangent_scale * tangent,
            float(normal_magnitude) * normal - tangent_scale * tangent,
        ]
        order = np.argsort([-float(np.dot(direction, delta)) for direction in directions])
        action_bank = [action_bank[int(index)] for index in order]
        action = action_bank[0]
        after = safe_step(env, action)
        after_error = float(np.linalg.norm(np.asarray(target_xy) - after[2:4]))
        records.append({
            "before_xy_error": before_error,
            "after_xy_error": after_error,
            "probe_axis": float(np.argmax([abs(np.dot(direction, normal)) for direction in directions[:1]])),
            "improved": float(after_error <= before_error),
        })
        current = after
        if after_error > before_error:
            # The local normal convention can flip under sliding.  One
            # observed opposite pulse is the only permitted correction; if it
            # also fails, return to orientation probing rather than chase an
            # unverified contact model.
            alternate = action_bank[1]
            alternate = safe_step(env, alternate)
            alternate_error = float(np.linalg.norm(np.asarray(target_xy) - alternate[2:4]))
            records[-1]["alternate_xy_error"] = alternate_error
            records[-1]["alternate_improved"] = float(alternate_error <= after_error)
            current = alternate if alternate_error <= after_error else after
            if alternate_error > before_error and after_error > before_error:
                # One tangent direction is allowed as a final local chart
                # probe; if it also fails, abstain instead of accumulating
                # unverified drift.
                tangent_probe = action_bank[2]
                tangent_after = safe_step(env, tangent_probe)
                tangent_error = float(np.linalg.norm(np.asarray(target_xy) - tangent_after[2:4]))
                records[-1]["tangent_xy_error"] = tangent_error
                records[-1]["tangent_improved"] = float(tangent_error <= before_error)
                current = tangent_after if tangent_error <= before_error else current
                if tangent_error > before_error:
                    break
    return current, records


def run_episode(env: gym.Env, root: dict[str, np.ndarray], args: argparse.Namespace) -> dict[str, Any]:
    state, _ = env.reset(seed=0, options={
        "state": np.asarray(root["state"]).copy(),
        "goal_state": np.asarray(root["goal"]).copy(),
    })
    current = np.asarray(state["state"], dtype=np.float64).copy()
    goal = np.asarray(root["goal"], dtype=np.float64).copy()
    # Isolate orientation closure: retain the observed contact XY and ask for
    # the root's target orientation.  This removes long-range planner error
    # from the active-probe mechanism test.
    goal[2:4] = current[2:4]
    target_xy = current[2:4].copy()
    initial_error = angle_error(current[4], goal[4])
    trace: list[dict[str, Any]] = []
    accepted = 0
    rejected = 0
    for cycle in range(int(args.cycles)):
        before_probe = current.copy()
        before_error = angle_error(current[4], goal[4])
        if before_error <= np.pi / 18.0:
            break
        desired = float(np.sign(((goal[4] - current[4] + np.pi) % (2 * np.pi)) - np.pi))
        if desired == 0.0:
            break
        # Two-sided local finite difference. The first sign is deliberately
        # deterministic; if it does not improve the observed task error, the
        # opposite sign is tried from the newly observed state.
        first_sign = 1.0 if cycle % 2 == 0 else -1.0
        first_action = signed_contact_action(
            env, current, first_sign, args.normal_magnitude, args.tangent_magnitude,
        )
        first_after = safe_step(env, first_action)
        first_error = angle_error(first_after[4], goal[4])
        chosen_sign = first_sign
        chosen_after = first_after
        if first_error > before_error:
            opposite = signed_contact_action(
                env, first_after, -first_sign, args.normal_magnitude, args.tangent_magnitude,
            )
            second_after = safe_step(env, opposite)
            second_error = angle_error(second_after[4], goal[4])
            rejected += 1
            if second_error < first_error:
                chosen_sign = -first_sign
                chosen_after = second_after
        else:
            accepted += 1
        current = chosen_after
        for _ in range(int(args.commit_steps)):
            if angle_error(current[4], goal[4]) <= np.pi / 18.0:
                break
            action = signed_contact_action(
                env, current, chosen_sign, args.normal_magnitude, args.tangent_magnitude,
            )
            current = safe_step(env, action)
        current, restore_trace = restore_xy(
            env, current, target_xy, args.xy_restore_magnitude,
            args.xy_restore_steps,
        )
        geometry = physical_contact_geometry(env, current)
        trace.append({
            "cycle": cycle,
            "before_angle_error": before_error,
            "first_angle_error": first_error,
            "chosen_angle_error": angle_error(current[4], goal[4]),
            "first_sign": first_sign,
            "chosen_sign": chosen_sign,
            "desired_sign": desired,
            "accepted_first_probe": first_error <= before_error,
            "boundary_distance": float(geometry["distance"]),
            "xy_error": float(np.linalg.norm(current[2:4] - goal[2:4])),
            "xy_budget_ok": bool(
                np.linalg.norm(current[2:4] - goal[2:4])
                <= float(args.xy_tolerance)
            ),
            "xy_restore": restore_trace,
        })
    final_error = angle_error(current[4], goal[4])
    return {
        "initial_angle_error": initial_error,
        "final_angle_error": final_error,
        "xy_error": float(np.linalg.norm(current[2:4] - goal[2:4])),
        "strict_orientation_success": bool(final_error < np.pi / 9.0),
        "contact_preserved": bool(all(row["boundary_distance"] <= 70.0 for row in trace)),
        "xy_budget_preserved": bool(all(row["xy_budget_ok"] for row in trace)),
        "accepted_first_probe": accepted,
        "rejected_first_probe": rejected,
        "trace": trace,
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    roots = make_boundary_contact_roots(env, int(args.episodes), int(args.seed))
    rows = [run_episode(env, root, args) for root in roots]
    report = {
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "episodes": len(rows),
        "config": vars(args) | {"output": str(args.output)},
        "metrics": {
            "orientation_success": float(np.mean([row["strict_orientation_success"] for row in rows])),
            "contact_preservation": float(np.mean([row["contact_preserved"] for row in rows])),
            "xy_budget_preservation": float(np.mean([row["xy_budget_preserved"] for row in rows])),
            "mean_initial_angle_error": float(np.mean([row["initial_angle_error"] for row in rows])),
            "mean_final_angle_error": float(np.mean([row["final_angle_error"] for row in rows])),
            "mean_xy_error": float(np.mean([row["xy_error"] for row in rows])),
        },
        "rows": rows,
        "claim_boundary": (
            "Privileged observed-effect orientation-interface diagnostic. It tests "
            "whether active signed pulses can close orientation after XY is fixed; "
            "it is not RGB, frozen-LeWM, or universal-world-model evidence."
        ),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({"output": str(args.output), "device": str(device), "metrics": report["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
