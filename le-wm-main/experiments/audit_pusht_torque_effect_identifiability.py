"""Audit whether near-goal PushT states admit a recoverable torque action.

This is a privileged simulator-state mechanism audit.  It does not train or
evaluate a visual LeWM planner.  For genuine trajectory states in the
approach/contact manifold, it evaluates a complete action fan from the exact
same state and records whether any action can improve orientation while
remaining inside a bounded translation tube.

The audit is deliberately separated from the learned planner because a failed
learned selector is only actionable if the simulator exposes a feasible local
effect to learn.  Each probe resets to the source state, executes one action,
and observes the one-step successor.  No successor is used to choose the
state or to construct another state's fan.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

import stable_worldmodel  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_pusht_contact_history_velocity_diagnostic import (
    collect_trajectory_data,
    mode_for_state,
    unit,
)
from experiments.run_pusht_object_centric_learned_planner import angle_error
from experiments.run_pusht_receding_block_pose_planner import make_state_only_env
from experiments.pusht_state import raw_block_angle

gym.logger.min_level = 40

DISTANCE_STRATA = ("0_20", "20_35", "35_45")
CONTACT_STRATA = ("stable_contact", "near_contact")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--states-per-stratum", type=int, default=12)
    parser.add_argument("--pool-transitions", type=int, default=2048)
    parser.add_argument("--directions", type=int, default=32)
    parser.add_argument("--magnitudes", nargs="+", type=float, default=[0.35, 0.55, 0.75])
    parser.add_argument("--xy-margin", type=float, default=4.0)
    parser.add_argument("--angle-target", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--bootstrap", type=int, default=4000)
    return parser.parse_args()


def distance_stratum(value: float) -> str | None:
    if value < 20.0:
        return "0_20"
    if value < 35.0:
        return "20_35"
    if value < 45.0:
        return "35_45"
    return None


def contact_stratum(state: np.ndarray) -> str:
    distance = float(np.linalg.norm(np.asarray(state)[2:4] - np.asarray(state)[:2]))
    return "stable_contact" if distance <= 45.0 else "near_contact"


def make_goal(state: np.ndarray, distance: float, seed: int, angle_target: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    direction = unit(rng.normal(size=2))
    goal = np.asarray(state, dtype=np.float64).copy()
    goal[2:4] = goal[2:4] + distance * direction
    sign = 1.0 if seed % 2 else -1.0
    goal[4] = float(raw_block_angle(state) + sign * angle_target)
    goal[5:7] = 0.0
    return goal


def fan_actions(directions: int, magnitudes: list[float]) -> np.ndarray:
    if directions < 8:
        raise ValueError("at least eight action directions are required")
    offsets = np.linspace(-np.pi, np.pi, directions, endpoint=False)
    actions = [np.zeros(2, dtype=np.float32)]
    for magnitude in magnitudes:
        if not 0.0 < magnitude <= 1.0:
            raise ValueError("action magnitudes must lie in (0, 1]")
        for offset in offsets:
            actions.append(np.asarray(
                [magnitude * np.cos(offset), magnitude * np.sin(offset)],
                dtype=np.float32,
            ))
    return np.asarray(actions, dtype=np.float32)


def probe_state(
    env: gym.Env,
    state: np.ndarray,
    goal: np.ndarray,
    action: np.ndarray,
    index: int,
) -> dict[str, Any]:
    env.reset(seed=index, options={"state": state.copy(), "goal_state": goal.copy()})
    observation, _, terminated, truncated, _ = env.step(action)
    after = np.asarray(observation["state"], dtype=np.float64).copy()
    relative = state[2:4] - state[:2]
    normal = unit(relative)
    tangent = np.asarray([-normal[1], normal[0]], dtype=np.float64)
    before_xy = float(np.linalg.norm(state[2:4] - goal[2:4]))
    after_xy = float(np.linalg.norm(after[2:4] - goal[2:4]))
    before_angle = angle_error(state[4], goal[4])
    after_angle = angle_error(after[4], goal[4])
    action64 = np.asarray(action, dtype=np.float64)
    xy_limit = min(45.0, before_xy + 4.0)
    xy_limit_broad = 45.0
    return {
        "action": action64.tolist(),
        "action_norm": float(np.linalg.norm(action64)),
        "normal_component": float(np.dot(action64, normal)),
        "tangential_component": float(np.dot(action64, tangent)),
        "contact_distance": float(np.linalg.norm(state[:2] - state[2:4])),
        "block_dx": float(after[2] - state[2]),
        "block_dy": float(after[3] - state[3]),
        "block_xy_delta": float(np.linalg.norm(after[2:4] - state[2:4])),
        "angle_delta": float((after[4] - state[4] + np.pi) % (2.0 * np.pi) - np.pi),
        "before_xy": before_xy,
        "after_xy": after_xy,
        "before_angle": before_angle,
        "after_angle": after_angle,
        "angle_improved": bool(after_angle < before_angle - 1e-5),
        "angle_improvement": float(before_angle - after_angle),
        "inside_monotone_tube": bool(after_xy <= xy_limit),
        "inside_broad_tube": bool(after_xy <= xy_limit_broad),
        "feasible_monotone": bool(after_angle < before_angle - 1e-5 and after_xy <= xy_limit),
        "feasible_broad": bool(after_angle < before_angle - 1e-5 and after_xy <= xy_limit_broad),
        "terminated": bool(terminated or truncated),
    }


def bootstrap(values: np.ndarray, draws: int, seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sample = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "ci95_low": float(np.quantile(sample, 0.025)),
        "ci95_high": float(np.quantile(sample, 0.975)),
        "n": int(len(values)),
    }


def summarize(records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key = f"{record['contact_stratum']}__{record['distance_stratum']}"
        groups.setdefault(key, []).append(record)
    summary = {}
    for index, (key, rows) in enumerate(sorted(groups.items())):
        summary[key] = {
            metric: bootstrap(
                np.asarray([float(row[metric]) for row in rows]),
                args.bootstrap,
                args.seed + 100 + index * 17 + offset,
            )
            for offset, metric in enumerate((
                "any_feasible_monotone",
                "any_feasible_broad",
                "best_angle_improvement",
                "best_tangential_abs",
                "any_positive_angle_and_nonworsening_xy",
            ))
        }
        summary[key]["n_actions_per_state"] = int(rows[0]["n_actions"])
    return summary


def main() -> None:
    args = parse_args()
    if args.states_per_stratum < 1:
        raise ValueError("states-per-stratum must be positive")
    started = time.perf_counter()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    env = make_state_only_env()
    try:
        pool = collect_trajectory_data(env, args.pool_transitions, args.seed)
        candidates: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
            f"{contact}__{distance}": []
            for contact in CONTACT_STRATA for distance in DISTANCE_STRATA
        }
        for state in pool["states"]:
            contact = contact_stratum(state)
            if contact == "near_contact" and float(np.linalg.norm(state[2:4] - state[:2])) > 65.0:
                continue
            for distance in DISTANCE_STRATA:
                key = f"{contact}__{distance}"
                if len(candidates[key]) >= args.states_per_stratum:
                    continue
                distance_value = {"0_20": 10.0, "20_35": 27.0, "35_45": 40.0}[distance]
                candidates[key].append((
                    np.asarray(state, dtype=np.float64).copy(),
                    make_goal(state, distance_value, args.seed + len(candidates[key]) + len(key), args.angle_target),
                ))
            if all(len(value) >= args.states_per_stratum for value in candidates.values()):
                break
        missing = {key: len(value) for key, value in candidates.items() if len(value) < args.states_per_stratum}
        if missing:
            raise RuntimeError(f"insufficient genuine states for strata: {missing}")
        actions = fan_actions(args.directions, args.magnitudes)
        state_summaries = []
        for key in sorted(candidates):
            contact, distance = key.split("__", 1)
            for state_index, (state, goal) in enumerate(candidates[key]):
                probes = [
                    probe_state(env, state, goal, action, args.seed + state_index * 10000 + action_index)
                    for action_index, action in enumerate(actions)
                ]
                state_summaries.append({
                    "contact_stratum": contact,
                    "distance_stratum": distance,
                    "state_index": state_index,
                    "source_mode": mode_for_state(state),
                    "contact_distance": float(np.linalg.norm(state[:2] - state[2:4])),
                    "before_xy": float(np.linalg.norm(state[2:4] - goal[2:4])),
                    "before_angle": angle_error(state[4], goal[4]),
                    "n_actions": len(probes),
                    "any_feasible_monotone": float(any(row["feasible_monotone"] for row in probes)),
                    "any_feasible_broad": float(any(row["feasible_broad"] for row in probes)),
                    "best_angle_improvement": float(max(row["angle_improvement"] for row in probes)),
                    "best_tangential_abs": float(max(abs(row["tangential_component"]) for row in probes)),
                    "any_positive_angle_and_nonworsening_xy": float(any(
                        row["angle_improved"] and row["after_xy"] <= row["before_xy"] + args.xy_margin
                        for row in probes
                    )),
                    "probes": probes,
                })
    finally:
        env.close()
    report = {
        "claim_boundary": (
            "Privileged one-step simulator action-fan audit on genuine approach/contact states. "
            "It identifies local controllability and does not establish visual planning or task success."
        ),
        "protocol": {
            "environment": "swm/PushT-v1",
            "states_per_stratum": args.states_per_stratum,
            "pool_transitions": args.pool_transitions,
            "directions": args.directions,
            "magnitudes": args.magnitudes,
            "actions_per_state": int(len(fan_actions(args.directions, args.magnitudes))),
            "xy_margin": args.xy_margin,
            "angle_target": args.angle_target,
            "seed": args.seed,
            "wall_seconds": time.perf_counter() - started,
        },
        "summary": summarize(state_summaries, args),
        "records": state_summaries,
        "sanity_checks": {
            "all_finite": bool(all(
                np.isfinite(float(row[metric]))
                for row in state_summaries
                for metric in (
                    "contact_distance", "before_xy", "before_angle",
                    "best_angle_improvement", "best_tangential_abs",
                )
            )),
            "expected_state_count": len(candidates) * args.states_per_stratum,
            "actual_state_count": len(state_summaries),
            "all_action_fans_complete": bool(all(
                row["n_actions"] == len(actions) for row in state_summaries
            )),
        },
    }
    (output / "torque_effect_identifiability.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    lines = [
        "# PushT torque-effect identifiability audit",
        "",
        "| stratum | states | monotone-feasible | broad-feasible | positive angle + non-worsening XY | best angle improvement |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, item in sorted(report["summary"].items()):
        lines.append(
            "| %s | %d | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] | %.4f [%.4f, %.4f] |"
            % (
                key,
                item["any_feasible_monotone"]["n"],
                item["any_feasible_monotone"]["estimate"], item["any_feasible_monotone"]["ci95_low"], item["any_feasible_monotone"]["ci95_high"],
                item["any_feasible_broad"]["estimate"], item["any_feasible_broad"]["ci95_low"], item["any_feasible_broad"]["ci95_high"],
                item["any_positive_angle_and_nonworsening_xy"]["estimate"], item["any_positive_angle_and_nonworsening_xy"]["ci95_low"], item["any_positive_angle_and_nonworsening_xy"]["ci95_high"],
                item["best_angle_improvement"]["estimate"], item["best_angle_improvement"]["ci95_low"], item["best_angle_improvement"]["ci95_high"],
            )
        )
    lines += ["", report["claim_boundary"], ""]
    (output / "torque_effect_identifiability.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(output), "sanity_checks": report["sanity_checks"], "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
